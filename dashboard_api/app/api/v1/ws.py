from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session
import asyncio

from app.db.session import get_db
from app.services.ws_manager import manager
from app.models.alert import Alert

router = APIRouter()


@router.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    await manager.connect(websocket)

    try:
        while True:
            await asyncio.sleep(2)

            # 🔥 простая реализация — берем последние alerts
            db: Session = next(get_db())

            alerts = (
                db.query(Alert)
                .order_by(Alert.created_at.desc())
                .limit(10)
                .all()
            )

            data = [
                {
                    "id": a.id,
                    "device_id": a.device_id,
                    "severity": a.severity,
                    "message": a.message,
                }
                for a in alerts
            ]

            await websocket.send_json(data)

    except WebSocketDisconnect:
        manager.disconnect(websocket)