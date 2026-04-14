import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.prediction import Prediction

router = APIRouter()


@router.websocket("/ws/predictions")
async def websocket_predictions(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            await asyncio.sleep(2)

            db: Session = next(get_db())

            predictions = (
                db.query(Prediction)
                .order_by(Prediction.created_at.desc())
                .limit(10)
                .all()
            )

            data = [
                {
                    "id": p.id,
                    "device_id": p.device_id,
                    "machine_id": p.machine_id,
                    "prediction": p.prediction,
                    "risk_score": p.risk_score,
                    "risk_level": p.risk_level,
                }
                for p in predictions
            ]

            await websocket.send_json(data)

    except WebSocketDisconnect:
        print("Prediction WS disconnected")