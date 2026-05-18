from fastapi import FastAPI
from fastapi.concurrency import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import api_router
from contextlib import asynccontextmanager
from app.api.v1 import alerts

from app.db.session import SessionLocal
from app.services.bootstrap import create_default_admin

@asynccontextmanager
async def lifespan(app: FastAPI):
    db = SessionLocal()

    try:
        create_default_admin(db)
    finally:
        db.close()

    yield


app = FastAPI(
    title="IIoT Dashboard API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(api_router)
app.include_router(alerts.router)

@app.get("/health")
def health():
    return {"status": "ok"}