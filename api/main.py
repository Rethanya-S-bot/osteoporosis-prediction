"""FastAPI entrypoint for osteoporosis risk prediction."""
from fastapi import FastAPI, Query

from api.routers.prediction import router as prediction_router
from src.db import fetch_predictions


app = FastAPI(
    title="Osteoporosis Prediction API",
    version="0.1.0",
    description="Research prototype for osteoporosis risk estimation; not a diagnostic tool.",
)
app.include_router(prediction_router)


@app.get("/history")
def history(limit: int = Query(default=50, ge=1, le=200)) -> list[dict]:
    """Return prediction history, newest first."""
    return fetch_predictions(limit)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}