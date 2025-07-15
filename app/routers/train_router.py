from fastapi import APIRouter, BackgroundTasks
from app.schemas.train import TrainRequest
from app.services.train_service import train_sync

router = APIRouter()

@router.post("/train")
async def train(
    req: TrainRequest,
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(train_sync, req)
    return {
        "status": "training_started",
        "roomId": req.roomId,
        "simulationId": req.simulationId,
    }