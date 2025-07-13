from pydantic import BaseModel

class TrainRequest(BaseModel):
    roomId: int
    simulationId: int
    HIDDEN_SIZE: int
    NUM_LAYERS: int
    SEQ_LENGTH: int
    batch_size: int
    epochs_number: int