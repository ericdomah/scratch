from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
import numpy as np
import torch
# Add ml_engine/src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ml_engine", "src"))

from inference import InferenceEngine
from xai_engine import XAIEngine

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="GridGuard AI: Backend Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from database import init_db, get_db, Detection, Meter
from sqlalchemy.orm import Session
from fastapi import Depends

# Initialize DB
init_db()

# Initialize engines
inference_engine = InferenceEngine(model_path='../ml_engine/src/best_model.pth')
xai_engine = XAIEngine(inference_engine.model)

class PredictionRequest(BaseModel):
    meter_id: str
    readings: list[float]

@app.get("/")
async def root():
    return {"message": "GridGuard AI API is online"}

@app.post("/api/v1/predict")
async def predict_theft(request: PredictionRequest, db: Session = Depends(get_db)):
    try:
        if len(request.readings) < 20:
            raise HTTPException(status_code=400, detail="Minimum 20 readings required for detection.")
        
        result = inference_engine.predict(np.array(request.readings))
        
        # Persist to DB
        new_detection = Detection(
            meter_id=request.meter_id,
            is_theft=result["is_theft"],
            confidence=result["confidence"]
        )
        db.add(new_detection)
        db.commit()
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/history")
async def get_history(db: Session = Depends(get_db)):
    detections = db.query(Detection).order_by(Detection.timestamp.desc()).limit(50).all()
    return detections

@app.post("/api/v1/explain")
async def explain_theft(request: PredictionRequest):
    try:
        readings = np.array(request.readings)
        # Prepare for attention extraction
        # (1, seq_len, 1)
        input_tensor = torch.tensor(readings[:20], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        attn_map = xai_engine.get_attention_map(input_tensor)
        
        return {
            "meter_id": request.meter_id,
            "attention_heatmap": attn_map.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
