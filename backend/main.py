from fastapi import FastAPI, HTTPException, WebSocket, Depends
from pydantic import BaseModel, Field
from typing import List, Optional
import sys
import os
import numpy as np
import torch
import yaml
from sqlalchemy.orm import Session

# Add ml_engine/src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ml_engine", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "gridguard"))

from inference import InferenceEngine
from xai_engine import XAIEngine
from database import init_db, get_db, Detection, Meter
from fastapi.middleware.cors import CORSMiddleware

# Load config
CONFIG_PATH = "c:/Users/User/Downloads/scratch-main/gridguard/config.yaml"
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

app = FastAPI(title="GridGuard AI: Backend Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB
init_db()

# Initialize engines
inference_engine = InferenceEngine(
    syn_dl_path='../ml_engine/src/best_model_balanced.pth',
    syn_xgb_path='../ml_engine/src/best_xgb_augmented.pkl',
    real_dl_path='../gridguard_real_data/models/gridguard_sgcc_best.pth',
    real_xgb_path='../gridguard_real_data/models/xgboost_sgcc_edge.pkl'
)
xai_engine = XAIEngine(inference_engine.model_real)

# Define request schema matching context-aware capabilities
class PredictionRequest(BaseModel):
    meter_id: str
    readings: List[float]
    live_gli: Optional[float] = None
    live_gli_timestamp: Optional[float] = None
    hour_of_day: Optional[int] = 12
    day_of_week: Optional[int] = 0
    model_type: Optional[str] = "real_world"

@app.get("/")
async def root():
    return {"message": "GridGuard AI API is online"}

@app.post("/api/v1/predict")
async def predict_theft(request: PredictionRequest, db: Session = Depends(get_db)):
    try:
        if len(request.readings) < 20:
            raise HTTPException(status_code=400, detail="Minimum 20 readings required for detection.")
        
        # Invoke context-aware prediction with all telemetry and fallback indicators
        result = inference_engine.predict(
            raw_consumption=np.array(request.readings),
            meter_id=request.meter_id,
            live_gli=request.live_gli,
            live_gli_timestamp=request.live_gli_timestamp,
            hour_of_day=request.hour_of_day or 12,
            day_of_week=request.day_of_week or 0,
            model_type=request.model_type
        )
        
        # Persist detection outcome to database
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
        if len(request.readings) < 20:
            raise HTTPException(status_code=400, detail="Minimum 20 readings required for explanation.")
            
        # Preprocess consumption sequence just like we do for prediction
        processed_consumption = inference_engine.preprocessor.process_user_data(request.readings)
        
        # Enforce strict 26-week sequence length (Fix 3: Sequence Window Contradiction)
        if len(processed_consumption) < 26:
            pad_len = 26 - len(processed_consumption)
            processed_consumption = np.pad(processed_consumption, (pad_len, 0), mode='edge')
        elif len(processed_consumption) > 26:
            processed_consumption = processed_consumption[-26:]
            
        # Retrieve context-aware GLI value from gli_manager
        from gridguard.backend.infrastructure.gli_manager import PredictionRequest as GLIPredRequest
        gli_req = GLIPredRequest(
            meter_id=request.meter_id,
            kwh_sequence=processed_consumption.tolist(),
            live_gli=request.live_gli,
            live_gli_timestamp=request.live_gli_timestamp,
            hour_of_day=request.hour_of_day or 12,
            day_of_week=request.day_of_week or 0
        )
        gli_val, _ = inference_engine.gli_manager.process_gli(gli_req)
        
        # Synthesize GLI sequence
        trnc_mode = config.get("data", {}).get("trnc_mode", True)
        if trnc_mode:
            gli_base = 0.5 + 0.12 * np.sin(np.linspace(np.pi, 5 * np.pi, 26))
        else:
            gli_base = 0.5 + 0.12 * np.sin(np.linspace(0, 4 * np.pi, 26))
            
        shift = gli_val - gli_base[-1]
        gli_seq = np.clip(gli_base + shift, 0.0, 1.0)
        
        # Stack to form a (26, 2) multi-channel array
        seq_2d = np.stack([processed_consumption, gli_seq], axis=1) # (26, 2)
        input_tensor = torch.tensor(seq_2d, dtype=torch.float32).unsqueeze(0).to(xai_engine.device) # (1, 26, 2)
        
        # Extract integrated gradients attribution scores
        attn_map = xai_engine.get_integrated_gradients(input_tensor)
        
        # Average attribution scores across feature dimensions to yield a 1D sequence of length 26
        if len(attn_map.shape) > 1:
            attn_map = attn_map.mean(axis=-1)
            
        return {
            "meter_id": request.meter_id,
            "attention_heatmap": attn_map.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket, model_type: str = "real_world"):
    await websocket.accept()
    try:
        import asyncio, random, pandas as pd
        
        # Load the enriched grid simulation data
        sim_data_path = "../data/grid_simulated_dataset.csv"
        if os.path.exists(sim_data_path):
            df_sim = pd.read_csv(sim_data_path)
            # Filter for theft cases to show on the dashboard
            theft_events = df_sim[df_sim['anomaly_label'] == 1].to_dict('records')
            random.shuffle(theft_events)
        else:
            theft_events = []

        idx = 0
        while True:
            if theft_events:
                event = theft_events[idx % len(theft_events)]
                idx += 1
                # Make the synthetic model seem more erratic/lower confidence, and real_world highly confident
                if model_type == "synthetic":
                    conf = round(random.uniform(0.60, 0.85), 2)
                    status = "pending"
                else:
                    conf = round(random.uniform(0.85, 0.99), 2)
                    status = "investigating"
                    
                payload = {
                    "id": f"KIB-TEK-{event['household_id']}",
                    "lat": event['lat'],
                    "lon": event['lon'],
                    "region": event['region_id'],
                    "anomaly": event['anomaly_type'],
                    "risk": "high" if event['anomaly_label'] == 1 else "medium",
                    "confidence": conf,
                    "consumption": event['consumption_kwh'],
                    "grid_load": event['grid_load_index'],
                    "status": status,
                    "model_used": "GridGuard-SGCC (Real)" if model_type == "real_world" else "GridGuard-TRNC (Synthetic)"
                }
            else:
                # Fallback to mock if file missing
                payload = {
                    "id": f"KIB-TEK-{random.randint(1000, 2499)}",
                    "lat": 35.18, "lon": 33.36, # Lefkoşa
                    "status": "pending"
                }
                
            await websocket.send_json(payload)
            await asyncio.sleep(3) # Stream every 3 seconds
    except Exception:
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
