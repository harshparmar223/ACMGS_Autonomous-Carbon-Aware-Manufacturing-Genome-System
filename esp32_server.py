"""
ACMGS v2.0 Real-Time Edge Telemetry & Surrogate Ingestion Server
Module: esp32_server.py

End-to-End Pipeline:
  1. Ingests streaming telemetry (DHT11 temp/humidity, ACS712 current) from ESP32 Node 1 or simulator.
  2. Maintains a rolling 128-point power/current waveform buffer.
  3. Evaluates Stage 1: PyTorch LSTM Autoencoder -> 16-D Energy DNA Latent Vector + Reconstruction Error.
  4. Assembles Stage 2: 25-D Batch Genome (Process 5 + Material 3 + Energy DNA 16 + Carbon 1).
  5. Evaluates Stage 3: XGBoost Surrogate Model -> Predicts Yield, Quality, Energy in <1ms.
  6. Evaluates Stage 4: Decision Engine -> Closed-loop MOSFET fan PWM (0-255) + Dual-Window Defect Abort.
  7. Computes Continuous Machine Health Index (0-100%) and TreeSHAP RCA diagnostics.
  8. Broadcasts live telemetry packets to WebSockets for Streamlit Cockpit display.
"""

import os
import sys
import json
import time
import asyncio
import logging
import sqlite3
import numpy as np
import torch
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import (
    DB_PATH, MODELS_DIR,
    ENERGY_INPUT_DIM, ENERGY_HIDDEN_DIM, ENERGY_LATENT_DIM, ENERGY_NUM_LAYERS,
    CARBON_HIGH_THRESHOLD, CARBON_LOW_THRESHOLD
)
from src.energy_dna.model import LSTMAutoencoder
from src.control.decision_engine import DecisionEngine, ActuationCommand
from src.intelligence.health_scorer import MachineHealthScorer, HealthTier
from src.intelligence.rca_engine import RCAEngine
from src.intelligence.golden_signature import GoldenSignatureEngine

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("esp32_server")

app = FastAPI(title="ACMGS v2.0 Edge Telemetry & Surrogate Brain Server", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== Telemetry Pydantic Models =====
class SensorData(BaseModel):
    temperature: float
    humidity: float
    current: float
    pressure: Optional[float] = 5.2
    speed: Optional[float] = 1850.0
    feed_rate: Optional[float] = 0.85
    carbon_intensity: Optional[float] = 250.0
    timestamp: Optional[int] = None
    rssi: Optional[int] = None
    ip: Optional[str] = None


class ActuationRequest(BaseModel):
    pwm: int
    feed_hold: bool = False
    source: str = "MANUAL_DASHBOARD"


# ===== Global System State =====
class EdgeBrainState:
    def __init__(self):
        self.raw_buffer: List[Dict[str, Any]] = []
        self.waveform_128: deque = deque(maxlen=128)
        self.lstm_model: Optional[LSTMAutoencoder] = None
        self.predictor_model: Any = None
        self.decision_engine: DecisionEngine = DecisionEngine()
        self.health_scorer: MachineHealthScorer = MachineHealthScorer()
        self.rca_engine: RCAEngine = RCAEngine()
        self.golden_engine: GoldenSignatureEngine = GoldenSignatureEngine()
        
        self.latest_telemetry: Dict[str, Any] = {}
        self.latest_genome: Optional[np.ndarray] = None
        self.latest_prediction: Dict[str, float] = {"yield": 0.95, "quality": 0.94, "energy": 125.0}
        self.latest_health: Dict[str, Any] = {}
        self.latest_actuation: Dict[str, Any] = {}
        self.active_websockets: List[WebSocket] = []
        
        # Warmup buffer with nominal 128 current points
        for _ in range(128):
            self.waveform_128.append(12.5 + np.random.normal(0, 0.2))


state = EdgeBrainState()


def load_ai_models():
    """Loads PyTorch LSTM Autoencoder and XGBoost Predictor."""
    # 1. Load LSTM Autoencoder
    lstm_path = os.path.join(MODELS_DIR, "lstm_autoencoder.pth")
    if os.path.exists(lstm_path):
        try:
            model = LSTMAutoencoder(
                input_dim=ENERGY_INPUT_DIM,
                hidden_dim=ENERGY_HIDDEN_DIM,
                latent_dim=ENERGY_LATENT_DIM,
                num_layers=ENERGY_NUM_LAYERS
            )
            model.load_state_dict(torch.load(lstm_path, map_location=torch.device("cpu"), weights_only=True))
            model.eval()
            state.lstm_model = model
            logger.info("✓ PyTorch LSTM Autoencoder loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load LSTM Autoencoder: {e}")

    # 2. Load XGBoost Predictor
    pred_path = os.path.join(MODELS_DIR, "predictor.pkl")
    if os.path.exists(pred_path):
        try:
            with open(pred_path, "rb") as f:
                state.predictor_model = pickle.load(f)
            logger.info("✓ XGBoost Multi-Target Surrogate loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load XGBoost Predictor: {e}")


@app.on_event("startup")
async def on_startup():
    load_ai_models()
    logger.info("ACMGS v2.0 Edge Brain Server ready on port 8000.")


def evaluate_stream_inference(data: SensorData) -> Dict[str, Any]:
    """
    Executes the complete v2.0 micro-speed inference pipeline (<1ms).
    """
    state.waveform_128.append(data.current)
    waveform = np.array(list(state.waveform_128), dtype=np.float32)

    # 1. Stage 1: PyTorch LSTM Autoencoder -> 16-D Latent + Recon Error
    recon_error = 0.045
    latent_vector = np.zeros(16, dtype=np.float32)
    
    if state.lstm_model is not None and len(waveform) == 128:
        try:
            # Normalize waveform
            w_mean = waveform.mean()
            w_std = waveform.std()
            w_std = 1.0 if w_std == 0 else w_std
            norm_w = (waveform - w_mean) / w_std
            
            tensor_in = torch.tensor(norm_w.reshape(1, 128, 1), dtype=torch.float32)
            with torch.no_grad():
                reconstructed, latent = state.lstm_model(tensor_in)
                err = ((tensor_in - reconstructed) ** 2).mean().item()
                recon_error = float(err)
                latent_vector = latent.cpu().numpy().flatten()
        except Exception as e:
            logger.debug(f"LSTM Autoencoder error: {e}")
    else:
        # Calibrated analytical estimate based on current variance
        var = float(np.var(waveform))
        recon_error = float(0.035 + min(0.35, var * 0.05))

    # 2. Stage 2: 25-D Batch Genome Assembly
    # [Temp, Press, Speed, Feed, Hum, Density, Hardness, Grade, z01..z16, Carbon]
    density = 7.85
    hardness = 200.0
    grade = 2.0
    carbon_int = float(data.carbon_intensity or 250.0)
    
    genome_25d = np.array([
        data.temperature, data.pressure or 5.2, data.speed or 1850.0, data.feed_rate or 0.85, data.humidity,
        density, hardness, grade,
        *latent_vector,
        carbon_int
    ], dtype=np.float32)
    state.latest_genome = genome_25d

    # 3. Stage 3: XGBoost Surrogate Model (<1ms)
    pred_yield = 0.952
    pred_quality = 0.948
    pred_energy = 126.5
    
    if state.predictor_model is not None:
        try:
            preds = state.predictor_model.predict(genome_25d.reshape(1, -1))
            if preds.ndim == 2:
                pred_yield = float(np.clip(preds[0, 0], 0.0, 1.0))
                pred_quality = float(np.clip(preds[0, 1], 0.0, 1.0))
                pred_energy = float(max(10.0, preds[0, 2]))
            else:
                pred_yield = float(np.clip(preds[0], 0.0, 1.0))
                pred_quality = float(np.clip(preds[1], 0.0, 1.0))
                pred_energy = float(max(10.0, preds[2]))
        except Exception as e:
            logger.debug(f"XGBoost Surrogate error: {e}")
    else:
        # Physics approximation
        temp_penalty = max(0.0, (data.temperature - 55.0) * 0.015)
        recon_penalty = max(0.0, (recon_error - 0.199) * 1.2)
        pred_quality = float(np.clip(0.96 - temp_penalty - recon_penalty, 0.15, 0.99))
        pred_yield = float(np.clip(0.97 - temp_penalty * 0.8 - recon_penalty * 0.9, 0.20, 0.99))
        pred_energy = float(data.current * 230.0 * 0.04)

    state.latest_prediction = {
        "yield": round(pred_yield, 4),
        "quality": round(pred_quality, 4),
        "energy": round(pred_energy, 1)
    }

    # 4. Stage 4: Decision Engine Closed-Loop & Dual-Window Guardrail
    act_cmd = state.decision_engine.evaluate_telemetry(
        temperature=data.temperature,
        current_rms=data.current,
        recon_error=recon_error,
        predicted_quality=pred_quality,
        predicted_yield=pred_yield,
        carbon_intensity=carbon_int
    )
    state.latest_actuation = {
        "pwm_duty": act_cmd.pwm_duty,
        "fan_speed_pct": act_cmd.fan_speed_pct,
        "feed_hold": act_cmd.feed_hold_asserted,
        "status": act_cmd.status,
        "sunk_energy_saved_kwh": act_cmd.sunk_energy_saved_kwh,
        "sunk_carbon_avoided_kg": act_cmd.sunk_carbon_avoided_kg,
        "reason": act_cmd.reason,
        "windows_confirmed": act_cmd.windows_confirmed
    }

    # 5. Continuous Machine Health Scorer (0-100%)
    health_rep = state.health_scorer.evaluate(
        recon_error=recon_error,
        current_rms=data.current,
        temperature=data.temperature
    )
    state.latest_health = {
        "health_index": health_rep.health_index,
        "tier": health_rep.tier.value,
        "recon_penalty": health_rep.recon_penalty,
        "current_penalty": health_rep.current_penalty,
        "temp_penalty": health_rep.temp_penalty,
        "status_summary": health_rep.status_summary,
        "action_recommendation": health_rep.action_recommendation,
        "color_hex": health_rep.color_hex
    }

    # 6. Assemble Full Telemetry Response
    resp = {
        "timestamp": datetime.now().isoformat(),
        "sensor": {
            "temperature": data.temperature,
            "humidity": data.humidity,
            "current": data.current,
            "pressure": data.pressure,
            "speed": data.speed,
            "feed_rate": data.feed_rate,
            "carbon_intensity": carbon_int
        },
        "energy_dna": {
            "recon_error": round(recon_error, 6),
            "threshold": 0.199084,
            "latent_vector_sample": [round(float(z), 3) for z in latent_vector[:4]],
            "is_anomalous": recon_error > 0.199084
        },
        "predictions": state.latest_prediction,
        "actuation": state.latest_actuation,
        "health": state.latest_health
    }
    state.latest_telemetry = resp
    return resp


# ===== REST API Endpoints =====

@app.post("/api/sensor-data")
async def post_sensor_data(data: SensorData):
    """Primary telemetry ingestion endpoint from Node 1 (ESP32) or simulator."""
    result = evaluate_stream_inference(data)
    
    # Broadcast to WebSockets asynchronously
    if state.active_websockets:
        asyncio.create_task(broadcast_telemetry(result))
        
    return result


@app.get("/api/telemetry/latest")
async def get_latest_telemetry():
    """Returns the most recent processed telemetry state."""
    if not state.latest_telemetry:
        # Return initialized state
        dummy = SensorData(temperature=38.5, humidity=48.0, current=12.2)
        return evaluate_stream_inference(dummy)
    return state.latest_telemetry


@app.get("/api/health")
async def get_machine_health():
    """Returns continuous Machine Health Index and operational tier."""
    if not state.latest_health:
        dummy = SensorData(temperature=38.5, humidity=48.0, current=12.2)
        evaluate_stream_inference(dummy)
    return state.latest_health


@app.get("/api/rca")
async def get_rca_explanation(target_idx: int = 0):
    """Computes TreeSHAP Explainable RCA on current batch genome."""
    if state.latest_genome is None:
        dummy = SensorData(temperature=42.0, humidity=45.0, current=13.0)
        evaluate_stream_inference(dummy)
        
    report = state.rca_engine.explain(
        genome_vector=state.latest_genome,
        target_index=target_idx,
        recon_error=state.latest_telemetry.get("energy_dna", {}).get("recon_error", 0.05)
    )
    return {
        "target": report.target_name,
        "predicted_value": report.predicted_value,
        "baseline_value": report.baseline_value,
        "primary_driver": report.primary_driver_text,
        "plain_english": report.plain_english_diagnosis,
        "top_attributions": [
            {
                "feature": a.display_name,
                "value": a.feature_value,
                "attribution_pct": a.attribution_pct,
                "direction": a.direction,
                "interpretation": a.interpretation
            }
            for a in report.top_attributions
        ]
    }


@app.get("/api/golden-signature")
async def get_golden_prescription(temp: float = 48.0, pressure: float = 4.8, speed: float = 1700.0, feed: float = 0.78):
    """Computes nearest-neighbor Golden Signature recipe deltas."""
    rec = state.golden_engine.find_nearest_golden_recipe(
        temperature=temp,
        pressure=pressure,
        speed=speed,
        feed_rate=feed
    )
    return {
        "nearest_batch_id": rec.nearest_batch_id,
        "target_yield": rec.target_yield,
        "target_quality": rec.target_quality,
        "target_energy": rec.target_energy,
        "deltas": rec.deltas,
        "prescriptive_text": rec.prescriptive_text,
        "similarity_score_pct": rec.similarity_score_pct
    }


@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """WebSocket stream for real-time dashboard visualization."""
    await websocket.accept()
    state.active_websockets.append(websocket)
    try:
        if state.latest_telemetry:
            await websocket.send_json(state.latest_telemetry)
        while True:
            # Keep-alive loop
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in state.active_websockets:
            state.active_websockets.remove(websocket)


async def broadcast_telemetry(payload: Dict[str, Any]):
    """Broadcasts message to all active WebSocket clients."""
    for ws in state.active_websockets[:]:
        try:
            await ws.send_json(payload)
        except Exception:
            if ws in state.active_websockets:
                state.active_websockets.remove(ws)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
