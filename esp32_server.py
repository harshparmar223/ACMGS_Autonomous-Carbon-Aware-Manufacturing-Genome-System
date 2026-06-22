"""
ESP32 Real-Time Data Server
Receives sensor data from ESP32 via WiFi and processes with ACMGS models
"""

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, List
import logging
from datetime import datetime
import json
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import ACMGS modules (optional - wrapped in try/except)
# from prediction.predictor import Predictor
# from optimization.optimizer import Optimizer
# from energy_dna.model import EnergyDNAModel
# from carbon_scheduler.scheduler import CarbonScheduler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="ACMGS Real-Time ESP32 Server", version="1.0.0")

# CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== Data Models =====
class SensorData(BaseModel):
    """Sensor data from ESP32"""
    temperature: float
    humidity: float
    current: float
    timestamp: int
    rssi: Optional[int] = None
    ip: Optional[str] = None


class ModelPrediction(BaseModel):
    """Model prediction result"""
    energy_predicted: float
    carbon_impact: float
    optimization_score: float
    recommendation: str
    timestamp: str


# ===== Global State =====
sensor_buffer: List[Dict] = []
MAX_BUFFER_SIZE = 3600  # 1 hour at 1 reading/sec

# Model instances
predictor = None
optimizer = None
energy_model = None
scheduler = None

# WebSocket connections
active_connections: List[WebSocket] = []


# ===== Initialization =====
async def init_models():
    """Initialize ACMGS models on startup"""
    global predictor, optimizer, energy_model, scheduler
    
    try:
        logger.info("Initializing ACMGS models...")
        
        # Models currently disabled - implement when classes are available
        # try:
        #     predictor = Predictor()
        #     logger.info("✓ Predictor loaded")
        # except Exception as e:
        #     logger.warning(f"Could not load Predictor: {e}")
        
        # try:
        #     optimizer = Optimizer()
        #     logger.info("✓ Optimizer loaded")
        # except Exception as e:
        #     logger.warning(f"Could not load Optimizer: {e}")
        
        # try:
        #     energy_model = EnergyDNAModel()
        #     logger.info("✓ Energy DNA model loaded")
        # except Exception as e:
        #     logger.warning(f"Could not load Energy DNA model: {e}")
        
        # try:
        #     scheduler = CarbonScheduler()
        #     logger.info("✓ Carbon Scheduler loaded")
        # except Exception as e:
        #     logger.warning(f"Could not load Carbon Scheduler: {e}")
        
        logger.info("✓ Server ready to receive sensor data")
        
    except Exception as e:
        logger.error(f"Error initializing models: {e}")


@app.on_event("startup")
async def startup_event():
    """Initialize on server startup"""
    await init_models()


# ===== Real-Time Processing =====
async def process_sensor_data(data: SensorData) -> ModelPrediction:
    """
    Process sensor data through ACMGS models
    """
    # Store in buffer
    sensor_reading = {
        'temperature': data.temperature,
        'humidity': data.humidity,
        'current': data.current,
        'timestamp': datetime.fromtimestamp(data.timestamp / 1000).isoformat(),
        'rssi': data.rssi,
        'ip': data.ip
    }
    sensor_buffer.append(sensor_reading)
    
    # Keep buffer manageable
    if len(sensor_buffer) > MAX_BUFFER_SIZE:
        sensor_buffer.pop(0)
    
    # Energy calculation
    voltage = 230.0  # Assuming 230V AC
    power_watts = data.current * voltage
    
    # Get prediction
    energy_predicted = 0.0
    try:
        if predictor:
            features = {
                'current': data.current,
                'temperature': data.temperature,
                'humidity': data.humidity
            }
            # prediction = predictor.predict(features)
            energy_predicted = power_watts * 0.001  # kWh estimation
    except Exception as e:
        logger.warning(f"Prediction error: {e}")
    
    # Get optimization recommendation
    optimization_score = 0.0
    recommendation = "Monitor energy consumption"
    try:
        if optimizer:
            # Get current constraint
            if power_watts > 50000:  # 50kW threshold
                recommendation = "HIGH LOAD: Consider scheduling reduction"
                optimization_score = 0.3
            elif power_watts > 30000:  # 30kW threshold
                recommendation = "MEDIUM LOAD: Optimize if possible"
                optimization_score = 0.6
            else:
                recommendation = "LOW LOAD: Operating normally"
                optimization_score = 0.9
    except Exception as e:
        logger.warning(f"Optimization error: {e}")
    
    # Carbon impact (estimate)
    carbon_impact = power_watts * 0.233 / 1000  # kg CO2 per kWh (grid-dependent)
    
    prediction = ModelPrediction(
        energy_predicted=energy_predicted,
        carbon_impact=carbon_impact,
        optimization_score=optimization_score,
        recommendation=recommendation,
        timestamp=datetime.now().isoformat()
    )
    
    return prediction


async def broadcast_to_clients(message: Dict):
    """Send message to all connected WebSocket clients"""
    for connection in active_connections[:]:
        try:
            await connection.send_json(message)
        except Exception as e:
            logger.warning(f"Error broadcasting to client: {e}")
            active_connections.remove(connection)


# ===== API Endpoints =====
@app.post("/api/sensor-data")
async def receive_sensor_data(data: SensorData):
    """Receive sensor data from ESP32 and process"""
    try:
        logger.info(f"Received sensor data: Temp={data.temperature}°C, Current={data.current}A")
        
        # Process through models
        prediction = await process_sensor_data(data)
        
        # Prepare response
        response = {
            "status": "success",
            "sensor": {
                "temperature": data.temperature,
                "humidity": data.humidity,
                "current": data.current
            },
            "prediction": prediction.dict(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Alert if thresholds exceeded
        if data.current > 100:  # 100A threshold
            response["alert"] = "CRITICAL: Current exceeds safe limit"
            response["alert_level"] = "critical"
        elif data.temperature > 60:  # 60°C threshold
            response["alert"] = "WARNING: Temperature too high"
            response["alert_level"] = "warning"
        
        # Broadcast to WebSocket clients
        await broadcast_to_clients(response)
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing sensor data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_statistics(window: int = 60):
    """Get statistics over last N readings"""
    if len(sensor_buffer) < window:
        window = len(sensor_buffer)
    
    if window == 0:
        return {"error": "No data available"}
    
    recent = sensor_buffer[-window:]
    
    temps = [r['temperature'] for r in recent]
    currents = [r['current'] for r in recent]
    humidities = [r['humidity'] for r in recent]
    
    return {
        "window_size": window,
        "temperature": {
            "min": min(temps),
            "max": max(temps),
            "avg": sum(temps) / len(temps)
        },
        "current": {
            "min": min(currents),
            "max": max(currents),
            "avg": sum(currents) / len(currents),
            "total": sum(currents)
        },
        "humidity": {
            "min": min(humidities),
            "max": max(humidities),
            "avg": sum(humidities) / len(humidities)
        },
        "power_avg_watts": (sum(currents) / len(currents)) * 230.0
    }


@app.get("/api/latest")
async def get_latest_reading():
    """Get most recent sensor reading"""
    if not sensor_buffer:
        return {"error": "No data available"}
    
    latest = sensor_buffer[-1]
    return {
        **latest,
        "power_watts": latest['current'] * 230.0
    }


@app.post("/api/predict")
async def predict_energy(duration_hours: float = 1.0):
    """Predict energy consumption for next N hours"""
    if not sensor_buffer:
        return {"error": "No sensor data available"}
    
    # Use average current to extrapolate
    currents = [r['current'] for r in sensor_buffer[-60:]]  # Last minute
    avg_current = sum(currents) / len(currents) if currents else 0
    
    power_watts = avg_current * 230.0
    energy_kwh = (power_watts / 1000) * duration_hours
    carbon_kg = energy_kwh * 0.233  # Grid-dependent
    cost_usd = energy_kwh * 0.15  # Adjust based on local rates
    
    return {
        "duration_hours": duration_hours,
        "predicted_energy_kwh": energy_kwh,
        "predicted_carbon_kg": carbon_kg,
        "predicted_cost_usd": cost_usd,
        "assumptions": {
            "voltage_v": 230,
            "avg_current_a": avg_current,
            "carbon_intensity_kg_per_kwh": 0.233,
            "electricity_rate_usd_per_kwh": 0.15
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {
            "predictor": predictor is not None,
            "optimizer": optimizer is not None,
            "energy_model": energy_model is not None,
            "scheduler": scheduler is not None
        },
        "buffer_size": len(sensor_buffer),
        "timestamp": datetime.now().isoformat()
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data streaming"""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        logger.info(f"WebSocket client connected. Total: {len(active_connections)}")
        
        # Send initial buffer
        await websocket.send_json({
            "type": "init",
            "buffer": sensor_buffer[-100:],  # Last 100 readings
            "count": len(sensor_buffer),
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive - listen for disconnect or ping
        while True:
            try:
                # Wait for client message with timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Periodic keep-alive
                if len(sensor_buffer) > 0:
                    await websocket.send_json(sensor_buffer[-1])
            except Exception as e:
                logger.debug(f"WebSocket receive error: {e}")
                break
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    
    finally:
        active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(active_connections)}")


# ===== Main =====
if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting ACMGS Real-Time ESP32 Server on http://0.0.0.0:8001")
    logger.info("API docs: http://localhost:8001/docs")
    logger.info("WebSocket: ws://localhost:8001/ws")
    
    uvicorn.run(app, host="0.0.0.0", port=8001)
