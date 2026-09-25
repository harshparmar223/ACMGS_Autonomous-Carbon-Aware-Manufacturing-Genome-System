"""
Unified Data Contract & Real-Time Machine State (Single Source of Truth)
Module: src/services/machine_state.py

Provides a centralized MachineState object read by all ACMGS modules to guarantee
consistency across Ingestion, Prediction, Digital Twin, Optimization, and Safety.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import numpy as np
import json
import logging

logger = logging.getLogger("machine_state")


@dataclass
class SensorReadingContract:
    """Standardized IoT telemetry message contract."""
    device_id: str
    machine_id: str
    node_type: str  # e.g. "SENSOR_NODE", "ACTUATOR_NODE"
    timestamp: str
    temperature: float
    humidity: float
    voltage: float
    current: float
    power: float
    units: Dict[str, str] = field(default_factory=lambda: {
        "temperature": "°C",
        "humidity": "%",
        "voltage": "V",
        "current": "A",
        "power": "W"
    })

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SensorReadingContract":
        """Validates and parses raw dictionary into standardized contract."""
        # Sanity & range validation
        temp = float(data.get("temperature", 25.0))
        hum = float(data.get("humidity", 50.0))
        volt = float(data.get("voltage", 230.0))
        curr = float(data.get("current", 0.0))
        
        # Calculate P = V * I if power is missing
        power = float(data.get("power", volt * curr))
        
        # Catch impossible sensor values
        if temp < -20.0 or temp > 500.0 or np.isnan(temp):
            logger.warning(f"Invalid temperature {temp}°C, clamping.")
            temp = max(0.0, min(500.0, temp))
        if curr < 0.0 or curr > 200.0 or np.isnan(curr):
            logger.warning(f"Invalid current {curr}A, clamping.")
            curr = max(0.0, min(200.0, curr))

        return cls(
            device_id=str(data.get("device_id", "ESP32_DEFAULT")),
            machine_id=str(data.get("machine_id", "MACHINE_01")),
            node_type=str(data.get("node_type", "SENSOR_NODE")),
            timestamp=str(data.get("timestamp", datetime.now().isoformat())),
            temperature=temp,
            humidity=hum,
            voltage=volt,
            current=curr,
            power=power,
            units=data.get("units", {
                "temperature": "°C", "humidity": "%", "voltage": "V", "current": "A", "power": "W"
            })
        )


@dataclass
class MachineState:
    """
    Centralized, unified machine state object.
    Every ACMGS component reads from this state to prevent data discrepancies.
    """
    machine_id: str
    temperature: float
    humidity: float
    voltage: float
    current: float
    power: float
    energy_kwh: float
    carbon_emissions_kg: float
    
    # AI & Metrology Layers
    energy_dna_embedding: List[float] = field(default_factory=lambda: [0.0] * 16)
    anomaly_score: float = 0.0
    anomaly_status: str = "NORMAL"  # NORMAL, WARNING, CRITICAL
    health_index: float = 100.0
    health_status: str = "HEALTHY"  # HEALTHY (80-100), WATCH (60-79), AT RISK (<60)
    quality_score: float = 0.95
    quality_risk: float = 0.05
    golden_similarity: float = 95.0
    maintenance_risk: str = "LOW"   # LOW, MEDIUM, HIGH
    carbon_intensity: float = 250.0 # gCO2/kWh
    machine_status: str = "RUNNING" # RUNNING, CORRECTING, CONTROLLED_STOP, RESUMING, IDLE
    batch_id: str = "BATCH_1042"
    batch_progress_pct: float = 45.0
    elapsed_minutes: float = 27.0
    total_planned_minutes: float = 60.0
    
    # Distinguish source of data
    data_mode: str = "REAL_SENSOR_DATA" # REAL_SENSOR_DATA, SIMULATED_SCENARIO, MODEL_PREDICTION
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "machine_id": self.machine_id,
            "temperature": self.temperature,
            "humidity": self.humidity,
            "voltage": self.voltage,
            "current": self.current,
            "power": self.power,
            "energy_kwh": self.energy_kwh,
            "carbon_emissions_kg": self.carbon_emissions_kg,
            "energy_dna_embedding": self.energy_dna_embedding,
            "anomaly_score": self.anomaly_score,
            "anomaly_status": self.anomaly_status,
            "health_index": self.health_index,
            "health_status": self.health_status,
            "quality_score": self.quality_score,
            "quality_risk": self.quality_risk,
            "golden_similarity": self.golden_similarity,
            "maintenance_risk": self.maintenance_risk,
            "carbon_intensity": self.carbon_intensity,
            "machine_status": self.machine_status,
            "batch_id": self.batch_id,
            "batch_progress_pct": self.batch_progress_pct,
            "elapsed_minutes": self.elapsed_minutes,
            "total_planned_minutes": self.total_planned_minutes,
            "data_mode": self.data_mode,
            "last_updated": self.last_updated
        }

    def update_telemetry(self, reading: SensorReadingContract, delta_t_hours: float = 0.000138):
        """Updates live electrical & thermal parameters."""
        self.temperature = reading.temperature
        self.humidity = reading.humidity
        self.voltage = reading.voltage
        self.current = reading.current
        self.power = reading.power
        
        # Incremental energy integration: Energy (kWh) = sum(P (kW) * dt)
        p_kw = self.power / 1000.0
        self.energy_kwh += p_kw * delta_t_hours
        self.carbon_emissions_kg = self.energy_kwh * (self.carbon_intensity / 1000.0)
        self.last_updated = datetime.now().isoformat()
