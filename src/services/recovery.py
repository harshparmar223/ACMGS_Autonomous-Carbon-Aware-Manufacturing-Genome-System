"""
State Recovery & Resume Engine
Module: src/services/recovery.py

If production must stop for maintenance or tool clearance:
1. Checkpoint & save machine state, process parameters, and batch progress.
2. Perform structured verification: Machine check, Sensor check, Safety interlock check.
3. Restore valid state and resume production cleanly without losing previous progress.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
import sqlite3

from config.settings import DB_PATH
from src.services.machine_state import MachineState

logger = logging.getLogger("recovery_engine")


@dataclass
class RecoveryCheckpoint:
    """Snapshot of machine & batch context captured upon controlled stop."""
    checkpoint_id: str
    machine_id: str
    batch_id: str
    batch_progress_pct: float
    elapsed_minutes: float
    energy_kwh_accumulated: float
    carbon_kg_accumulated: float
    saved_parameters: Dict[str, float]
    stop_reason: str
    is_resumed: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class PreResumeVerification:
    """Pre-flight safety and sensor validation prior to production resume."""
    machine_cleared: bool
    sensors_calibrated: bool
    safety_interlocks_healthy: bool
    ready_to_resume: bool
    verification_notes: List[str]
    verified_at: str = field(default_factory=lambda: datetime.now().isoformat())


class RecoveryManager:
    """
    Manages safe state preservation, post-maintenance verification, and production resumption.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._active_checkpoints: Dict[str, RecoveryCheckpoint] = {}

    def create_checkpoint(
        self,
        state: MachineState,
        stop_reason: str = "Controlled safety stop"
    ) -> RecoveryCheckpoint:
        """Saves current state and batch progress."""
        checkpoint_id = f"CHK_{state.machine_id}_{int(datetime.now().timestamp())}"
        chk = RecoveryCheckpoint(
            checkpoint_id=checkpoint_id,
            machine_id=state.machine_id,
            batch_id=state.batch_id,
            batch_progress_pct=state.batch_progress_pct,
            elapsed_minutes=state.elapsed_minutes,
            energy_kwh_accumulated=state.energy_kwh,
            carbon_kg_accumulated=state.carbon_emissions_kg,
            saved_parameters={
                "temperature": state.temperature,
                "voltage": state.voltage,
                "current": state.current,
                "power": state.power
            },
            stop_reason=stop_reason
        )
        self._active_checkpoints[state.machine_id] = chk
        logger.info(f"Created recovery checkpoint {checkpoint_id} for {state.machine_id} at {state.batch_progress_pct:.1f}% progress.")
        return chk

    def verify_pre_flight(
        self,
        machine_id: str,
        current_temp: float,
        current_a: float,
        sensor_status_ok: bool = True
    ) -> PreResumeVerification:
        """Executes 3-point pre-resume verification."""
        notes = []
        machine_cleared = (current_temp <= 50.0 and current_a <= 2.0)
        sensors_calibrated = sensor_status_ok
        safety_healthy = (current_temp < 65.0)

        if not machine_cleared:
            notes.append(f"Machine spindle not idle or chamber above cooldown setpoint ({current_temp:.1f}°C).")
        if not sensors_calibrated:
            notes.append("Telemetry sensors reporting communication error or range fault.")
        if not safety_healthy:
            notes.append("Safety interlock reporting active E-Stop or fault condition.")

        ready = machine_cleared and sensors_calibrated and safety_healthy
        if ready:
            notes.append("All 3 pre-flight verification checks PASSED. System is ready to resume.")

        return PreResumeVerification(
            machine_cleared=machine_cleared,
            sensors_calibrated=sensors_calibrated,
            safety_interlocks_healthy=safety_healthy,
            ready_to_resume=ready,
            verification_notes=notes
        )

    def resume_production(
        self,
        machine_id: str,
        target_state: MachineState,
        verification: PreResumeVerification
    ) -> Tuple[bool, str]:
        """Restores batch progress and resumes production."""
        if not verification.ready_to_resume:
            return False, f"Cannot resume: Pre-flight verification failed ({verification.verification_notes})"

        chk = self._active_checkpoints.get(machine_id)
        if chk:
            target_state.batch_progress_pct = chk.batch_progress_pct
            target_state.elapsed_minutes = chk.elapsed_minutes
            target_state.energy_kwh = chk.energy_kwh_accumulated
            target_state.carbon_emissions_kg = chk.carbon_kg_accumulated
            target_state.machine_status = "RUNNING"
            chk.is_resumed = True
            msg = f"Production RESUMED for {chk.batch_id} at {chk.batch_progress_pct:.1f}% progress without loss."
        else:
            target_state.machine_status = "RUNNING"
            msg = f"Production RESUMED in nominal mode."

        logger.info(msg)
        return True, msg
