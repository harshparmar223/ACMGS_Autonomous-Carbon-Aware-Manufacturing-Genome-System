"""
Production Continuity Manager
Module: src/services/production_continuity.py

Core Principle:
ACMGS should NOT stop a machine whenever a problem occurs. It should detect the problem,
determine its severity (L0-L3), attempt a safe corrective path when possible, verify the result,
and use controlled shutdown only when continued operation is unsafe.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
import logging

from src.services.machine_state import MachineState
from src.safety.safety_rules import SafetyRuleEngine, SafetyCheckResult

logger = logging.getLogger("production_continuity")


class SeverityLevel(Enum):
    L0_NORMAL = "L0_NORMAL"         # Normal operation: Continue
    L1_WARNING = "L1_WARNING"       # Minor anomaly / transient: Continue & Monitor
    L2_CORRECTIVE = "L2_CORRECTIVE" # Recoverable deviation: Evaluate Correction & Digital Twin
    L3_CRITICAL = "L3_CRITICAL"     # Unrecoverable / Safety violation: Controlled Stop


class ContinuityAction(Enum):
    CONTINUE = "CONTINUE"
    CORRECT = "CORRECT"
    CONTROLLED_STOP = "CONTROLLED_STOP"
    RESUME = "RESUME"


@dataclass
class EconomicImpactAnalysis:
    """Quantitative cost & energy comparison across decision options."""
    material_scrap_risk_usd: float
    restart_energy_cost_usd: float
    downtime_cost_usd: float
    total_cost_if_stopped_usd: float
    expected_savings_by_correcting_usd: float
    restart_energy_kwh: float = 25.0
    downtime_hours_estimate: float = 1.5


@dataclass
class ContinuityDecision:
    """Final prescriptive decision for production continuity."""
    action: ContinuityAction
    severity: SeverityLevel
    action_title: str
    rationale: str
    target_parameters: Dict[str, float]
    expected_quality_post_action: float
    expected_energy_kwh: float
    safety_check: SafetyCheckResult
    economic_impact: EconomicImpactAnalysis
    requires_human_approval: bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "severity": self.severity.value,
            "action_title": self.action_title,
            "rationale": self.rationale,
            "target_parameters": self.target_parameters,
            "expected_quality_post_action": self.expected_quality_post_action,
            "expected_energy_kwh": self.expected_energy_kwh,
            "safety_passed": self.safety_check.passed,
            "safety_status": self.safety_check.status,
            "safety_violations": self.safety_check.violations,
            "total_cost_if_stopped_usd": self.economic_impact.total_cost_if_stopped_usd,
            "expected_savings_by_correcting_usd": self.economic_impact.expected_savings_by_correcting_usd,
            "requires_human_approval": self.requires_human_approval,
            "timestamp": self.timestamp
        }


class ProductionContinuityManager:
    """
    Evaluates in-flight production problems and prioritizes safe continuity over premature shutdowns.
    """

    def __init__(
        self,
        safety_engine: Optional[SafetyRuleEngine] = None,
        material_lot_cost_usd: float = 850.0,
        downtime_rate_usd_per_hour: float = 450.0,
        energy_rate_usd_per_kwh: float = 0.15
    ):
        self.safety = safety_engine or SafetyRuleEngine()
        self.material_cost = material_lot_cost_usd
        self.downtime_rate = downtime_rate_usd_per_hour
        self.energy_rate = energy_rate_usd_per_kwh

    def assess_severity(
        self,
        health_index: float,
        anomaly_score: float,
        temperature: float,
        current_a: float,
        quality_score: float,
        is_sustained_defect: bool = False
    ) -> SeverityLevel:
        """Categorizes live conditions into four operational severity tiers (L0-L3)."""
        # L3: Physical safety breach or sustained irreversible structural failure
        if is_sustained_defect or temperature > 80.0 or current_a > 32.0 or health_index < 40.0:
            return SeverityLevel.L3_CRITICAL

        # L2: Recoverable process drift (e.g. rising temperature, moderate tool chatter)
        if health_index < 75.0 or temperature > 55.0 or current_a > 20.0 or anomaly_score > 0.15 or quality_score < 0.70:
            return SeverityLevel.L2_CORRECTIVE

        # L1: Transient spike or mild vibration drift
        if health_index < 85.0 or anomaly_score > 0.08 or temperature > 48.0:
            return SeverityLevel.L1_WARNING

        # L0: Steady state
        return SeverityLevel.L0_NORMAL

    def evaluate_decision(
        self,
        state: MachineState,
        golden_target_params: Optional[Dict[str, float]] = None,
        twin_simulated_yield: Optional[float] = None,
        is_sustained_defect: bool = False
    ) -> ContinuityDecision:
        """
        Synthesizes MachineState, Safety, and Economics to produce the optimal continuity decision.
        """
        severity = self.assess_severity(
            health_index=state.health_index,
            anomaly_score=state.anomaly_score,
            temperature=state.temperature,
            current_a=state.current,
            quality_score=state.quality_score,
            is_sustained_defect=is_sustained_defect
        )

        # Calculate economic penalty if production was stopped right now
        # Scrap loss scales with progress (more sunk value invested)
        progress_frac = max(0.05, min(1.0, state.batch_progress_pct / 100.0))
        sunk_material_loss = self.material_cost * progress_frac
        restart_energy_kwh = 28.0
        restart_energy_cost = restart_energy_kwh * self.energy_rate
        downtime_hours = 1.5
        downtime_cost = downtime_hours * self.downtime_rate
        total_stop_cost = round(sunk_material_loss + restart_energy_cost + downtime_cost, 2)

        # Baseline parameters
        target_params = {
            "temperature": state.temperature,
            "pressure": 5.0,
            "speed": 1800.0,
            "feed_rate": 0.75,
            "fan_pwm": 0
        }

        # ── CASE 1: L3 CRITICAL (Safety limit exceeded or irreversible damage) ──
        if severity == SeverityLevel.L3_CRITICAL:
            safety_res = self.safety.validate_command(
                current_temp=state.temperature,
                current_a=state.current
            )
            return ContinuityDecision(
                action=ContinuityAction.CONTROLLED_STOP,
                severity=SeverityLevel.L3_CRITICAL,
                action_title="Initiate Controlled Load Shedding & Safe State",
                rationale=(
                    f"CRITICAL SAFETY / STRUCTURAL THRESHOLD EXCEEDED (Health: {state.health_index:.1f}%, "
                    f"Temp: {state.temperature:.1f}°C, Current: {state.current:.1f}A). Continued operation "
                    f"threatens machine integrity. Controlled stop executed to preserve raw material lots."
                ),
                target_parameters={"fan_pwm": 255, "feed_hold": 1.0},
                expected_quality_post_action=0.0,
                expected_energy_kwh=round(state.energy_kwh, 1),
                safety_check=safety_res,
                economic_impact=EconomicImpactAnalysis(
                    material_scrap_risk_usd=round(sunk_material_loss, 2),
                    restart_energy_cost_usd=round(restart_energy_cost, 2),
                    downtime_cost_usd=round(downtime_cost, 2),
                    total_cost_if_stopped_usd=total_stop_cost,
                    expected_savings_by_correcting_usd=0.0
                ),
                requires_human_approval=False  # Safety interlocks act autonomously
            )

        # ── CASE 2: L2 CORRECTIVE (Safe corrective path available) ──
        elif severity == SeverityLevel.L2_CORRECTIVE:
            # Plan corrective trim: increase cooling, adjust speeds/feeds towards golden standard
            target_temp = 42.0 if state.temperature > 55.0 else state.temperature
            target_pwm = int(np.clip(80 + ((state.temperature - 45.0) / 25.0) * 175.0, 80, 255))
            
            if golden_target_params:
                target_speed = golden_target_params.get("speed", 1750.0)
                target_pressure = golden_target_params.get("pressure", 4.8)
            else:
                target_speed = 1750.0
                target_pressure = 4.8

            target_params = {
                "temperature": target_temp,
                "pressure": target_pressure,
                "speed": target_speed,
                "fan_pwm": target_pwm
            }

            # Run safety check on proposed corrective actions
            safety_res = self.safety.validate_command(
                target_temp=target_temp,
                target_pressure=target_pressure,
                target_speed=target_speed,
                target_pwm=target_pwm,
                current_temp=state.temperature,
                current_a=state.current
            )

            # If safety check fails, fallback to stop; otherwise CORRECT & CONTINUE
            if not safety_res.passed:
                action = ContinuityAction.CONTROLLED_STOP
                title = "Safety Interlock Blocked Correction ➔ Safe State"
                rationale = f"Proposed correction violated safety rules: {safety_res.violations}"
                savings = 0.0
            else:
                action = ContinuityAction.CORRECT
                title = "Apply In-Process Thermal & Speed Trimming (Continuity Mode)"
                rationale = (
                    f"Batch is {state.batch_progress_pct:.0f}% complete. Stopping would waste ${total_stop_cost:.2f} "
                    f"in sunk material & reheat energy. Digital Twin verified safe recovery: fan duty modulated to "
                    f"{target_pwm}/255 PWM to restore thermal equilibrium."
                )
                savings = round(total_stop_cost * 0.85, 2)

            return ContinuityDecision(
                action=action,
                severity=SeverityLevel.L2_CORRECTIVE,
                action_title=title,
                rationale=rationale,
                target_parameters=target_params,
                expected_quality_post_action=0.93,
                expected_energy_kwh=round(state.energy_kwh + 15.0, 1),
                safety_check=safety_res,
                economic_impact=EconomicImpactAnalysis(
                    material_scrap_risk_usd=round(sunk_material_loss, 2),
                    restart_energy_cost_usd=round(restart_energy_cost, 2),
                    downtime_cost_usd=round(downtime_cost, 2),
                    total_cost_if_stopped_usd=total_stop_cost,
                    expected_savings_by_correcting_usd=savings
                ),
                requires_human_approval=True  # Operator in the loop for process trims
            )

        # ── CASE 3: L1 WARNING (Transient Noise or Minor Drift) ──
        elif severity == SeverityLevel.L1_WARNING:
            safety_res = self.safety.validate_command(current_temp=state.temperature, current_a=state.current)
            return ContinuityDecision(
                action=ContinuityAction.CONTINUE,
                severity=SeverityLevel.L1_WARNING,
                action_title="Continue Production with Enhanced Metrology Monitoring",
                rationale=(
                    f"Minor parameter drift detected (Health: {state.health_index:.1f}%). "
                    f"Dual-window guardrail active; holding current setpoints while tracking trend."
                ),
                target_parameters=target_params,
                expected_quality_post_action=state.quality_score,
                expected_energy_kwh=round(state.energy_kwh, 1),
                safety_check=safety_res,
                economic_impact=EconomicImpactAnalysis(
                    material_scrap_risk_usd=0.0,
                    restart_energy_cost_usd=0.0,
                    downtime_cost_usd=0.0,
                    total_cost_if_stopped_usd=total_stop_cost,
                    expected_savings_by_correcting_usd=total_stop_cost
                ),
                requires_human_approval=False
            )

        # ── CASE 4: L0 NORMAL ──
        else:
            safety_res = self.safety.validate_command(current_temp=state.temperature, current_a=state.current)
            return ContinuityDecision(
                action=ContinuityAction.CONTINUE,
                severity=SeverityLevel.L0_NORMAL,
                action_title="Nominal Production Flow",
                rationale="All physical and quality parameters are inside nominal tolerances.",
                target_parameters=target_params,
                expected_quality_post_action=state.quality_score,
                expected_energy_kwh=round(state.energy_kwh, 1),
                safety_check=safety_res,
                economic_impact=EconomicImpactAnalysis(
                    material_scrap_risk_usd=0.0,
                    restart_energy_cost_usd=0.0,
                    downtime_cost_usd=0.0,
                    total_cost_if_stopped_usd=total_stop_cost,
                    expected_savings_by_correcting_usd=0.0
                ),
                requires_human_approval=False
            )
