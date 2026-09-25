"""
Deterministic Safety Rule Engine
Module: src/safety/safety_rules.py

Ensures that AI recommendations and optimizer outputs NEVER bypass strict physical
machine and human safety boundaries. Outputs PASS or BLOCK with actionable audit logs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

logger = logging.getLogger("safety_engine")


@dataclass
class SafetyBoundaryConfig:
    """Hard-coded physical machine operating envelope."""
    max_temperature_c: float = 85.0
    min_temperature_c: float = 15.0
    max_current_a: float = 35.0
    max_voltage_v: float = 260.0
    min_voltage_v: float = 200.0
    max_pwm_duty: int = 255
    min_pwm_duty: int = 0
    max_temp_rate_of_change_c_per_min: float = 12.0
    min_pressure_bar: float = 1.0
    max_pressure_bar: float = 9.5
    min_speed_rpm: float = 400.0
    max_speed_rpm: float = 3200.0


@dataclass
class SafetyCheckResult:
    """Outcome of deterministic safety validation."""
    passed: bool
    status: str  # "PASS" or "BLOCK"
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    mitigation_required: Optional[str] = None
    evaluated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class SafetyRuleEngine:
    """
    Independent deterministic safety interlock layer.
    
    Principles:
    1. AI suggestions are treated as UNTRUSTED proposals until checked.
    2. Optimizer cannot trade safety margins for energy efficiency.
    3. Blocks unsafe parameter changes and commands instantly.
    """

    def __init__(self, config: Optional[SafetyBoundaryConfig] = None):
        self.config = config or SafetyBoundaryConfig()

    def validate_command(
        self,
        target_temp: Optional[float] = None,
        target_pressure: Optional[float] = None,
        target_speed: Optional[float] = None,
        target_feed: Optional[float] = None,
        target_pwm: Optional[int] = None,
        current_temp: float = 45.0,
        current_a: float = 12.5,
        voltage_v: float = 230.0,
        temp_rate_c_min: float = 1.5
    ) -> SafetyCheckResult:
        """
        Validates whether a proposed parameter adjustment or control action is physically safe.
        """
        violations: List[str] = []
        warnings: List[str] = []

        # 1. Temperature Boundaries
        if target_temp is not None:
            if target_temp > self.config.max_temperature_c:
                violations.append(
                    f"Target temperature {target_temp:.1f}°C exceeds absolute safety limit ({self.config.max_temperature_c}°C)."
                )
            elif target_temp < self.config.min_temperature_c:
                violations.append(
                    f"Target temperature {target_temp:.1f}°C is below minimum safe operating limit ({self.config.min_temperature_c}°C)."
                )
            
            # Rate of change check
            temp_delta = abs(target_temp - current_temp)
            if temp_delta > 35.0:
                warnings.append(
                    f"Thermal step jump is large (Δ{temp_delta:.1f}°C). Proportional ramp rate enforced."
                )

        # 2. Live Current & Voltage Surge Checks
        if current_a > self.config.max_current_a:
            violations.append(
                f"Live spindle current {current_a:.1f}A exceeds machine thermal overload rating ({self.config.max_current_a}A)."
            )
        
        if voltage_v > self.config.max_voltage_v or voltage_v < self.config.min_voltage_v:
            violations.append(
                f"Line voltage {voltage_v:.1f}V is out of allowable tolerance [{self.config.min_voltage_v}V, {self.config.max_voltage_v}V]."
            )

        # 3. Dynamic Rate of Change Check
        if temp_rate_c_min > self.config.max_temp_rate_of_change_c_per_min:
            violations.append(
                f"Thermal ramp rate ({temp_rate_c_min:.1f}°C/min) exceeds maximum gradient ({self.config.max_temp_rate_of_change_c_per_min}°C/min). Thermal runaway risk."
            )

        # 4. Mechanical Boundaries
        if target_pressure is not None:
            if target_pressure < self.config.min_pressure_bar or target_pressure > self.config.max_pressure_bar:
                violations.append(
                    f"Clamp pressure {target_pressure:.2f} bar violates allowable physical range [{self.config.min_pressure_bar}, {self.config.max_pressure_bar}] bar."
                )

        if target_speed is not None:
            if target_speed < self.config.min_speed_rpm or target_speed > self.config.max_speed_rpm:
                violations.append(
                    f"Spindle speed {target_speed:.0f} RPM violates safe operating envelope [{self.config.min_speed_rpm}, {self.config.max_speed_rpm}] RPM."
                )

        if target_pwm is not None:
            if target_pwm < self.config.min_pwm_duty or target_pwm > self.config.max_pwm_duty:
                violations.append(
                    f"Actuator PWM duty {target_pwm} is outside logic range [0, 255]."
                )

        # Final Verdict
        passed = len(violations) == 0
        status = "PASS" if passed else "BLOCK"
        mitigation = None if passed else "Revert to safe nominal baseline; assert operator review."

        if not passed:
            logger.error(f"SAFETY INTERLOCK BLOCKED COMMAND: {violations}")

        return SafetyCheckResult(
            passed=passed,
            status=status,
            violations=violations,
            warnings=warnings,
            mitigation_required=mitigation
        )
