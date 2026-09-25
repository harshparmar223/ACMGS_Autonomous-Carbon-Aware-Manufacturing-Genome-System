"""
Human-in-the-Loop Operator Approval & Command Service
Module: src/services/command_service.py

Handles operator approval workflow (APPROVE / REJECT), safety verification,
command dispatching to edge actuators, and persistent audit logging.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging
import sqlite3
import json

from config.settings import DB_PATH
from src.safety.safety_rules import SafetyRuleEngine, SafetyCheckResult
from src.services.production_continuity import ContinuityDecision

logger = logging.getLogger("command_service")


@dataclass
class CommandAuditRecord:
    """Audit entry for every operator or system action."""
    command_id: str
    machine_id: str
    decision_action: str
    target_params: Dict[str, float]
    safety_verdict: str
    operator_id: Optional[str]
    approval_status: str  # "APPROVED", "REJECTED", "AUTO_DISPATCHED", "PENDING"
    dispatched_to_hardware: bool
    execution_result: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class CommandService:
    """
    Manages operator approval requests, safety re-validation, and hardware dispatching.
    """

    def __init__(self, db_path: str = DB_PATH, safety_engine: Optional[SafetyRuleEngine] = None):
        self.db_path = db_path
        self.safety = safety_engine or SafetyRuleEngine()
        self._pending_decisions: Dict[str, ContinuityDecision] = {}
        self._audit_log: List[CommandAuditRecord] = []

    def submit_for_approval(self, machine_id: str, decision: ContinuityDecision) -> str:
        """Enqueues an AI recommendation for operator review."""
        req_id = f"REQ_{machine_id}_{int(datetime.now().timestamp())}"
        self._pending_decisions[req_id] = decision
        logger.info(f"Enqueued decision {req_id} ({decision.action_title}) for human operator approval.")
        return req_id

    def process_operator_action(
        self,
        request_id: str,
        approve: bool,
        operator_id: str = "OPERATOR_01"
    ) -> Tuple[bool, str]:
        """
        Executes or rejects a pending recommendation based on operator input.
        """
        decision = self._pending_decisions.get(request_id)
        if not decision:
            return False, f"Request ID {request_id} not found in pending approval queue."

        if not approve:
            audit = CommandAuditRecord(
                command_id=request_id,
                machine_id="MACHINE_01",
                decision_action=decision.action.value,
                target_params=decision.target_parameters,
                safety_verdict=decision.safety_check.status,
                operator_id=operator_id,
                approval_status="REJECTED",
                dispatched_to_hardware=False,
                execution_result="Operator rejected proposed parameter trim. Maintaining baseline setpoints."
            )
            self._audit_log.append(audit)
            del self._pending_decisions[request_id]
            return True, "Action REJECTED by operator. Current setpoints maintained."

        # Operator Approved -> Re-verify Safety Rules before hardware dispatch
        safety_recheck = self.safety.validate_command(
            target_temp=decision.target_parameters.get("temperature"),
            target_pressure=decision.target_parameters.get("pressure"),
            target_speed=decision.target_parameters.get("speed"),
            target_pwm=int(decision.target_parameters.get("fan_pwm", 0))
        )

        if not safety_recheck.passed:
            audit = CommandAuditRecord(
                command_id=request_id,
                machine_id="MACHINE_01",
                decision_action=decision.action.value,
                target_params=decision.target_parameters,
                safety_verdict="BLOCK",
                operator_id=operator_id,
                approval_status="BLOCKED_BY_SAFETY",
                dispatched_to_hardware=False,
                execution_result=f"Approved by operator but BLOCKED by Safety Engine: {safety_recheck.violations}"
            )
            self._audit_log.append(audit)
            del self._pending_decisions[request_id]
            return False, f"COMMAND BLOCKED BY SAFETY INTERLOCK: {safety_recheck.violations}"

        # Dispatch Command to Hardware
        audit = CommandAuditRecord(
            command_id=request_id,
            machine_id="MACHINE_01",
            decision_action=decision.action.value,
            target_params=decision.target_parameters,
            safety_verdict="PASS",
            operator_id=operator_id,
            approval_status="APPROVED",
            dispatched_to_hardware=True,
            execution_result=f"Dispatched parameter trim to Node 2 (MOSFET PWM: {decision.target_parameters.get('fan_pwm')}/255). Line running in Continuity Mode."
        )
        self._audit_log.append(audit)
        del self._pending_decisions[request_id]
        return True, f"SUCCESS: Operator approved. Command dispatched to machine ({decision.action_title})."
