"""
Unit Test Suite for Production Continuity, Safety Engine & Recovery
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.machine_state import MachineState, SensorReadingContract
from src.safety.safety_rules import SafetyRuleEngine, SafetyCheckResult, SafetyBoundaryConfig
from src.services.production_continuity import (
    ProductionContinuityManager,
    ContinuityAction,
    SeverityLevel
)
from src.services.recovery import RecoveryManager
from src.services.command_service import CommandService


class TestProductionContinuitySuite(unittest.TestCase):

    def setUp(self):
        self.state = MachineState(
            machine_id="MACHINE_01",
            temperature=45.0,
            humidity=50.0,
            voltage=230.0,
            current=12.5,
            power=2875.0,
            energy_kwh=120.0,
            carbon_emissions_kg=30.0,
            health_index=95.0,
            quality_score=0.95,
            batch_progress_pct=60.0
        )
        self.safety = SafetyRuleEngine()
        self.continuity = ProductionContinuityManager(safety_engine=self.safety)
        self.recovery = RecoveryManager()
        self.cmd_svc = CommandService(safety_engine=self.safety)

    def test_sensor_reading_contract_validation(self):
        # Valid reading
        raw = {
            "device_id": "ESP32_01",
            "machine_id": "MACHINE_01",
            "temperature": 42.5,
            "humidity": 45.0,
            "voltage": 230.0,
            "current": 13.0
        }
        contract = SensorReadingContract.from_dict(raw)
        self.assertEqual(contract.temperature, 42.5)
        self.assertEqual(contract.power, 230.0 * 13.0)

        # Catch impossible sensor values
        bad_raw = {"temperature": 999.0, "current": -10.0}
        contract_bad = SensorReadingContract.from_dict(bad_raw)
        self.assertEqual(contract_bad.temperature, 500.0)
        self.assertEqual(contract_bad.current, 0.0)

    def test_safety_rule_engine_pass_and_block(self):
        # Safe command
        res_pass = self.safety.validate_command(
            target_temp=45.0,
            target_pressure=5.0,
            target_speed=1800.0,
            current_a=12.5
        )
        self.assertTrue(res_pass.passed)
        self.assertEqual(res_pass.status, "PASS")

        # Over-temperature command -> BLOCK
        res_block_temp = self.safety.validate_command(
            target_temp=98.0,  # exceeds 85.0C max
            current_a=12.5
        )
        self.assertFalse(res_block_temp.passed)
        self.assertEqual(res_block_temp.status, "BLOCK")
        self.assertTrue(any("exceeds absolute safety limit" in v for v in res_block_temp.violations))

        # Over-current surge -> BLOCK
        res_block_curr = self.safety.validate_command(
            current_a=42.0  # exceeds 35.0A max
        )
        self.assertFalse(res_block_curr.passed)
        self.assertEqual(res_block_curr.status, "BLOCK")

    def test_continuity_l0_normal(self):
        dec = self.continuity.evaluate_decision(self.state)
        self.assertEqual(dec.severity, SeverityLevel.L0_NORMAL)
        self.assertEqual(dec.action, ContinuityAction.CONTINUE)
        self.assertFalse(dec.requires_human_approval)

    def test_continuity_l2_corrective_savings(self):
        # Moderate thermal drift at 70% batch progress -> Safe corrective trim
        self.state.temperature = 62.0
        self.state.health_index = 68.0
        self.state.batch_progress_pct = 70.0

        dec = self.continuity.evaluate_decision(self.state)
        self.assertEqual(dec.severity, SeverityLevel.L2_CORRECTIVE)
        self.assertEqual(dec.action, ContinuityAction.CORRECT)
        self.assertTrue(dec.requires_human_approval)
        self.assertGreater(dec.economic_impact.expected_savings_by_correcting_usd, 0.0)
        self.assertIn("In-Process Thermal", dec.action_title)

    def test_continuity_l3_critical_controlled_stop(self):
        # Severe critical defect
        self.state.temperature = 88.0
        self.state.current = 36.0
        self.state.health_index = 32.0

        dec = self.continuity.evaluate_decision(self.state, is_sustained_defect=True)
        self.assertEqual(dec.severity, SeverityLevel.L3_CRITICAL)
        self.assertEqual(dec.action, ContinuityAction.CONTROLLED_STOP)
        self.assertFalse(dec.requires_human_approval)

    def test_recovery_checkpoint_and_resume(self):
        # Create checkpoint at 65% progress
        self.state.batch_progress_pct = 65.0
        chk = self.recovery.create_checkpoint(self.state, stop_reason="Tool change required")
        self.assertEqual(chk.batch_progress_pct, 65.0)

        # Pre-flight check before cooldown -> Not ready
        verif_fail = self.recovery.verify_pre_flight("MACHINE_01", current_temp=62.0, current_a=0.0)
        self.assertFalse(verif_fail.ready_to_resume)

        # Pre-flight check after cooldown & clearance -> Ready
        verif_ok = self.recovery.verify_pre_flight("MACHINE_01", current_temp=38.0, current_a=0.0)
        self.assertTrue(verif_ok.ready_to_resume)

        # Resume state
        resumed, msg = self.recovery.resume_production("MACHINE_01", self.state, verif_ok)
        self.assertTrue(resumed)
        self.assertEqual(self.state.batch_progress_pct, 65.0)
        self.assertEqual(self.state.machine_status, "RUNNING")

    def test_command_service_approval_workflow(self):
        # Enqueue corrective decision
        self.state.temperature = 60.0
        dec = self.continuity.evaluate_decision(self.state)
        req_id = self.cmd_svc.submit_for_approval("MACHINE_01", dec)
        self.assertIn("REQ_MACHINE_01", req_id)

        # Operator Approves
        ok, msg = self.cmd_svc.process_operator_action(req_id, approve=True)
        self.assertTrue(ok)
        self.assertIn("SUCCESS", msg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
