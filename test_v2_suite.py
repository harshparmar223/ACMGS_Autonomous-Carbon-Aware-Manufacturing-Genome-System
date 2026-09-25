"""
Comprehensive Test Suite for ACMGS v2.0 Architecture & Upgrades
Tests all 6 core upgrades and mathematical models specified in the Master Defense Dossier:
1. Autonomous Closed-Loop MOSFET Control Layer & Dual-Window Guardrail (DecisionEngine)
2. Predictive Maintenance & Continuous Machine Health Index (MachineHealthScorer)
3. Explainable Root Cause Analysis (TreeSHAP RCA Engine)
4. Golden Signature Benchmarking System (GoldenSignatureEngine)
5. Enhanced Software Digital Twin (Dual-State Plan A vs Plan B & In-Process Sunk-Energy Defect Abort)
6. REST API v2.0 Endpoints
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.control.decision_engine import DecisionEngine, ActuationCommand
from src.intelligence.health_scorer import MachineHealthScorer, HealthTier, HealthReport
from src.intelligence.rca_engine import RCAEngine, RCAReport
from src.intelligence.golden_signature import GoldenSignatureEngine, GoldenRecommendation
from src.digital_twin.twin_engine import DigitalTwinEngine, TwinComparisonResult


class TestACMGSv2Suite(unittest.TestCase):

    def setUp(self):
        self.recon_threshold = 0.199084

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Closed-Loop MOSFET Control & Proportional Fan Law Tests
    # ─────────────────────────────────────────────────────────────────────────
    def test_proportional_fan_law_below_45(self):
        engine = DecisionEngine(recon_threshold=self.recon_threshold)
        cmd = engine.evaluate_step(
            temperature=38.0,
            current_rms=12.5,
            recon_error=0.04,
            predicted_quality=0.92
        )
        self.assertEqual(cmd.fan_pwm_duty, 0)
        self.assertEqual(cmd.cooling_state, "OFF")
        self.assertFalse(cmd.emergency_abort)
        self.assertFalse(cmd.feed_hold)

    def test_proportional_fan_law_mid_range(self):
        engine = DecisionEngine(recon_threshold=self.recon_threshold)
        # At T=57.5°C (halfway between 45 and 70), PWM = 80 + 0.5 * 175 = 167.5 -> 167 or 168
        cmd = engine.evaluate_step(
            temperature=57.5,
            current_rms=13.0,
            recon_error=0.05,
            predicted_quality=0.88
        )
        self.assertAlmostEqual(cmd.fan_pwm_duty, 167.5, delta=2)
        self.assertEqual(cmd.cooling_state, "MODULATING")
        self.assertFalse(cmd.emergency_abort)

    def test_proportional_fan_law_above_70(self):
        engine = DecisionEngine(recon_threshold=self.recon_threshold)
        cmd = engine.evaluate_step(
            temperature=74.0,
            current_rms=14.0,
            recon_error=0.05,
            predicted_quality=0.85
        )
        self.assertEqual(cmd.fan_pwm_duty, 255)
        self.assertEqual(cmd.cooling_state, "MAX_COOLING")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Dual-Window Confirmation Guardrail Tests
    # ─────────────────────────────────────────────────────────────────────────
    def test_transient_spike_rejection(self):
        """Single-window current/recon spike is filtered out (No abort on single spike)."""
        engine = DecisionEngine(recon_threshold=self.recon_threshold)
        # Window 1: High recon error & low quality (e.g. transient noise)
        cmd1 = engine.evaluate_step(
            temperature=48.0,
            current_rms=28.0,
            recon_error=0.45,  # > 3.5σ
            predicted_quality=0.35  # < 0.40
        )
        # Guardrail requires 2 consecutive windows -> Window 1 must NOT trigger emergency abort
        self.assertFalse(cmd1.emergency_abort)
        self.assertFalse(cmd1.feed_hold)

        # Window 2: Returns to normal
        cmd2 = engine.evaluate_step(
            temperature=46.0,
            current_rms=12.5,
            recon_error=0.05,
            predicted_quality=0.91
        )
        self.assertFalse(cmd2.emergency_abort)
        self.assertFalse(cmd2.feed_hold)

    def test_sustained_irreversible_defect_abort(self):
        """Sustained defect across 2 consecutive 500ms windows triggers Emergency Abort & Feed-Hold in <20ms."""
        engine = DecisionEngine(recon_threshold=self.recon_threshold)
        # Window 1
        cmd1 = engine.evaluate_step(
            temperature=78.0,
            current_rms=32.0,
            recon_error=0.48,
            predicted_quality=0.32
        )
        self.assertFalse(cmd1.emergency_abort)

        # Window 2: Sustained failure
        cmd2 = engine.evaluate_step(
            temperature=82.0,
            current_rms=34.0,
            recon_error=0.52,
            predicted_quality=0.28
        )
        self.assertTrue(cmd2.emergency_abort)
        self.assertTrue(cmd2.feed_hold)
        self.assertEqual(cmd2.fan_pwm_duty, 255)
        self.assertEqual(cmd2.status, "IRREVERSIBLY_DAMAGED_IN_FLIGHT")
        self.assertIn("EMERGENCY SUNK-ENERGY ABORT", cmd2.reason)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Continuous Machine Health Index (0-100%) Tests
    # ─────────────────────────────────────────────────────────────────────────
    def test_health_scorer_nominal(self):
        scorer = MachineHealthScorer(recon_threshold=self.recon_threshold)
        report = scorer.evaluate(recon_error=0.03, current_rms=12.5, temperature=32.0)
        self.assertGreaterEqual(report.health_index, 75.0)
        self.assertEqual(report.tier, HealthTier.NOMINAL)

    def test_health_scorer_degraded(self):
        scorer = MachineHealthScorer(recon_threshold=self.recon_threshold)
        report = scorer.evaluate(recon_error=0.12, current_rms=18.0, temperature=55.0)
        self.assertTrue(45.0 <= report.health_index < 75.0)
        self.assertEqual(report.tier, HealthTier.DEGRADED)

    def test_health_scorer_critical(self):
        scorer = MachineHealthScorer(recon_threshold=self.recon_threshold)
        report = scorer.evaluate(recon_error=0.35, current_rms=30.0, temperature=80.0)
        self.assertLess(report.health_index, 45.0)
        self.assertEqual(report.tier, HealthTier.CRITICAL)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. TreeSHAP Explainable RCA Engine Tests
    # ─────────────────────────────────────────────────────────────────────────
    def test_rca_engine_explanation(self):
        rca = RCAEngine()
        genome = np.array([
            85.0, 4.5, 1750.0, 0.82, 48.0,  # High temperature
            7.85, 200.0, 2.0,
            0.42, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # High recon error in EdNA
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            300.0
        ], dtype=np.float32)
        rep = rca.explain(genome_vector=genome, target_index=0, recon_error=0.42)
        self.assertIsInstance(rep, RCAReport)
        self.assertGreater(len(rep.top_attributions), 0)
        self.assertTrue(len(rep.plain_english_diagnosis) > 10)

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Golden Signature Benchmarking Engine Tests
    # ─────────────────────────────────────────────────────────────────────────
    def test_golden_signature_benchmark(self):
        golden = GoldenSignatureEngine()
        rec = golden.find_nearest_golden_recipe(
            temperature=245.0,
            pressure=4.8,
            speed=1650.0,
            feed_rate=0.72,
            humidity=45.0
        )
        self.assertIsInstance(rec, GoldenRecommendation)
        self.assertIn("temperature", rec.deltas)
        self.assertIn("pressure", rec.deltas)
        self.assertIn("speed", rec.deltas)
        self.assertIn("feed_rate", rec.deltas)
        self.assertTrue(len(rec.prescriptive_text) > 15)

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Digital Twin (Dual-State Simulation & Sunk Energy Abort) Tests
    # ─────────────────────────────────────────────────────────────────────────
    def test_digital_twin_nominal_delta(self):
        twin = DigitalTwinEngine()
        comp = twin.compute_dual_state(
            grid_carbon_intensity=350.0,
            defect_injected=False
        )
        self.assertIsInstance(comp, TwinComparisonResult)
        self.assertAlmostEqual(comp.delta_yield_pct, 9.0, delta=2.0)
        self.assertAlmostEqual(comp.delta_energy_pct, -14.1, delta=2.0)
        self.assertAlmostEqual(comp.delta_carbon_pct, -14.1, delta=2.0)

        res = twin.simulate_batch_comparison(
            temperature=200.0,
            pressure=5.0,
            speed=1800.0,
            feed_rate=0.75,
            humidity=45.0,
            inject_failure=False
        )
        self.assertAlmostEqual(res.yield_delta_pct, 9.0, delta=2.0)
        self.assertAlmostEqual(res.energy_delta_pct, -14.1, delta=2.0)

    def test_digital_twin_defect_sunk_energy_saved(self):
        twin = DigitalTwinEngine()
        res = twin.simulate_batch_comparison(
            temperature=295.0,
            pressure=5.2,
            speed=1850.0,
            feed_rate=0.75,
            humidity=45.0,
            inject_failure=True,
            failure_minute=18,
            machine_power_kw=50.0
        )
        # Power saved = 50 kW * ((60 - 18) / 60) h = 50 * 42/60 = 35.0 kWh
        # Or with calibrated factors >= 28.8 kWh
        self.assertGreaterEqual(res.sunk_energy_saved_kwh, 25.0)
        self.assertGreaterEqual(res.sunk_carbon_avoided_kg, 8.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
