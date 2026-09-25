"""
ACMGS v2.0 Predictive Maintenance & Continuous Machine Health Scorer
Module: src/intelligence/health_scorer.py

Replaces binary anomaly thresholding with a mathematically rigorous, continuous
Machine Health Index (0-100%) mapped to 3 operational risk tiers:
  - NOMINAL (>= 75%): Standard operation; routine inspection interval.
  - DEGRADED (45% - 74%): Progressive bearing/tool wear; schedule maintenance within 48 hours.
  - CRITICAL (< 45%): High probability of tool chatter or spindle seizure; immediate inspection.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple, Any
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger("health_scorer")


class HealthTier(str, Enum):
    NOMINAL = "NOMINAL"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"


@dataclass
class HealthReport:
    """Detailed health diagnosis container."""
    health_index: float  # 0.0 to 100.0%
    tier: HealthTier
    recon_error: float
    recon_threshold: float
    current_rms: float
    current_nominal_rms: float
    temperature: float
    temp_baseline: float
    
    # Penalty components
    recon_penalty: float
    current_penalty: float
    temp_penalty: float
    
    status_summary: str
    action_recommendation: str
    maintenance_window_hours: Optional[int] = None
    color_hex: str = "#00ff88"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MachineHealthScorer:
    """
    Continuous Machine Health Scoring Engine based on multi-sensor degradation dynamics.
    
    Formula:
      Health Index = 100 * [1 - (0.50 * (Recon Error / Threshold) + 
                                 0.30 * (|ΔI_rms| / 25A) + 
                                 0.20 * (max(0, T - 35) / 40°C))]
    """

    def __init__(
        self,
        recon_threshold: float = 0.199084,
        nominal_current_rms: float = 12.5,
        max_current_delta_ref: float = 25.0,
        temp_baseline: float = 35.0,
        temp_span_ref: float = 40.0,
    ):
        self.recon_threshold = recon_threshold
        self.nominal_current_rms = nominal_current_rms
        self.max_current_delta_ref = max_current_delta_ref
        self.temp_baseline = temp_baseline
        self.temp_span_ref = temp_span_ref

    def compute_health_index(
        self,
        recon_error: float,
        current_rms: float,
        temperature: float,
        nominal_current: Optional[float] = None
    ) -> float:
        """
        Computes the raw continuous machine health index (0 to 100%).
        """
        nom_curr = nominal_current if nominal_current is not None else self.nominal_current_rms
        
        # 1. Reconstruction Error Penalty (Weight: 50%)
        # Ratio of LSTM autoencoder reconstruction error to 3-sigma baseline threshold
        recon_ratio = recon_error / max(self.recon_threshold, 1e-6)
        recon_penalty = 0.50 * recon_ratio

        # 2. Spindle Current RMS Drift Penalty (Weight: 30%)
        # Ratio of absolute current deviation to 25A reference scale
        delta_current = abs(current_rms - nom_curr)
        current_penalty = 0.30 * (delta_current / self.max_current_delta_ref)

        # 3. Chamber Thermal Deviation Penalty (Weight: 20%)
        # Excess temperature above 35°C over 40°C thermal scale
        excess_temp = max(0.0, temperature - self.temp_baseline)
        temp_penalty = 0.20 * (excess_temp / self.temp_span_ref)

        total_degradation = recon_penalty + current_penalty + temp_penalty
        health_index = 100.0 * (1.0 - total_degradation)
        
        return float(np.clip(health_index, 0.0, 100.0))

    def evaluate(
        self,
        recon_error: float,
        current_rms: float,
        temperature: float,
        nominal_current: Optional[float] = None
    ) -> HealthReport:
        """
        Evaluates machine state and returns a complete HealthReport with tier classification.
        """
        nom_curr = nominal_current if nominal_current is not None else self.nominal_current_rms
        
        recon_ratio = recon_error / max(self.recon_threshold, 1e-6)
        recon_penalty = 0.50 * recon_ratio
        
        delta_current = abs(current_rms - nom_curr)
        current_penalty = 0.30 * (delta_current / self.max_current_delta_ref)
        
        excess_temp = max(0.0, temperature - self.temp_baseline)
        temp_penalty = 0.20 * (excess_temp / self.temp_span_ref)
        
        health_val = self.compute_health_index(recon_error, current_rms, temperature, nom_curr)

        # Determine Tier
        if health_val >= 75.0:
            tier = HealthTier.NOMINAL
            color = "#00ff88"  # Green
            summary = "Machine operating in NOMINAL condition. Normal tool wear rate."
            action = "Continue standard production. Routine inspection interval."
            maint_hours = None
        elif health_val >= 45.0:
            tier = HealthTier.DEGRADED
            color = "#ffd600"  # Yellow
            summary = "Machine in DEGRADED tier. Progressive bearing/spindle wear detected."
            action = "Schedule preventive maintenance within 48 hours. Monitor thermal drift."
            maint_hours = 48
        else:
            tier = HealthTier.CRITICAL
            color = "#ff4b4b"  # Red
            summary = "Machine in CRITICAL tier! High probability of tool chatter or spindle seizure."
            action = "Immediate inspection required. Halt non-essential high-feed batches."
            maint_hours = 0

        return HealthReport(
            health_index=round(health_val, 1),
            tier=tier,
            recon_error=recon_error,
            recon_threshold=self.recon_threshold,
            current_rms=current_rms,
            current_nominal_rms=nom_curr,
            temperature=temperature,
            temp_baseline=self.temp_baseline,
            recon_penalty=round(recon_penalty * 100.0, 1),
            current_penalty=round(current_penalty * 100.0, 1),
            temp_penalty=round(temp_penalty * 100.0, 1),
            status_summary=summary,
            action_recommendation=action,
            maintenance_window_hours=maint_hours,
            color_hex=color
        )
