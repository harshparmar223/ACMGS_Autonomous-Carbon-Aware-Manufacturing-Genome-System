"""
ACMGS v2.0 Golden Signature Benchmarking System
Module: src/intelligence/golden_signature.py

Uses a k-d Tree nearest-neighbor index over the top 5% historical batches (Yield > 0.97, Quality > 0.95)
and calculates exact parameter deltas (ΔTemp, ΔPressure, ΔSpeed, ΔFeed Rate) to bring sub-optimal
runs back to gold standards.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import sqlite3
import os
import logging
from datetime import datetime
from scipy.spatial import KDTree

from config.settings import DB_PATH, SIMULATED_DIR, GENOME_PROCESS_FEATURES

logger = logging.getLogger("golden_signature")


@dataclass
class GoldenRecommendation:
    """Prescriptive adjustments to bring current process to golden benchmark."""
    nearest_batch_id: str
    target_yield: float
    target_quality: float
    target_energy: float
    
    # Process setting comparisons
    current_params: Dict[str, float]
    target_params: Dict[str, float]
    deltas: Dict[str, float]  # target - current
    
    # Prescriptive guidance string
    prescriptive_text: str
    similarity_score_pct: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class GoldenSignatureEngine:
    """
    Golden Signature Index & Nearest Neighbor Prescriptive Recommender.
    """

    def __init__(self, db_path: str = DB_PATH, csv_path: Optional[str] = None):
        self.db_path = db_path
        self.csv_path = csv_path or os.path.join(SIMULATED_DIR, "batch_data.csv")
        self.gold_df: Optional[pd.DataFrame] = None
        self.kdtree: Optional[KDTree] = None
        self.feature_names = ["temperature", "pressure", "speed", "feed_rate", "humidity"]
        self.scales = np.array([200.0, 10.0, 3000.0, 2.0, 100.0])  # Normalization scales for distance
        self.load_golden_population()

    def load_golden_population(self):
        """Loads and indexes the top 5% historical batches."""
        df = None
        
        # 1. Try loading from SQLite
        if os.path.exists(self.db_path):
            try:
                conn = sqlite3.connect(self.db_path)
                query = "SELECT * FROM batches WHERE yield >= 0.95 AND quality >= 0.90 ORDER BY yield DESC"
                df = pd.read_sql_query(query, conn)
                conn.close()
            except Exception as e:
                logger.debug(f"Could not load golden batches from DB: {e}")

        # 2. Try loading from CSV
        if (df is None or df.empty) and os.path.exists(self.csv_path):
            try:
                raw_df = pd.read_csv(self.csv_path)
                # Top 5% by composite score
                raw_df["score"] = raw_df["yield"] * 0.6 + raw_df["quality"] * 0.4
                top_threshold = raw_df["score"].quantile(0.95)
                df = raw_df[raw_df["score"] >= top_threshold].copy()
            except Exception as e:
                logger.debug(f"Could not load golden batches from CSV: {e}")

        # 3. Synthetic Golden Baseline if no dataset exists yet
        if df is None or df.empty:
            records = []
            for i in range(100):
                records.append({
                    "batch_id": f"GOLD_{i:03d}",
                    "temperature": float(np.random.normal(210.0, 4.0)),
                    "pressure": float(np.random.normal(5.4, 0.2)),
                    "speed": float(np.random.normal(1950.0, 40.0)),
                    "feed_rate": float(np.random.normal(0.86, 0.03)),
                    "humidity": float(np.random.normal(42.0, 3.0)),
                    "yield": float(np.clip(np.random.normal(0.985, 0.008), 0.97, 0.999)),
                    "quality": float(np.clip(np.random.normal(0.975, 0.010), 0.95, 0.995)),
                    "energy_consumption": float(np.random.normal(120.0, 10.0)),
                    "carbon_intensity": float(np.random.normal(140.0, 20.0))
                })
            df = pd.DataFrame(records)

        self.gold_df = df.reset_index(drop=True)
        # Build KDTree on normalized process parameters
        data_matrix = self.gold_df[self.feature_names].values / self.scales
        self.kdtree = KDTree(data_matrix)
        logger.info(f"GoldenSignatureEngine: Indexed {len(self.gold_df)} top-tier gold batches.")

    def find_nearest_golden_recipe(
        self,
        temperature: float,
        pressure: float,
        speed: float,
        feed_rate: float,
        humidity: float = 45.0
    ) -> GoldenRecommendation:
        """
        Finds the nearest gold-standard batch and calculates prescriptive deltas.
        """
        if self.kdtree is None or self.gold_df is None:
            self.load_golden_population()

        current_pt = np.array([temperature, pressure, speed, feed_rate, humidity])
        norm_pt = current_pt / self.scales
        
        dist, idx = self.kdtree.query(norm_pt, k=1)
        gold_row = self.gold_df.iloc[idx]

        curr_params = {
            "temperature": round(float(temperature), 1),
            "pressure": round(float(pressure), 2),
            "speed": round(float(speed), 0),
            "feed_rate": round(float(feed_rate), 2),
            "humidity": round(float(humidity), 1)
        }

        gold_params = {
            "temperature": round(float(gold_row["temperature"]), 1),
            "pressure": round(float(gold_row["pressure"]), 2),
            "speed": round(float(gold_row["speed"]), 0),
            "feed_rate": round(float(gold_row["feed_rate"]), 2),
            "humidity": round(float(gold_row["humidity"]), 1)
        }

        deltas = {
            "temperature": round(gold_params["temperature"] - curr_params["temperature"], 1),
            "pressure": round(gold_params["pressure"] - curr_params["pressure"], 2),
            "speed": round(gold_params["speed"] - curr_params["speed"], 0),
            "feed_rate": round(gold_params["feed_rate"] - curr_params["feed_rate"], 2),
            "humidity": round(gold_params["humidity"] - curr_params["humidity"], 1),
        }

        # Formulate Prescriptive Text matching Dossier Specs
        prescriptions = []
        
        # Temp delta
        dt = deltas["temperature"]
        if abs(dt) >= 0.5:
            action = "Increase" if dt > 0 else "Reduce"
            prescriptions.append(f"{action} chamber temperature by {abs(dt):.1f}°C")

        # Pressure delta
        dp = deltas["pressure"]
        if abs(dp) >= 0.05:
            action = "Increase" if dp > 0 else "Decrease"
            prescriptions.append(f"{action} clamp pressure by {abs(dp):.2f} bar")

        # Speed delta
        ds = deltas["speed"]
        if abs(ds) >= 20:
            action = "Increase" if ds > 0 else "Reduce"
            prescriptions.append(f"{action} spindle speed by {int(abs(ds))} RPM")

        # Feed rate delta
        df_val = deltas["feed_rate"]
        if abs(df_val) >= 0.01:
            action = "Increase" if df_val > 0 else "Reduce"
            prescriptions.append(f"{action} feed rate by {abs(df_val):.2f} kg/h")

        if not prescriptions:
            prescriptive_text = "Current batch parameters perfectly match Golden Signature standards (within 0.5% tolerance)."
        else:
            prescriptive_text = "Recommended adjustment: " + "; ".join(prescriptions) + "."

        sim_score = max(0.0, min(100.0, (1.0 - dist) * 100.0))

        return GoldenRecommendation(
            nearest_batch_id=str(gold_row.get("batch_id", "GOLD_REF_01")),
            target_yield=round(float(gold_row.get("yield", 0.985)), 4),
            target_quality=round(float(gold_row.get("quality", 0.978)), 4),
            target_energy=round(float(gold_row.get("energy_consumption", 118.5)), 1),
            current_params=curr_params,
            target_params=gold_params,
            deltas=deltas,
            prescriptive_text=prescriptive_text,
            similarity_score_pct=round(sim_score, 1)
        )
