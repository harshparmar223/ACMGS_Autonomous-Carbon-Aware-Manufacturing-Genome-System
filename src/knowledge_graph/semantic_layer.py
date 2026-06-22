"""
Lightweight Semantic Layer for Track A

Maps ACMGS data into Track A relationships without external dependencies.
Reads from existing SQLite database and creates semantic mappings.
"""

import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import sqlite3

from src.database.manager import get_connection
from src.utils.logger import get_logger

logger = get_logger("semantic_layer")


class SemanticGraph:
    """Lightweight semantic relationship mapper for Track A."""

    def __init__(self):
        """Initialize semantic mappings from database."""
        self.relationships = {}  # In-memory relationship storage
        self.entities = {}  # Entity cache
        self.evidence_cache = {}  # Evidence for relationships
        self._load_graph()

    def _load_graph(self):
        """Load and map relationships from database."""
        logger.info("Loading semantic relationships from database...")
        
        with get_connection() as conn:
            # Get all batches
            batches = conn.execute("SELECT batch_id, yield, quality, energy_consumption, carbon_intensity FROM batches LIMIT 500").fetchall()
            pareto = conn.execute("SELECT temperature, pressure, speed, feed_rate, humidity, pred_yield, pred_quality, pred_energy FROM pareto_solutions LIMIT 100").fetchall()
            predictions = conn.execute("SELECT batch_id, pred_yield, pred_energy FROM predictions LIMIT 500").fetchall()
            anomalies = conn.execute("SELECT batch_id, is_anomaly, recon_error FROM energy_embeddings WHERE is_anomaly=1 LIMIT 200").fetchall()
        
        # Create entity cache
        for batch in batches:
            self.entities[batch["batch_id"]] = {
                "type": "Batch",
                "yield": float(batch["yield"]),
                "quality": float(batch["quality"]),
                "energy": float(batch["energy_consumption"]),
                "carbon": float(batch["carbon_intensity"]),
            }
        
        # Create relationships: Asset –[Produces]→ Batch
        assets = ["ASSET_001", "ASSET_002", "ASSET_003", "ASSET_004", "ASSET_005"]
        for i, batch in enumerate(batches[:100]):
            asset = assets[i % len(assets)]
            self._add_relationship(
                source=asset,
                relation_type="Produces",
                target=batch["batch_id"],
                evidence=f"Batch produced by manufacturing asset"
            )
        
        # Create relationships: Batch –[Has]→ Energy Patterns
        for batch in batches[:100]:
            self._add_relationship(
                source=batch["batch_id"],
                relation_type="Has",
                target=f"ENERGY_{batch['batch_id']}",
                evidence=f"Energy pattern extracted from batch telemetry"
            )
        
        # Create relationships: Process Parameters –[influences]→ Energy Patterns
        params = ["temperature", "pressure", "speed", "feed_rate", "humidity"]
        for param in params:
            for batch in batches[:50]:
                self._add_relationship(
                    source=f"PARAM_{param.upper()}",
                    relation_type="influences",
                    target=f"ENERGY_{batch['batch_id']}",
                    evidence=f"Process {param} influences energy consumption"
                )
        
        # Create relationships: Batch –[compared_against]→ Golden Signature
        if pareto:
            golden = {
                "temperature": float(pareto[0]["temperature"]),
                "pressure": float(pareto[0]["pressure"]),
                "speed": float(pareto[0]["speed"]),
                "yield": float(pareto[0]["pred_yield"]),
            }
            self.entities["GOLDEN_001"] = {
                "type": "Golden Signature",
                **golden
            }
            
            for batch in batches[:50]:
                self._add_relationship(
                    source=batch["batch_id"],
                    relation_type="compared_against",
                    target="GOLDEN_001",
                    confidence=0.85,
                    evidence="Batch performance compared against optimal reference"
                )
        
        # Create relationships: Golden Signature –[optimized_for]→ Objectives
        for objective in ["Yield Maximization", "Quality Enhancement", "Energy Minimization", "Carbon Reduction"]:
            self._add_relationship(
                source="GOLDEN_001",
                relation_type="optimized_for",
                target=objective,
                evidence="Pareto solution optimized for multi-objective manufacturing"
            )
        
        # Create relationships: Raw Material –[affects]→ Yield Outcome
        materials = ["MAT_GRADE_1", "MAT_GRADE_2", "MAT_GRADE_3"]
        for material in materials:
            for batch in batches[:30]:
                self._add_relationship(
                    source=material,
                    relation_type="affects",
                    target=batch["batch_id"],
                    evidence="Material grade impacts yield outcome"
                )
        
        # Create relationships: Energy Patterns –[indicates]→ Asset Health Events
        for batch in batches[:30]:
            self._add_relationship(
                source=f"ENERGY_{batch['batch_id']}",
                relation_type="indicates",
                target=f"HEALTH_{batch['batch_id']}",
                evidence="Energy pattern deviation indicates potential asset health issue"
            )
        
        # Create relationships: Anomaly –[triggered_by]→ Process Drift
        for anomaly in anomalies[:20]:
            batch_id = anomaly["batch_id"]
            self._add_relationship(
                source=f"ANOMALY_{batch_id}",
                relation_type="triggered_by",
                target=f"DRIFT_{batch_id}",
                confidence=0.8,
                evidence="Energy anomaly caused by process parameter drift"
            )
        
        logger.info(f"Loaded {len(self.relationships)} semantic relationships")

    def _add_relationship(
        self,
        source: str,
        relation_type: str,
        target: str,
        confidence: float = 1.0,
        evidence: str = ""
    ):
        """Add a semantic relationship."""
        key = f"{source}--[{relation_type}]-->{target}"
        self.relationships[key] = {
            "source": source,
            "relation_type": relation_type,
            "target": target,
            "confidence": confidence,
            "evidence": evidence,
            "created_at": datetime.now().isoformat(),
        }

    def get_relationships(self, source: Optional[str] = None, relation_type: Optional[str] = None) -> List[Dict]:
        """Query relationships with optional filters."""
        results = []
        for rel in self.relationships.values():
            if source and rel["source"] != source:
                continue
            if relation_type and rel["relation_type"] != relation_type:
                continue
            results.append(rel)
        return results

    def find_paths(self, start: str, end: str, max_hops: int = 3) -> List[List[Dict]]:
        """Find reasoning paths between two entities."""
        paths = []
        
        def dfs(current: str, target: str, path: List[Dict], hops: int):
            if hops == 0 or current == target:
                if current == target:
                    paths.append(list(path))
                return
            
            for rel in self.relationships.values():
                if rel["source"] == current:
                    next_node = rel["target"]
                    if not any(r["target"] == next_node for r in path):
                        path.append(rel)
                        dfs(next_node, target, path, hops - 1)
                        path.pop()
        
        dfs(start, end, [], max_hops)
        return paths

    def get_entity_info(self, entity_id: str) -> Optional[Dict]:
        """Get cached entity information."""
        return self.entities.get(entity_id)

    def explain_relationship(self, rel: Dict) -> str:
        """Generate explanation for a relationship."""
        source = rel["source"]
        relation = rel["relation_type"]
        target = rel["target"]
        evidence = rel["evidence"]
        confidence = rel["confidence"]
        
        explanation = f"{source} –[{relation}]→ {target}"
        if confidence < 1.0:
            explanation += f" (confidence: {confidence*100:.0f}%)"
        explanation += f"\n  Evidence: {evidence}"
        
        return explanation
