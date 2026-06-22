"""
Knowledge Graph Population

Populates the knowledge graph from ACMGS database batches, predictions, and simulations.
Implements the Track A relationships.
"""

import json
from typing import Dict, List, Optional
import numpy as np

from src.database.manager import get_connection
from src.utils.logger import get_logger
from .graph_store import KnowledgeGraphStore

logger = get_logger("kg_population")


class KnowledgeGraphPopulator:
    """Populates manufacturing knowledge graph from production data."""

    def __init__(self, kg_store: KnowledgeGraphStore):
        self.kg = kg_store
        self.top_quartile_yield = 0.85
        self.top_quartile_quality = 0.85

    def populate_from_database(self, limit: Optional[int] = None) -> Dict:
        """Populate entire graph from database."""
        logger.info("Starting knowledge graph population...")
        
        stats = {
            "assets_created": 0,
            "batches_created": 0,
            "energy_patterns_created": 0,
            "golden_signatures_created": 0,
            "materials_created": 0,
            "anomalies_created": 0,
            "relationships_created": 0,
        }
        
        # 1. Create Asset entities (manufacturing equipment)
        stats["assets_created"] = self._create_assets()
        
        # 2. Create Batch entities from database
        stats["batches_created"], batch_ids = self._create_batches(limit)
        
        # 3. Create Asset → Produces → Batch relationships
        p_count = self._create_produces_relationships(batch_ids)
        stats["relationships_created"] += p_count
        
        # 4. Create Energy Pattern entities and relationships
        stats["energy_patterns_created"], energy_pattern_ids = self._create_energy_patterns(batch_ids)
        ep_count = self._create_has_relationships(batch_ids, energy_pattern_ids)
        stats["relationships_created"] += ep_count
        
        # 5. Create Material entities and relationships
        stats["materials_created"], material_ids = self._create_materials()
        mat_count = self._create_material_affects_relationships(batch_ids, material_ids)
        stats["relationships_created"] += mat_count
        
        # 6. Create Golden Signature entities (best performing batches)
        stats["golden_signatures_created"], golden_ids = self._create_golden_signatures(batch_ids)
        cmp_count = self._create_comparison_relationships(batch_ids, golden_ids)
        stats["relationships_created"] += cmp_count
        
        # 7. Create Anomaly entities from energy anomalies
        stats["anomalies_created"], anomaly_ids = self._create_anomalies(batch_ids, energy_pattern_ids)
        anom_count = self._create_anomaly_trigger_relationships(batch_ids, anomaly_ids)
        stats["relationships_created"] += anom_count
        
        # 8. Create influences relationships (Process Parameters → Energy Patterns)
        inf_count = self._create_influences_relationships(batch_ids, energy_pattern_ids)
        stats["relationships_created"] += inf_count
        
        logger.info(f"Population complete: {stats}")
        return stats

    # ─── Asset Entity Creation ────────────────────────────────────────────────

    def _create_assets(self) -> int:
        """Create Asset entities representing manufacturing equipment."""
        # Generic assets - in real scenario would come from asset registry
        assets = [
            {"asset_id": "ASSET_001", "name": "Injection Molder A", "type": "Molding Machine", "uptime": 0.95},
            {"asset_id": "ASSET_002", "name": "Injection Molder B", "type": "Molding Machine", "uptime": 0.92},
            {"asset_id": "ASSET_003", "name": "CNC Machinist", "type": "CNC Machine", "uptime": 0.88},
            {"asset_id": "ASSET_004", "name": "Assembly Line 1", "type": "Assembly Station", "uptime": 0.97},
            {"asset_id": "ASSET_005", "name": "Quality Oven", "type": "Curing Station", "uptime": 0.99},
        ]
        
        for asset in assets:
            self.kg.create_entity(
                entity_id=asset["asset_id"],
                entity_type="Asset",
                name=asset["name"],
                attributes={
                    "equipment_type": asset["type"],
                    "uptime_percent": asset["uptime"] * 100,
                    "failure_rate": (1 - asset["uptime"]) * 100,
                }
            )
        
        logger.info(f"Created {len(assets)} asset entities")
        return len(assets)

    # ─── Batch Entity Creation ────────────────────────────────────────────────

    def _create_batches(self, limit: Optional[int] = None) -> tuple[int, List[str]]:
        """Create Batch entities from database."""
        with get_connection() as conn:
            query = "SELECT batch_id, temperature, pressure, speed, feed_rate, humidity, yield, quality, energy_consumption, carbon_intensity FROM batches"
            
            if limit:
                query += f" LIMIT {limit}"
            
            rows = conn.execute(query).fetchall()
        
        batch_ids = []
        for row in rows:
            batch_id = row["batch_id"]
            batch_ids.append(batch_id)
            
            self.kg.create_entity(
                entity_id=batch_id,
                entity_type="Batch",
                name=batch_id,
                attributes={
                    "temperature": float(row["temperature"]),
                    "pressure": float(row["pressure"]),
                    "speed": float(row["speed"]),
                    "feed_rate": float(row["feed_rate"]),
                    "humidity": float(row["humidity"]),
                    "yield": float(row["yield"]),
                    "quality": float(row["quality"]),
                    "energy_consumption": float(row["energy_consumption"]),
                    "carbon_intensity": float(row["carbon_intensity"]),
                }
            )
        
        logger.info(f"Created {len(batch_ids)} batch entities")
        return len(batch_ids), batch_ids

    # ─── Golden Signature Creation ────────────────────────────────────────────

    def _create_golden_signatures(self, batch_ids: List[str]) -> tuple[int, List[str]]:
        """Create Golden Signature entities from top-quartile batches."""
        if not batch_ids:
            return 0, []
        
        # Get batch data for ranking
        with get_connection() as conn:
            rows = conn.execute(
                f"SELECT batch_id, yield, quality, energy_consumption FROM batches WHERE batch_id IN ({','.join('?' * len(batch_ids))})",
                batch_ids
            ).fetchall()
        
        # Calculate composite score and identify top quartile
        batches_with_score = []
        for row in rows:
            score = (float(row["yield"]) * 0.4 + 
                    float(row["quality"]) * 0.4 - 
                    float(row["energy_consumption"]) / 1000 * 0.2)
            batches_with_score.append((row["batch_id"], score, float(row["yield"]), float(row["quality"])))
        
        batches_with_score.sort(key=lambda x: x[1], reverse=True)
        top_quartile = batches_with_score[:max(1, len(batches_with_score) // 4)]
        
        golden_ids = []
        for i, (batch_id, score, yield_val, quality_val) in enumerate(top_quartile):
            golden_id = f"GOLDEN_{i:03d}"
            golden_ids.append(golden_id)
            
            batch_ent = self.kg.get_entity(batch_id)
            if batch_ent:
                self.kg.create_entity(
                    entity_id=golden_id,
                    entity_type="Golden Signature",
                    name=f"Golden Reference {i}",
                    attributes={
                        "source_batch": batch_id,
                        "composite_score": float(score),
                        "yield": float(yield_val),
                        "quality": float(quality_val),
                        **batch_ent["attributes"]
                    }
                )
        
        logger.info(f"Created {len(golden_ids)} golden signature entities")
        return len(golden_ids), golden_ids

    # ─── Material Entity Creation ─────────────────────────────────────────────

    def _create_materials(self) -> tuple[int, List[str]]:
        """Create Raw Material entities."""
        materials = [
            {"id": "MAT_GRADE_1", "name": "Premium Grade Polymer", "grade": 1, "impact": 0.05},
            {"id": "MAT_GRADE_2", "name": "Standard Grade Polymer", "grade": 2, "impact": -0.02},
            {"id": "MAT_GRADE_3", "name": "Economy Grade Polymer", "grade": 3, "impact": -0.15},
            {"id": "MAT_REINFORCED", "name": "Fiber-Reinforced Composite", "grade": 1.5, "impact": 0.08},
        ]
        
        material_ids = []
        for mat in materials:
            material_ids.append(mat["id"])
            self.kg.create_entity(
                entity_id=mat["id"],
                entity_type="Raw Material",
                name=mat["name"],
                attributes={
                    "grade": mat["grade"],
                    "yield_impact": mat["impact"],
                }
            )
        
        logger.info(f"Created {len(material_ids)} material entities")
        return len(material_ids), material_ids

    # ─── Energy Pattern Creation ──────────────────────────────────────────────

    def _create_energy_patterns(self, batch_ids: List[str]) -> tuple[int, List[str]]:
        """Create Energy Pattern entities from energy embeddings."""
        with get_connection() as conn:
            rows = conn.execute(
                f"SELECT batch_id, embedding, is_anomaly, recon_error FROM energy_embeddings WHERE batch_id IN ({','.join('?' * len(batch_ids))})",
                batch_ids
            ).fetchall()
        
        energy_pattern_ids = []
        for row in rows:
            batch_id = row["batch_id"]
            pattern_id = f"ENERGY_{batch_id}"
            energy_pattern_ids.append(pattern_id)
            
            try:
                embedding = json.loads(row["embedding"])
                recon_error = float(row["recon_error"]) if row["recon_error"] else 0.0
            except:
                embedding = [0.0] * 16
                recon_error = 0.0
            
            self.kg.create_entity(
                entity_id=pattern_id,
                entity_type="Energy Pattern",
                name=f"Energy Signature for {batch_id}",
                attributes={
                    "source_batch": batch_id,
                    "reconstruction_error": recon_error,
                    "is_anomaly": bool(row["is_anomaly"]),
                    "embedding_magnitude": float(np.linalg.norm(embedding)) if embedding else 0.0,
                }
            )
        
        logger.info(f"Created {len(energy_pattern_ids)} energy pattern entities")
        return len(energy_pattern_ids), energy_pattern_ids

    # ─── Anomaly Entity Creation ──────────────────────────────────────────────

    def _create_anomalies(self, batch_ids: List[str], energy_pattern_ids: List[str]) -> tuple[int, List[str]]:
        """Create Anomaly entities from detected energy anomalies."""
        anomaly_ids = []
        anomaly_count = 0
        
        with get_connection() as conn:
            rows = conn.execute(
                f"SELECT batch_id, is_anomaly, recon_error FROM energy_embeddings WHERE batch_id IN ({','.join('?' * len(batch_ids))}) AND is_anomaly = 1",
                batch_ids
            ).fetchall()
        
        for row in rows:
            batch_id = row["batch_id"]
            anomaly_id = f"ANOMALY_{batch_id}_{anomaly_count}"
            anomaly_ids.append(anomaly_id)
            anomaly_count += 1
            
            recon_error = float(row["recon_error"]) if row["recon_error"] else 0.0
            
            self.kg.create_entity(
                entity_id=anomaly_id,
                entity_type="Anomaly",
                name=f"Energy Anomaly in {batch_id}",
                attributes={
                    "affected_batch": batch_id,
                    "anomaly_type": "energy_signature_deviation",
                    "reconstruction_error": recon_error,
                    "severity": min(1.0, recon_error / 0.5),  # Normalized severity
                }
            )
        
        logger.info(f"Created {len(anomaly_ids)} anomaly entities")
        return len(anomaly_ids), anomaly_ids

    # ─── Relationship Creation ────────────────────────────────────────────────

    def _create_produces_relationships(self, batch_ids: List[str]) -> int:
        """Create Asset –[Produces]→ Batch relationships."""
        count = 0
        assets = [f"ASSET_{i:03d}" for i in range(1, 6)]
        
        for batch_id in batch_ids:
            # Randomly assign batch to an asset
            import hashlib
            asset_idx = int(hashlib.md5(batch_id.encode()).hexdigest(), 16) % len(assets)
            asset_id = assets[asset_idx]
            
            self.kg.create_relationship(
                source_id=asset_id,
                relation_type="Produces",
                target_id=batch_id,
                confidence=1.0,
                evidence=[f"Batch {batch_id} produced by {asset_id}"]
            )
            count += 1
        
        logger.info(f"Created {count} 'Produces' relationships")
        return count

    def _create_has_relationships(self, batch_ids: List[str], energy_pattern_ids: List[str]) -> int:
        """Create Batch –[Has]→ Energy Pattern relationships."""
        count = 0
        for i, batch_id in enumerate(batch_ids):
            if i < len(energy_pattern_ids):
                self.kg.create_relationship(
                    source_id=batch_id,
                    relation_type="Has",
                    target_id=energy_pattern_ids[i],
                    confidence=1.0,
                    evidence=["Energy pattern analyzed from batch telemetry"]
                )
                count += 1
        
        logger.info(f"Created {count} 'Has' relationships")
        return count

    def _create_comparison_relationships(self, batch_ids: List[str], golden_ids: List[str]) -> int:
        """Create Batch –[compared_against]→ Golden Signature relationships."""
        count = 0
        if not golden_ids:
            return count
        
        # Compare each batch to a random golden
        import random
        for batch_id in batch_ids:
            golden_id = random.choice(golden_ids)
            confidence = 0.8 + random.random() * 0.2  # 0.8-1.0
            
            self.kg.create_relationship(
                source_id=batch_id,
                relation_type="compared_against",
                target_id=golden_id,
                confidence=confidence,
                evidence=["Batch performance compared to golden reference"]
            )
            count += 1
        
        logger.info(f"Created {count} 'compared_against' relationships")
        return count

    def _create_material_affects_relationships(self, batch_ids: List[str], material_ids: List[str]) -> int:
        """Create Raw Material –[affects]→ Yield Outcome relationships."""
        count = 0
        if not material_ids or not batch_ids:
            return count
        
        import random
        with get_connection() as conn:
            batch_data = conn.execute(
                f"SELECT batch_id, material_grade, yield FROM batches WHERE batch_id IN ({','.join('?' * len(batch_ids))})",
                batch_ids
            ).fetchall()
        
        for row in batch_data:
            batch_id = row["batch_id"]
            material_grade = int(row["material_grade"])
            yield_val = float(row["yield"])
            
            # Map material grade to material entity
            if material_grade == 1:
                material_id = "MAT_GRADE_1"
            elif material_grade == 3:
                material_id = "MAT_GRADE_3"
            else:
                material_id = "MAT_GRADE_2"
            
            if material_id in material_ids:
                self.kg.create_relationship(
                    source_id=material_id,
                    relation_type="affects",
                    target_id=batch_id,
                    properties={
                        "impact_magnitude": yield_val - 0.8,
                        "impact_type": "positive" if yield_val > 0.85 else "negative",
                    },
                    confidence=0.9,
                    evidence=[f"Material grade {material_grade} affects yield {yield_val}"]
                )
                count += 1
        
        logger.info(f"Created {count} 'affects' relationships")
        return count

    def _create_anomaly_trigger_relationships(self, batch_ids: List[str], anomaly_ids: List[str]) -> int:
        """Create Anomaly –[triggered_by]→ Process Drift relationships."""
        count = 0
        if not anomaly_ids:
            return count
        
        for anomaly_id in anomaly_ids:
            anomaly = self.kg.get_entity(anomaly_id)
            if anomaly and "affected_batch" in anomaly.get("attributes", {}):
                batch_id = anomaly["attributes"]["affected_batch"]
                drift_id = f"DRIFT_{batch_id}"
                
                # Create process drift entity if needed
                self.kg.create_entity(
                    entity_id=drift_id,
                    entity_type="Process Drift",
                    name=f"Parameter Drift in {batch_id}",
                    attributes={"source_batch": batch_id}
                )
                
                self.kg.create_relationship(
                    source_id=anomaly_id,
                    relation_type="triggered_by",
                    target_id=drift_id,
                    confidence=0.8,
                    evidence=["Energy anomaly caused by parameter drift"]
                )
                count += 1
        
        logger.info(f"Created {count} 'triggered_by' relationships")
        return count

    def _create_influences_relationships(self, batch_ids: List[str], energy_pattern_ids: List[str]) -> int:
        """Create Process Parameters –[influences]→ Energy Patterns relationships."""
        count = 0
        
        # Create process parameter entities
        parameters = [
            "temperature", "pressure", "speed", "feed_rate", "humidity"
        ]
        
        for param in parameters:
            param_id = f"PARAM_{param.upper()}"
            self.kg.create_entity(
                entity_id=param_id,
                entity_type="Process Parameters",
                name=f"Process Parameter: {param}",
                attributes={"parameter_name": param}
            )
        
        # Create influences relationships
        batch_data = {}
        with get_connection() as conn:
            rows = conn.execute(
                f"SELECT batch_id, temperature, pressure, speed, feed_rate, humidity FROM batches WHERE batch_id IN ({','.join('?' * len(batch_ids))})",
                batch_ids
            ).fetchall()
            
            for row in rows:
                batch_data[row["batch_id"]] = row
        
        for i, batch_id in enumerate(batch_ids):
            if i < len(energy_pattern_ids) and batch_id in batch_data:
                energy_pattern_id = energy_pattern_ids[i]
                row = batch_data[batch_id]
                
                for param in parameters:
                    param_id = f"PARAM_{param.upper()}"
                    param_value = float(row[param]) if row[param] else 0
                    
                    self.kg.create_relationship(
                        source_id=param_id,
                        relation_type="influences",
                        target_id=energy_pattern_id,
                        properties={
                            "parameter_value": param_value,
                            "strength": "high" if param_value > 50 else "moderate" if param_value > 30 else "low"
                        },
                        confidence=0.85,
                        evidence=[f"{param}={param_value} influences energy pattern"]
                    )
                    count += 1
        
        logger.info(f"Created {count} 'influences' relationships")
        return count
