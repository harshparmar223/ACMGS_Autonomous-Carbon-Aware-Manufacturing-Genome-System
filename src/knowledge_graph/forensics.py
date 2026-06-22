"""
Batch Forensics: Why-Why Analysis

Explains batch performance and issues through semantic reasoning.
"""

from typing import Dict, List, Optional
from src.database.manager import get_connection
from src.utils.logger import get_logger
from .semantic_layer import SemanticGraph

logger = get_logger("batch_forensics")


class BatchForensics:
    """Conversational why-why analysis for batch performance."""

    def __init__(self, sg: SemanticGraph):
        self.sg = sg

    def analyze_batch(self, batch_id: str) -> Dict:
        """Perform comprehensive why-why analysis on a batch."""
        
        batch_info = self.sg.get_entity_info(batch_id)
        if not batch_info:
            return {"error": f"Batch {batch_id} not found"}
        
        analysis = {
            "batch_id": batch_id,
            "summary": f"Batch {batch_id} Analysis",
            "performance": {
                "yield": batch_info.get("yield"),
                "quality": batch_info.get("quality"),
                "energy": batch_info.get("energy"),
                "carbon": batch_info.get("carbon"),
            },
            "why_questions": [],
        }
        
        # Why 1: What produced this batch?
        producer_rels = [r for r in self.sg.get_relationships(relation_type="Produces") if r["target"] == batch_id]
        if producer_rels:
            analysis["why_questions"].append({
                "question": "Why was this asset selected to produce this batch?",
                "answer": f"Batch was assigned to {producer_rels[0]['source']} based on production schedule.",
                "relationship": producer_rels[0],
            })
        
        # Why 2: What energy patterns characterized this batch?
        energy_rels = [r for r in self.sg.get_relationships(source=batch_id, relation_type="Has")]
        if energy_rels:
            analysis["why_questions"].append({
                "question": "What was the energy signature of this batch?",
                "answer": f"Batch exhibited energy pattern {energy_rels[0]['target']}.",
                "relationship": energy_rels[0],
            })
        
        # Why 3: How does this batch compare to golden?
        comparison_rels = [r for r in self.sg.get_relationships(source=batch_id, relation_type="compared_against")]
        if comparison_rels:
            golden_id = comparison_rels[0]["target"]
            golden = self.sg.get_entity_info(golden_id)
            
            delta_yield = (batch_info.get("yield", 0) - golden.get("yield", 0)) if golden else 0
            analysis["why_questions"].append({
                "question": "How does this batch compare to the golden signature?",
                "answer": f"Batch yield is {delta_yield:+.2%} vs. golden reference. {'Above' if delta_yield > 0 else 'Below'} optimal.",
                "relationship": comparison_rels[0],
                "golden_id": golden_id,
            })
        
        # Why 4: What materials were used?
        material_rels = [r for r in self.sg.get_relationships(relation_type="affects") if r["target"] == batch_id]
        if material_rels:
            analysis["why_questions"].append({
                "question": "What materials affected this batch's performance?",
                "answer": f"Batch used {material_rels[0]['source']}, which affects yield outcome.",
                "relationship": material_rels[0],
            })
        
        # Why 5: Were there any anomalies?
        anomaly_rels = [r for r in self.sg.get_relationships(source=f"ANOMALY_{batch_id}")]
        if anomaly_rels:
            analysis["why_questions"].append({
                "question": "Why did anomalies occur in this batch?",
                "answer": f"Anomalies detected, triggered by {anomaly_rels[0]['target']} (process drift).",
                "relationship": anomaly_rels[0],
                "severity": "high",
            })
        
        analysis["narrative"] = self._generate_narrative(analysis)
        return analysis

    def compare_batches(self, batch1_id: str, batch2_id: str) -> Dict:
        """Compare two batches using semantic relationships."""
        b1 = self.sg.get_entity_info(batch1_id)
        b2 = self.sg.get_entity_info(batch2_id)
        
        if not b1 or not b2:
            return {"error": "One or both batches not found"}
        
        return {
            "batch1": batch1_id,
            "batch2": batch2_id,
            "comparison": {
                "yield_diff": (b2.get("yield", 0) - b1.get("yield", 0)),
                "quality_diff": (b2.get("quality", 0) - b1.get("quality", 0)),
                "energy_diff": (b1.get("energy", 0) - b2.get("energy", 0)),  # Lower is better
                "carbon_diff": (b1.get("carbon", 0) - b2.get("carbon", 0)),  # Lower is better
            },
            "winner": self._determine_winner(b1, b2),
        }

    def trace_root_cause(self, batch_id: str) -> Dict:
        """Trace root causes of poor batch performance."""
        batch = self.sg.get_entity_info(batch_id)
        if not batch:
            return {"error": f"Batch {batch_id} not found"}
        
        causes = {
            "batch_id": batch_id,
            "performance_score": self._calculate_performance_score(batch),
            "root_causes": [],
        }
        
        # Check for material issues
        material_rels = [r for r in self.sg.get_relationships(relation_type="affects") if r["target"] == batch_id]
        if material_rels and batch.get("yield", 1) < 0.75:
            for rel in material_rels:
                causes["root_causes"].append({
                    "cause": f"Suboptimal material selection: {rel['source']}",
                    "impact": "Low yield",
                    "evidence": rel["evidence"],
                    "confidence": rel["confidence"],
                })
        
        # Check for energy efficiency
        if batch.get("energy", float('inf')) > 100:
            causes["root_causes"].append({
                "cause": "High energy consumption during process",
                "impact": "Increased operating cost and carbon footprint",
                "evidence": "Energy embeddings show elevated power signature",
                "confidence": 0.85,
            })
        
        # Check for anomalies
        anomaly_rels = [r for r in self.sg.get_relationships(source=f"ANOMALY_{batch_id}")]
        if anomaly_rels:
            for rel in anomaly_rels:
                causes["root_causes"].append({
                    "cause": f"Process drift detected: {rel['target']}",
                    "impact": "Quality degradation",
                    "evidence": rel["evidence"],
                    "confidence": rel["confidence"],
                })
        
        causes["narrative"] = self._generate_root_cause_narrative(causes)
        return causes

    # ──── Private Helper Methods ────────────────────────────────────────────

    def _generate_narrative(self, analysis: Dict) -> str:
        """Generate readable narrative from why-why analysis."""
        narrative = f"\n**BATCH FORENSICS REPORT: {analysis['batch_id']}**\n"
        narrative += f"**Performance Metrics:**\n"
        perf = analysis["performance"]
        narrative += f"  • Yield: {perf['yield']:.2%}\n"
        narrative += f"  • Quality: {perf['quality']:.2%}\n"
        narrative += f"  • Energy: {perf['energy']:.1f} units\n"
        narrative += f"  • Carbon Impact: {perf['carbon']:.2f} kg CO2\n\n"
        
        narrative += "**Why-Why Analysis:**\n"
        for i, q in enumerate(analysis["why_questions"], 1):
            narrative += f"\n**Q{i}: {q['question']}**\n"
            narrative += f"A: {q['answer']}\n"
        
        return narrative

    def _generate_root_cause_narrative(self, causes: Dict) -> str:
        """Generate narrative for root cause analysis."""
        narrative = f"\n**ROOT CAUSE ANALYSIS: {causes['batch_id']}**\n"
        narrative += f"**Performance Score: {causes['performance_score']:.0%}**\n\n"
        
        if not causes["root_causes"]:
            narrative += "✅ No major root causes identified. Batch performed within acceptable ranges.\n"
        else:
            narrative += f"**{len(causes['root_causes'])} Root Cause(s) Identified:**\n\n"
            for i, cause in enumerate(causes["root_causes"], 1):
                narrative += f"**{i}. {cause['cause']}**\n"
                narrative += f"   Impact: {cause['impact']}\n"
                narrative += f"   Evidence: {cause['evidence']}\n"
                narrative += f"   Confidence: {cause['confidence']*100:.0f}%\n\n"
        
        return narrative

    def _calculate_performance_score(self, batch: Dict) -> float:
        """Calculate composite performance score (0-1)."""
        yield_score = batch.get("yield", 0) * 0.4
        quality_score = batch.get("quality", 0) * 0.35
        energy_score = max(0, 1 - batch.get("energy", 100) / 100) * 0.15
        carbon_score = max(0, 1 - batch.get("carbon", 100) / 100) * 0.1
        
        return yield_score + quality_score + energy_score + carbon_score

    def _determine_winner(self, b1: Dict, b2: Dict) -> str:
        """Determine which batch performed better."""
        score1 = self._calculate_performance_score(b1)
        score2 = self._calculate_performance_score(b2)
        
        if score1 > score2:
            return "batch1"
        elif score2 > score1:
            return "batch2"
        else:
            return "tie"
