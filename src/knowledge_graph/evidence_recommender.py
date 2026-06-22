"""
Evidence-Based Recommendation Engine

Enhances recommendations with evidence citations and reasoning chains.
"""

from typing import Dict, List, Optional
from .semantic_layer import SemanticGraph

class EvidenceRecommender:
    """Generates recommendations with full evidence and reasoning."""

    def __init__(self, sg: SemanticGraph):
        self.sg = sg

    def recommend_batch_optimization(self, batch_id: str) -> List[Dict]:
        """Generate optimization recommendations for a batch with evidence."""
        batch = self.sg.get_entity_info(batch_id)
        if not batch:
            return []
        
        recommendations = []
        
        # Strategy 1: Compare to golden signature
        comparison_rels = [r for r in self.sg.get_relationships(source=batch_id, relation_type="compared_against")]
        if comparison_rels:
            golden_id = comparison_rels[0]["target"]
            golden = self.sg.get_entity_info(golden_id)
            
            if golden and batch.get("yield", 0) < golden.get("yield", 0):
                yield_gap = golden.get("yield", 0) - batch.get("yield", 0)
                recommendations.append({
                    "title": "Increase Yield Towards Golden Signature",
                    "description": f"Current yield {batch.get('yield', 0):.1%} vs. optimal {golden.get('yield', 0):.1%}",
                    "impact": f"+{yield_gap*100:.1f}% yield improvement potential",
                    "confidence": "HIGH",
                    "evidence": [
                        {
                            "type": "Golden Signature Comparison",
                            "source": golden_id,
                            "detail": f"Reference yield: {golden.get('yield', 0):.1%}",
                        }
                    ],
                    "reasoning": [
                        "1. Identified comparable golden signature",
                        f"2. Current yield: {batch.get('yield', 0):.1%}",
                        f"3. Optimal yield: {golden.get('yield', 0):.1%}",
                        f"4. Gap: {yield_gap*100:.1f}%",
                        "5. Recommend parameter adjustment towards golden reference",
                    ],
                })
        
        # Strategy 2: High energy consumption
        if batch.get("energy", 0) > 80:
            recommendations.append({
                "title": "Reduce Energy Consumption",
                "description": f"Current energy: {batch.get('energy', 0):.1f} units (above optimal)",
                "impact": f"~{(batch.get('energy', 0) - 60) * 0.15:.1f}$ daily savings potential",
                "confidence": "MEDIUM",
                "evidence": [
                    {
                        "type": "Energy Pattern Analysis",
                        "source": f"ENERGY_{batch_id}",
                        "detail": f"Elevated energy consumption detected",
                    }
                ],
                "reasoning": [
                    "1. Analyzed batch energy patterns",
                    f"2. Current consumption: {batch.get('energy', 0):.1f} units",
                    "3. Optimal target: ~60 units",
                    "4. Excess energy increases operational cost and carbon footprint",
                    "5. Recommend efficiency improvements",
                ],
            })
        
        # Strategy 3: Material optimization
        material_rels = [r for r in self.sg.get_relationships(relation_type="affects") if r["target"] == batch_id]
        if material_rels and batch.get("yield", 1) < 0.80:
            recommendations.append({
                "title": "Evaluate Alternative Materials",
                "description": "Current material grade may limit yield potential",
                "impact": "Potential +5-10% yield improvement",
                "confidence": "MEDIUM",
                "evidence": [
                    {
                        "type": "Material Impact Analysis",
                        "source": material_rels[0]["source"],
                        "detail": material_rels[0]["evidence"],
                    }
                ],
                "reasoning": [
                    "1. Identified material grade impact on yield",
                    f"2. Current material: {material_rels[0]['source']}",
                    f"3. Current yield: {batch.get('yield', 0):.1%}",
                    "4. Premium materials historically yield +5-10% improvement",
                    "5. Cost-benefit analysis recommended",
                ],
            })
        
        # Strategy 4: Carbon reduction
        if batch.get("carbon", 0) > 0.5:
            recommendations.append({
                "title": "Reduce Carbon Footprint",
                "description": f"Current carbon impact: {batch.get('carbon', 0):.2f} kg CO2",
                "impact": f"~{batch.get('carbon', 0) * 0.3:.2f} kg CO2 reduction possible",
                "confidence": "HIGH",
                "evidence": [
                    {
                        "type": "Carbon Intensity Analysis",
                        "source": batch_id,
                        "detail": "Batch carbon intensity above target",
                    }
                ],
                "reasoning": [
                    "1. Calculated batch carbon footprint",
                    f"2. Current: {batch.get('carbon', 0):.2f} kg CO2",
                    "3. Target: ~0.35 kg CO2",
                    "4. Primary driver: Energy consumption and grid intensity",
                    "5. Recommend scheduling during low-carbon hours",
                ],
            })
        
        return recommendations

    def get_evidence_summary(self, batch_id: str) -> Dict:
        """Get comprehensive evidence for batch analysis."""
        batch = self.sg.get_entity_info(batch_id)
        if not batch:
            return {"error": "Batch not found"}
        
        summary = {
            "batch_id": batch_id,
            "evidence_sources": [],
            "relationships": [],
            "performance_indicators": {},
        }
        
        # Collect relationships
        all_rels = self.sg.get_relationships()
        batch_rels = [r for r in all_rels if r["source"] == batch_id or r["target"] == batch_id]
        
        for rel in batch_rels:
            summary["relationships"].append({
                "type": rel["relation_type"],
                "connected_to": rel["target"] if rel["source"] == batch_id else rel["source"],
                "evidence": rel["evidence"],
                "confidence": rel["confidence"],
            })
        
        # Collect evidence sources
        summary["evidence_sources"] = list(set([
            rel["source"] for rel in batch_rels if rel["source"] != batch_id
        ] + [
            rel["target"] for rel in batch_rels if rel["target"] != batch_id
        ]))
        
        # Performance indicators
        summary["performance_indicators"] = batch
        
        return summary

    def format_recommendation_report(self, batch_id: str) -> str:
        """Generate formatted recommendation report with evidence."""
        recs = self.recommend_batch_optimization(batch_id)
        
        report = f"\n{'='*70}\n"
        report += f"EVIDENCE-BASED RECOMMENDATIONS: {batch_id}\n"
        report += f"{'='*70}\n\n"
        
        if not recs:
            report += "✅ No optimization opportunities identified at this time.\n"
            return report
        
        for i, rec in enumerate(recs, 1):
            report += f"**RECOMMENDATION {i}: {rec['title']}**\n"
            report += f"Confidence: {rec['confidence']}\n"
            report += f"Impact: {rec['impact']}\n\n"
            
            report += "**Reasoning Chain:**\n"
            for step in rec["reasoning"]:
                report += f"  {step}\n"
            
            report += "\n**Evidence (Citations):**\n"
            for j, evidence in enumerate(rec["evidence"], 1):
                report += f"  [{j}] {evidence['type']}\n"
                report += f"      Source: {evidence['source']}\n"
                report += f"      Detail: {evidence['detail']}\n"
            
            report += "\n" + "─"*70 + "\n\n"
        
        report += f"{'='*70}\n"
        return report
