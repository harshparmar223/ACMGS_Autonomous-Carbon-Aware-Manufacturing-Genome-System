"""
Track A Semantic Layer: Knowledge Graph & Forensics

Lightweight semantic reasoning on top of ACMGS ML pipeline.
Implements Track A relationships without external dependencies.
"""

from .semantic_layer import SemanticGraph
from .forensics import BatchForensics
from .evidence_recommender import EvidenceRecommender

__all__ = ["SemanticGraph", "BatchForensics", "EvidenceRecommender"]
