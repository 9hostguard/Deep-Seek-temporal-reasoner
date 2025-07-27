"""
4D Quantum Temporal Reasoning Engine Core Module
Provides foundational components for temporal reasoning and consciousness simulation.
"""

__version__ = "1.0.0"
__all__ = [
    "QuantumTemporalReasoner",
    "TemporalMemoryMatrix", 
    "ConsciousnessEngine",
    "decompose"
]

from .quantum_temporal_reasoner import QuantumTemporalReasoner
from .temporal_memory_matrix import TemporalMemoryMatrix
from .consciousness_engine import ConsciousnessEngine
from .decomposition import decompose