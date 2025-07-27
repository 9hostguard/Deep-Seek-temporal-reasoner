"""
Temporal Memory Matrix - Persistent dimensional memory system.
"""

import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import json
import pickle
from collections import defaultdict


class TemporalMemoryMatrix:
    """
    4D memory matrix for storing and retrieving temporal reasoning states.
    Implements persistent memory across temporal dimensions with quantum coherence.
    """
    
    def __init__(self, dimensions: int = 4, memory_capacity: int = 10000):
        """
        Initialize temporal memory matrix.
        
        Args:
            dimensions: Number of temporal dimensions
            memory_capacity: Maximum number of memory states to store
        """
        self.dimensions = dimensions
        self.memory_capacity = memory_capacity
        
        # Core memory structures
        self.memory_states = {}  # Key -> TemporalState
        self.dimensional_index = defaultdict(list)  # Dimension -> [keys]
        self.temporal_graph = defaultdict(list)  # Key -> [related_keys]
        self.access_patterns = defaultdict(int)  # Key -> access_count
        
        # Quantum coherence tracking
        self.coherence_matrix = np.eye(dimensions) * 0.85
        self.entanglement_pairs = []
        
        # Memory analytics
        self.analytics = {
            "total_states": 0,
            "coherence_evolution": [],
            "pattern_frequencies": defaultdict(int),
            "dimensional_activity": defaultdict(int)
        }
    
    async def store_temporal_state(self, 
                                 prompt: str,
                                 temporal_breakdown: Dict[str, Any],
                                 consciousness_level: float) -> str:
        """
        Store a temporal reasoning state in memory matrix.
        
        Args:
            prompt: Original prompt
            temporal_breakdown: Decomposed temporal segments
            consciousness_level: Current consciousness level
            
        Returns:
            Memory key for retrieval
        """
        memory_key = self._generate_memory_key(prompt)
        timestamp = datetime.now(timezone.utc)
        
        # Create temporal state
        temporal_state = {
            "key": memory_key,
            "prompt": prompt,
            "temporal_breakdown": temporal_breakdown,
            "consciousness_level": consciousness_level,
            "timestamp": timestamp,
            "access_count": 0,
            "coherence_score": temporal_breakdown.get("quantum_coherence", 0.5),
            "dimensional_weights": self._calculate_dimensional_weights(temporal_breakdown)
        }
        
        # Store in memory matrix
        self.memory_states[memory_key] = temporal_state
        
        # Update dimensional indices
        for dimension in temporal_breakdown["temporal_segments"].keys():
            self.dimensional_index[dimension].append(memory_key)
            self.analytics["dimensional_activity"][dimension] += 1
        
        # Update coherence matrix
        await self._update_coherence_matrix(temporal_state)
        
        # Manage memory capacity
        if len(self.memory_states) > self.memory_capacity:
            await self._prune_memory()
        
        self.analytics["total_states"] += 1
        
        return memory_key
    
    async def retrieve_temporal_state(self, memory_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a temporal state from memory."""
        if memory_key not in self.memory_states:
            return None
        
        state = self.memory_states[memory_key]
        state["access_count"] += 1
        self.access_patterns[memory_key] += 1
        
        return state.copy()
    
    async def find_similar_states(self, 
                                prompt: str,
                                temporal_breakdown: Dict[str, Any],
                                similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find temporally similar states in memory."""
        current_weights = self._calculate_dimensional_weights(temporal_breakdown)
        similar_states = []
        
        for key, state in self.memory_states.items():
            similarity = await self._calculate_temporal_similarity(
                current_weights, state["dimensional_weights"]
            )
            
            if similarity >= similarity_threshold:
                similar_states.append({
                    "state": state,
                    "similarity": similarity
                })
        
        # Sort by similarity score
        similar_states.sort(key=lambda x: x["similarity"], reverse=True)
        return similar_states[:10]  # Return top 10 similar states
    
    async def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze temporal reasoning patterns in memory."""
        if not self.memory_states:
            return {"error": "No memory states to analyze"}
        
        # Temporal distribution analysis
        temporal_distribution = defaultdict(int)
        consciousness_levels = []
        coherence_scores = []
        
        for state in self.memory_states.values():
            for dimension, content in state["temporal_breakdown"]["temporal_segments"].items():
                if content.strip():
                    temporal_distribution[dimension] += 1
            
            consciousness_levels.append(state["consciousness_level"])
            coherence_scores.append(state["coherence_score"])
        
        # Pattern frequency analysis
        pattern_analysis = {
            "temporal_distribution": dict(temporal_distribution),
            "average_consciousness": np.mean(consciousness_levels),
            "consciousness_std": np.std(consciousness_levels),
            "average_coherence": np.mean(coherence_scores),
            "coherence_std": np.std(coherence_scores),
            "most_accessed": sorted(
                [(k, v) for k, v in self.access_patterns.items()],
                key=lambda x: x[1], reverse=True
            )[:5],
            "dimensional_activity": dict(self.analytics["dimensional_activity"]),
            "total_states": len(self.memory_states)
        }
        
        return pattern_analysis
    
    def _generate_memory_key(self, prompt: str) -> str:
        """Generate unique memory key for prompt."""
        import hashlib
        timestamp = str(datetime.now(timezone.utc).timestamp())
        content = f"{prompt}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _calculate_dimensional_weights(self, temporal_breakdown: Dict[str, Any]) -> np.ndarray:
        """Calculate weight vector for temporal dimensions."""
        segments = temporal_breakdown["temporal_segments"]
        confidence_matrix = temporal_breakdown.get("confidence_matrix", {})
        
        weights = np.zeros(self.dimensions)
        
        dimension_map = {"past": 0, "present": 1, "future": 2}
        
        for dimension, content in segments.items():
            if dimension in dimension_map and content.strip():
                dim_idx = dimension_map[dimension]
                content_weight = len(content.split()) / 100.0  # Normalize by word count
                confidence_weight = confidence_matrix.get(dimension, 0.5)
                weights[dim_idx] = content_weight * confidence_weight
        
        # Normalize weights
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        
        return weights
    
    async def _calculate_temporal_similarity(self, 
                                           weights1: np.ndarray, 
                                           weights2: np.ndarray) -> float:
        """Calculate similarity between temporal weight vectors."""
        # Use cosine similarity
        dot_product = np.dot(weights1, weights2)
        norm1 = np.linalg.norm(weights1)
        norm2 = np.linalg.norm(weights2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def _update_coherence_matrix(self, temporal_state: Dict[str, Any]):
        """Update quantum coherence matrix based on new temporal state."""
        coherence_score = temporal_state["coherence_score"]
        
        # Update coherence matrix with exponential moving average
        alpha = 0.1  # Learning rate
        self.coherence_matrix = (1 - alpha) * self.coherence_matrix + alpha * np.eye(self.dimensions) * coherence_score
        
        # Track coherence evolution
        self.analytics["coherence_evolution"].append({
            "timestamp": temporal_state["timestamp"].isoformat(),
            "coherence": coherence_score,
            "matrix_trace": np.trace(self.coherence_matrix)
        })
    
    async def _prune_memory(self):
        """Remove least accessed memory states to maintain capacity."""
        # Sort by access count (ascending) and remove bottom 10%
        sorted_states = sorted(
            self.memory_states.items(),
            key=lambda x: (x[1]["access_count"], x[1]["timestamp"])
        )
        
        prune_count = max(1, len(sorted_states) // 10)
        
        for i in range(prune_count):
            key, state = sorted_states[i]
            
            # Remove from memory
            del self.memory_states[key]
            
            # Remove from dimensional indices
            for dimension in state["temporal_breakdown"]["temporal_segments"].keys():
                if key in self.dimensional_index[dimension]:
                    self.dimensional_index[dimension].remove(key)
            
            # Remove from access patterns
            if key in self.access_patterns:
                del self.access_patterns[key]
    
    async def export_memory_matrix(self, filepath: str):
        """Export memory matrix to file for persistence."""
        export_data = {
            "memory_states": {k: self._serialize_state(v) for k, v in self.memory_states.items()},
            "dimensional_index": dict(self.dimensional_index),
            "coherence_matrix": self.coherence_matrix.tolist(),
            "analytics": dict(self.analytics)
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    async def import_memory_matrix(self, filepath: str):
        """Import memory matrix from file."""
        with open(filepath, 'r') as f:
            import_data = json.load(f)
        
        self.memory_states = {k: self._deserialize_state(v) for k, v in import_data["memory_states"].items()}
        self.dimensional_index = defaultdict(list, import_data["dimensional_index"])
        self.coherence_matrix = np.array(import_data["coherence_matrix"])
        self.analytics = defaultdict(int, import_data["analytics"])
    
    def _serialize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize temporal state for JSON export."""
        serialized = state.copy()
        serialized["timestamp"] = state["timestamp"].isoformat()
        serialized["dimensional_weights"] = state["dimensional_weights"].tolist()
        return serialized
    
    def _deserialize_state(self, state_data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize temporal state from JSON import."""
        state = state_data.copy()
        state["timestamp"] = datetime.fromisoformat(state_data["timestamp"])
        state["dimensional_weights"] = np.array(state_data["dimensional_weights"])
        return state