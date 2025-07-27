"""
Temporal decomposition module for breaking down prompts across time dimensions.
"""

import re
from typing import Dict, Any
import asyncio


def decompose(prompt: str) -> Dict[str, str]:
    """
    Decompose a prompt into temporal segments: past, present, and future.
    
    Args:
        prompt: Input prompt to decompose
        
    Returns:
        Dictionary with 'past', 'present', 'future' keys
    """
    # Simple temporal decomposition based on temporal keywords and patterns
    temporal_segments = {
        "past": "",
        "present": "",
        "future": ""
    }
    
    # Keywords that indicate temporal context
    past_indicators = ['was', 'were', 'had', 'did', 'before', 'previously', 'earlier', 'ago', 'yesterday', 'last', 'learned']
    future_indicators = ['will', 'shall', 'going to', 'tomorrow', 'next', 'later', 'soon', 'predict', 'forecast', 'deploy']
    present_indicators = ['is', 'are', 'am', 'being', 'now', 'currently', 'today', 'present', 'coding']
    
    # Split by common conjunctions and sentence boundaries
    segments = re.split(r'[,.!?;]|(?:\s+and\s+)|(?:\s+but\s+)|(?:\s+or\s+)', prompt)
    
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
            
        segment_lower = segment.lower()
        
        # Check for temporal indicators
        past_score = sum(1 for word in past_indicators if word in segment_lower)
        future_score = sum(1 for word in future_indicators if word in segment_lower)
        present_score = sum(1 for word in present_indicators if word in segment_lower)
        
        # Assign to highest scoring temporal dimension
        if past_score > future_score and past_score > present_score:
            temporal_segments["past"] += segment + ". "
        elif future_score > present_score and future_score > past_score:
            temporal_segments["future"] += segment + ". "
        else:
            temporal_segments["present"] += segment + ". "
    
    # If no clear temporal assignment, assign to present
    if not any(temporal_segments.values()):
        temporal_segments["present"] = prompt
        
    return temporal_segments


async def async_decompose(prompt: str) -> Dict[str, str]:
    """Async version of decompose for use in async contexts."""
    return decompose(prompt)


def quantum_decompose(prompt: str, dimensions: int = 4) -> Dict[str, Any]:
    """
    Advanced decomposition across multiple temporal dimensions.
    
    Args:
        prompt: Input prompt
        dimensions: Number of temporal dimensions (default 4)
        
    Returns:
        Multi-dimensional temporal breakdown
    """
    base_decomposition = decompose(prompt)
    
    quantum_result = {
        "temporal_segments": base_decomposition,
        "confidence_matrix": {
            "past": 0.8 if base_decomposition["past"] else 0.1,
            "present": 0.9 if base_decomposition["present"] else 0.1,
            "future": 0.7 if base_decomposition["future"] else 0.1
        },
        "quantum_coherence": 0.85,
        "dimensional_depth": dimensions
    }
    
    return quantum_result