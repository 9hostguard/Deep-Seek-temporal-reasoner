"""
Temporal decomposition module for breaking down prompts into temporal segments
"""

def decompose(prompt):
    """
    Decompose a prompt into temporal segments (past, present, future)
    
    Args:
        prompt (str): The input prompt to decompose
        
    Returns:
        dict: Dictionary with temporal segments as keys and focused prompts as values
    """
    # Basic temporal decomposition logic
    return {
        "past": f"{prompt} (focused on the past)",
        "present": f"{prompt} (focused on the present)",
        "future": f"{prompt} (focused on the future)",
    }