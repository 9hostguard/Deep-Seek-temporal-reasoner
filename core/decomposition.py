"""Temporal decomposition module for breaking down prompts into temporal components."""

import re
from typing import Dict


def decompose(prompt: str) -> Dict[str, str]:
    """
    Decompose a prompt into past, present, and future temporal components.

    Args:
        prompt (str): The input prompt to decompose

    Returns:
        Dict[str, str]: Dictionary with 'past', 'present', and 'future' keys
    """
    # Simple temporal keyword detection for basic functionality
    past_keywords = [
        "was",
        "were",
        "had",
        "did",
        "ago",
        "before",
        "yesterday",
        "past",
        "previous",
        "earlier",
    ]
    present_keywords = [
        "is",
        "are",
        "am",
        "now",
        "today",
        "current",
        "present",
        "currently",
    ]
    future_keywords = [
        "will",
        "shall",
        "going to",
        "tomorrow",
        "future",
        "next",
        "later",
        "soon",
    ]

    # Initialize result dictionary
    result = {"past": "", "present": "", "future": ""}

    # Split prompt into sentences for analysis
    sentences = re.split(r"[.!?]+", prompt)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_lower = sentence.lower()

        # Determine temporal orientation based on keywords
        has_past = any(keyword in sentence_lower for keyword in past_keywords)
        has_present = any(keyword in sentence_lower for keyword in present_keywords)
        has_future = any(keyword in sentence_lower for keyword in future_keywords)

        # Assign to the most likely temporal category
        if has_past and not has_present and not has_future:
            result["past"] += sentence + ". "
        elif has_future and not has_past and not has_present:
            result["future"] += sentence + ". "
        elif has_present or (not has_past and not has_future):
            # Default to present if no clear temporal indicators
            result["present"] += sentence + ". "
        else:
            # If multiple temporal indicators, add to present as default
            result["present"] += sentence + ". "

    # Clean up trailing spaces and periods
    for key in result:
        result[key] = result[key].strip()

    # If no specific temporal content found, put everything in present
    if not any(result.values()):
        result["present"] = prompt

    return result
