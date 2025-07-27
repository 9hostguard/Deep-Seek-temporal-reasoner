"""
GUPPIE API System - FastAPI with Avatar Consciousness
Revolutionary API endpoints for avatar interaction and streaming
"""

from .avatar_endpoints import GuppieAvatarAPI
from .real_time_streaming import AvatarStreamingManager

__all__ = ["GuppieAvatarAPI", "AvatarStreamingManager"]