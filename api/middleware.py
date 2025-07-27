"""
Middleware - Quantum request processing middleware.
"""

import asyncio
from typing import Dict, Any, Optional
import time
import logging
from datetime import datetime, timezone


class Middleware:
    """
    Quantum request processing middleware for enhanced request handling.
    """
    
    def __init__(self):
        """Initialize middleware."""
        self.request_count = 0
        self.processing_times = []
        self.error_count = 0
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming request with quantum enhancements.
        
        Args:
            request_data: Raw request data
            
        Returns:
            Processed request data
        """
        start_time = time.time()
        
        try:
            # Increment request counter
            self.request_count += 1
            
            # Log request
            self.logger.info(f"Processing request #{self.request_count}")
            
            # Validate and enhance request
            processed_request = await self._validate_request(request_data)
            processed_request = await self._enhance_request(processed_request)
            
            # Track processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Limit processing time history
            if len(self.processing_times) > 1000:
                self.processing_times = self.processing_times[-1000:]
            
            return processed_request
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Request processing error: {str(e)}")
            raise
    
    async def _validate_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate request data."""
        
        # Ensure required fields
        if "prompt" not in request_data or not request_data["prompt"].strip():
            raise ValueError("Prompt is required and cannot be empty")
        
        # Validate focus_dimensions if provided
        if "focus_dimensions" in request_data and request_data["focus_dimensions"]:
            valid_dimensions = ["past", "present", "future"]
            for dim in request_data["focus_dimensions"]:
                if dim not in valid_dimensions:
                    raise ValueError(f"Invalid dimension: {dim}. Must be one of {valid_dimensions}")
        
        # Validate consciousness_level if provided
        if "consciousness_level" in request_data and request_data["consciousness_level"] is not None:
            level = request_data["consciousness_level"]
            if not 0.0 <= level <= 1.0:
                raise ValueError("Consciousness level must be between 0.0 and 1.0")
        
        return request_data
    
    async def _enhance_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance request with quantum processing."""
        
        enhanced_request = request_data.copy()
        
        # Add timestamp
        enhanced_request["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Add processing metadata
        enhanced_request["processing_metadata"] = {
            "request_id": f"req_{self.request_count}",
            "middleware_version": "1.0.0",
            "quantum_enhanced": True
        }
        
        # Enhance prompt if it's too short
        if len(enhanced_request["prompt"].split()) < 3:
            enhanced_request["prompt"] = f"Please analyze and reason about: {enhanced_request['prompt']}"
        
        # Set default values
        if "focus_dimensions" not in enhanced_request:
            enhanced_request["focus_dimensions"] = None
        
        if "self_reflect" not in enhanced_request:
            enhanced_request["self_reflect"] = True
        
        return enhanced_request
    
    def get_middleware_stats(self) -> Dict[str, Any]:
        """Get middleware performance statistics."""
        
        avg_processing_time = (
            sum(self.processing_times) / len(self.processing_times)
            if self.processing_times else 0.0
        )
        
        return {
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "success_rate": (self.request_count - self.error_count) / max(1, self.request_count),
            "average_processing_time": avg_processing_time,
            "recent_processing_times": self.processing_times[-10:] if self.processing_times else []
        }