"""
Temporal Streaming - Real-time consciousness streaming capabilities.
"""

import asyncio
import json
from typing import Dict, Any, AsyncGenerator, Optional
from datetime import datetime, timezone
import time


class TemporalStreaming:
    """
    Real-time consciousness streaming for continuous temporal reasoning.
    """
    
    def __init__(self):
        """Initialize temporal streaming system."""
        self.active_streams = {}
        self.stream_count = 0
        
    async def create_consciousness_stream(self, 
                                        session_id: str,
                                        reasoner,
                                        stream_config: Optional[Dict[str, Any]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Create real-time consciousness streaming.
        
        Args:
            session_id: Session identifier
            reasoner: Quantum temporal reasoner instance
            stream_config: Streaming configuration
            
        Yields:
            Real-time consciousness and quantum state updates
        """
        self.stream_count += 1
        stream_id = f"stream_{self.stream_count}"
        
        config = stream_config or {}
        update_interval = config.get("update_interval", 2.0)  # seconds
        include_quantum_state = config.get("include_quantum_state", True)
        include_memory_insights = config.get("include_memory_insights", True)
        
        try:
            self.active_streams[stream_id] = {
                "session_id": session_id,
                "start_time": datetime.now(timezone.utc),
                "config": config,
                "update_count": 0
            }
            
            while stream_id in self.active_streams:
                # Collect current state
                stream_data = await self._collect_stream_data(
                    reasoner, 
                    include_quantum_state, 
                    include_memory_insights
                )
                
                # Add stream metadata
                stream_data.update({
                    "stream_id": stream_id,
                    "session_id": session_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "update_count": self.active_streams[stream_id]["update_count"]
                })
                
                self.active_streams[stream_id]["update_count"] += 1
                
                yield stream_data
                
                # Wait for next update
                await asyncio.sleep(update_interval)
                
        except Exception as e:
            yield {
                "error": f"Stream error: {str(e)}",
                "stream_id": stream_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        finally:
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
    
    async def _collect_stream_data(self, 
                                 reasoner,
                                 include_quantum_state: bool,
                                 include_memory_insights: bool) -> Dict[str, Any]:
        """Collect current stream data from reasoner."""
        
        stream_data = {
            "consciousness_level": reasoner.consciousness_level,
            "processing_metrics": {
                "queries_processed": reasoner.reasoning_metrics["queries_processed"],
                "average_confidence": reasoner.reasoning_metrics["average_confidence"]
            }
        }
        
        if include_quantum_state:
            stream_data["quantum_state"] = reasoner.quantum_state.copy()
        
        if include_memory_insights:
            try:
                memory_patterns = await reasoner.memory_matrix.analyze_patterns()
                stream_data["memory_insights"] = {
                    "total_states": memory_patterns.get("total_states", 0),
                    "average_consciousness": memory_patterns.get("average_consciousness", 0.0),
                    "temporal_distribution": memory_patterns.get("temporal_distribution", {})
                }
            except Exception:
                stream_data["memory_insights"] = {"error": "Unable to collect memory insights"}
        
        return stream_data
    
    async def create_temporal_event_stream(self, 
                                         reasoner,
                                         event_filter: Optional[Dict[str, Any]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Create stream for specific temporal events.
        
        Args:
            reasoner: Quantum temporal reasoner instance
            event_filter: Filter criteria for events
            
        Yields:
            Filtered temporal events as they occur
        """
        filter_config = event_filter or {}
        consciousness_threshold = filter_config.get("consciousness_threshold", 0.0)
        evolution_events_only = filter_config.get("evolution_events_only", False)
        
        last_consciousness_level = reasoner.consciousness_level
        last_evolution_count = len(reasoner.consciousness_engine.evolution_events)
        
        while True:
            current_consciousness = reasoner.consciousness_level
            current_evolution_count = len(reasoner.consciousness_engine.evolution_events)
            
            # Check for consciousness evolution
            if current_evolution_count > last_evolution_count:
                yield {
                    "event_type": "consciousness_evolution",
                    "old_level": last_consciousness_level,
                    "new_level": current_consciousness,
                    "evolution_event": reasoner.consciousness_engine.evolution_events[-1],
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                last_evolution_count = current_evolution_count
            
            # Check for significant consciousness changes
            consciousness_change = abs(current_consciousness - last_consciousness_level)
            if consciousness_change > consciousness_threshold and not evolution_events_only:
                yield {
                    "event_type": "consciousness_change",
                    "change_magnitude": consciousness_change,
                    "old_level": last_consciousness_level,
                    "new_level": current_consciousness,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            
            last_consciousness_level = current_consciousness
            
            await asyncio.sleep(1.0)  # Check every second
    
    def stop_stream(self, stream_id: str) -> bool:
        """Stop a specific stream."""
        if stream_id in self.active_streams:
            del self.active_streams[stream_id]
            return True
        return False
    
    def get_active_streams(self) -> Dict[str, Any]:
        """Get information about active streams."""
        stream_info = {}
        
        for stream_id, stream_data in self.active_streams.items():
            stream_info[stream_id] = {
                "session_id": stream_data["session_id"],
                "start_time": stream_data["start_time"].isoformat(),
                "update_count": stream_data["update_count"],
                "config": stream_data["config"]
            }
        
        return {
            "active_streams": stream_info,
            "total_streams": len(self.active_streams)
        }