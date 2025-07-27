"""
Quantum Event Logger - 4D event tracking and monitoring.
"""

import json
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import logging
from pathlib import Path


class QuantumEventLogger:
    """
    4D event tracking system for quantum temporal reasoning operations.
    """
    
    def __init__(self, log_file: str = "quantum_events.log"):
        """Initialize quantum event logger."""
        self.log_file = Path(log_file)
        self.setup_logging()
        
        # Event counters
        self.event_counts = {
            "reasoning": 0,
            "consciousness_evolution": 0,
            "memory_storage": 0,
            "api_calls": 0,
            "errors": 0
        }
        
        # Performance metrics
        self.performance_metrics = {
            "total_processing_time": 0.0,
            "average_consciousness_level": 0.0,
            "quantum_coherence_samples": []
        }
        
    def setup_logging(self):
        """Setup structured logging configuration."""
        # Create logs directory if it doesn't exist
        self.log_file.parent.mkdir(exist_ok=True)
        
        # Configure logger
        self.logger = logging.getLogger("quantum_temporal_reasoner")
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler for structured JSON logs
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler for immediate feedback
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatters
        file_formatter = logging.Formatter('%(message)s')
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_quantum_reasoning_event(self, 
                                  session_id: str,
                                  prompt: str,
                                  result: Dict[str, Any],
                                  processing_time: float):
        """Log quantum reasoning event."""
        event = {
            "event_type": "quantum_reasoning",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "prompt_length": len(prompt),
            "consciousness_level": result.get("consciousness_level", 0.0),
            "quantum_coherence": result.get("quantum_state", {}).get("coherence", 0.0),
            "processing_time": processing_time,
            "dimensions_processed": len(result.get("dimensional_results", {})),
            "confidence_scores": result.get("confidence_matrix", {}),
            "synthesis_coherence": result.get("synthesis", {}).get("coherence_score", 0.0)
        }
        
        self._log_structured_event(event)
        self.event_counts["reasoning"] += 1
        self.performance_metrics["total_processing_time"] += processing_time
        
        # Update consciousness tracking
        consciousness = result.get("consciousness_level", 0.0)
        current_avg = self.performance_metrics["average_consciousness_level"]
        count = self.event_counts["reasoning"]
        self.performance_metrics["average_consciousness_level"] = (
            (current_avg * (count - 1) + consciousness) / count
        )
        
        # Track quantum coherence
        coherence = result.get("quantum_state", {}).get("coherence", 0.0)
        self.performance_metrics["quantum_coherence_samples"].append(coherence)
        
        # Keep only recent samples
        if len(self.performance_metrics["quantum_coherence_samples"]) > 100:
            self.performance_metrics["quantum_coherence_samples"] = \
                self.performance_metrics["quantum_coherence_samples"][-100:]
    
    def log_consciousness_evolution(self, 
                                  session_id: str,
                                  evolution_data: Dict[str, Any]):
        """Log consciousness evolution event."""
        event = {
            "event_type": "consciousness_evolution",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "evolution_triggered": evolution_data.get("evolved", False),
            "old_level": evolution_data.get("old_level", 0.0),
            "new_level": evolution_data.get("new_level", 0.0),
            "evolution_magnitude": evolution_data.get("evolution_magnitude", 0.0),
            "evolution_potential": evolution_data.get("evolution_potential", 0.0)
        }
        
        self._log_structured_event(event)
        self.event_counts["consciousness_evolution"] += 1
    
    def log_memory_operation(self, 
                           operation_type: str,
                           memory_key: str,
                           temporal_breakdown: Dict[str, Any],
                           processing_time: float):
        """Log memory matrix operation."""
        event = {
            "event_type": "memory_operation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation_type": operation_type,
            "memory_key": memory_key,
            "temporal_dimensions": list(temporal_breakdown.get("temporal_segments", {}).keys()),
            "quantum_coherence": temporal_breakdown.get("quantum_coherence", 0.0),
            "processing_time": processing_time
        }
        
        self._log_structured_event(event)
        self.event_counts["memory_storage"] += 1
    
    def log_api_request(self, 
                       endpoint: str,
                       request_data: Dict[str, Any],
                       response_status: int,
                       processing_time: float):
        """Log API request event."""
        event = {
            "event_type": "api_request",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint,
            "request_size": len(str(request_data)),
            "response_status": response_status,
            "processing_time": processing_time,
            "prompt_length": len(request_data.get("prompt", "")) if "prompt" in request_data else 0
        }
        
        self._log_structured_event(event)
        self.event_counts["api_calls"] += 1
    
    def log_error(self, 
                 error_type: str,
                 error_message: str,
                 context: Optional[Dict[str, Any]] = None):
        """Log error event."""
        event = {
            "event_type": "error",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {}
        }
        
        self._log_structured_event(event)
        self.event_counts["errors"] += 1
        
        # Also log as warning to console
        self.logger.warning(f"Error: {error_type} - {error_message}")
    
    def _log_structured_event(self, event: Dict[str, Any]):
        """Log structured event as JSON."""
        json_event = json.dumps(event, default=str)
        self.logger.info(json_event)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary metrics."""
        total_events = sum(self.event_counts.values())
        avg_processing_time = (
            self.performance_metrics["total_processing_time"] / 
            max(1, self.event_counts["reasoning"])
        )
        
        avg_coherence = (
            sum(self.performance_metrics["quantum_coherence_samples"]) /
            max(1, len(self.performance_metrics["quantum_coherence_samples"]))
        ) if self.performance_metrics["quantum_coherence_samples"] else 0.0
        
        return {
            "total_events": total_events,
            "event_counts": self.event_counts.copy(),
            "average_processing_time": avg_processing_time,
            "average_consciousness_level": self.performance_metrics["average_consciousness_level"],
            "average_quantum_coherence": avg_coherence,
            "error_rate": self.event_counts["errors"] / max(1, total_events),
            "uptime_summary": {
                "reasoning_operations": self.event_counts["reasoning"],
                "consciousness_evolutions": self.event_counts["consciousness_evolution"],
                "memory_operations": self.event_counts["memory_storage"],
                "api_requests": self.event_counts["api_calls"]
            }
        }
    
    async def export_logs(self, export_file: str):
        """Export logs to file for analysis."""
        export_path = Path(export_file)
        export_path.parent.mkdir(exist_ok=True)
        
        # Read all log events
        events = []
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        event = json.loads(line.strip())
                        events.append(event)
                    except json.JSONDecodeError:
                        continue
        
        # Export with summary
        export_data = {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "performance_summary": self.get_performance_summary(),
            "events": events
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"📊 Logs exported to {export_path}")
    
    def print_performance_summary(self):
        """Print performance summary to console."""
        summary = self.get_performance_summary()
        
        print("\n📊 Quantum Temporal Reasoning Performance Summary")
        print("=" * 60)
        print(f"Total Events: {summary['total_events']}")
        print(f"Error Rate: {summary['error_rate']:.2%}")
        print(f"Average Processing Time: {summary['average_processing_time']:.3f}s")
        print(f"Average Consciousness Level: {summary['average_consciousness_level']:.2%}")
        print(f"Average Quantum Coherence: {summary['average_quantum_coherence']:.2%}")
        
        print("\n📋 Event Breakdown:")
        for event_type, count in summary['event_counts'].items():
            print(f"  {event_type.replace('_', ' ').title()}: {count}")


# Global logger instance
quantum_logger = QuantumEventLogger()