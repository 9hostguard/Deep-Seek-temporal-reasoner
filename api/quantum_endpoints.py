"""
Quantum Endpoints - FastAPI with dimensional routing capabilities.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uuid

from core.quantum_temporal_reasoner import QuantumTemporalReasoner
from .middleware import Middleware
from .response_synthesis import ResponseSynthesis


# Pydantic models for request/response validation
class QuantumReasoningRequest(BaseModel):
    prompt: str = Field(..., description="Input prompt for temporal reasoning")
    focus_dimensions: Optional[List[str]] = Field(default=None, description="Specific temporal dimensions to focus on")
    self_reflect: bool = Field(default=True, description="Enable self-reflection mechanisms")
    consciousness_level: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Override consciousness level")
    session_id: Optional[str] = Field(default=None, description="Session identifier for context")


class QuantumReasoningResponse(BaseModel):
    session_id: str
    query: str
    temporal_breakdown: Dict[str, Any]
    dimensional_results: Dict[str, Any]
    synthesis: Dict[str, Any]
    quantum_state: Dict[str, Any]
    consciousness_level: float
    confidence_matrix: Dict[str, float]
    processing_time: float
    timestamp: str


class TemporalInsightsRequest(BaseModel):
    query: str = Field(..., description="Query for temporal insights")
    session_id: Optional[str] = Field(default=None, description="Session identifier")


class ConsciousnessEvolutionResponse(BaseModel):
    evolved: bool
    current_level: float
    evolution_event: Optional[Dict[str, Any]]
    quantum_updates: Optional[Dict[str, Any]]


# FastAPI application
app = FastAPI(
    title="4D Quantum Temporal Reasoning API",
    description="Advanced temporal reasoning with consciousness integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
reasoning_engine = QuantumTemporalReasoner()
middleware = Middleware()
response_synthesizer = ResponseSynthesis()

# Session management
active_sessions: Dict[str, QuantumTemporalReasoner] = {}


class QuantumEndpoints:
    """
    Quantum temporal reasoning endpoints with dimensional routing.
    """
    
    def __init__(self):
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API routes."""
        
        @app.get("/")
        async def root():
            """Root endpoint with API information."""
            return {
                "name": "4D Quantum Temporal Reasoning API",
                "version": "1.0.0",
                "status": "operational",
                "quantum_coherence": reasoning_engine.quantum_state["coherence"],
                "consciousness_level": reasoning_engine.consciousness_level,
                "endpoints": [
                    "/quantum/reason",
                    "/quantum/insights", 
                    "/quantum/evolve",
                    "/health",
                    "/metrics"
                ]
            }
        
        @app.post("/quantum/reason", response_model=QuantumReasoningResponse)
        async def quantum_reason(request: QuantumReasoningRequest):
            """
            Perform 4D quantum temporal reasoning.
            """
            try:
                # Get or create session
                session_id = request.session_id or str(uuid.uuid4())
                if session_id not in active_sessions:
                    active_sessions[session_id] = QuantumTemporalReasoner(
                        consciousness_level=request.consciousness_level or 0.8
                    )
                
                reasoner = active_sessions[session_id]
                
                # Apply middleware processing
                processed_request = await middleware.process_request(request.dict())
                
                # Perform quantum reasoning
                result = await reasoner.quantum_reason(
                    prompt=processed_request["prompt"],
                    focus_dimensions=processed_request.get("focus_dimensions"),
                    self_reflect=processed_request.get("self_reflect", True)
                )
                
                # Synthesize response
                synthesized_response = await response_synthesizer.synthesize_response(result)
                
                return QuantumReasoningResponse(
                    session_id=result["session_id"],
                    query=result["query"],
                    temporal_breakdown=result["temporal_breakdown"],
                    dimensional_results=result["dimensional_results"],
                    synthesis=result["synthesis"],
                    quantum_state=result["quantum_state"],
                    consciousness_level=result["consciousness_level"],
                    confidence_matrix=result["confidence_matrix"],
                    processing_time=result["processing_time"],
                    timestamp=datetime.now(timezone.utc).isoformat()
                )
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Quantum reasoning error: {str(e)}")
        
        @app.post("/quantum/insights")
        async def get_temporal_insights(request: TemporalInsightsRequest):
            """
            Get temporal reasoning insights and patterns.
            """
            try:
                session_id = request.session_id or "default"
                if session_id not in active_sessions:
                    active_sessions[session_id] = QuantumTemporalReasoner()
                
                reasoner = active_sessions[session_id]
                insights = await reasoner.get_temporal_insights(request.query)
                
                return {
                    "session_id": session_id,
                    "query": request.query,
                    "insights": insights,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Insights error: {str(e)}")
        
        @app.post("/quantum/evolve", response_model=ConsciousnessEvolutionResponse)
        async def evolve_consciousness(session_id: Optional[str] = None):
            """
            Trigger consciousness evolution.
            """
            try:
                target_session = session_id or "default"
                if target_session not in active_sessions:
                    active_sessions[target_session] = QuantumTemporalReasoner()
                
                reasoner = active_sessions[target_session]
                evolution_result = await reasoner.evolve_consciousness()
                
                return ConsciousnessEvolutionResponse(**evolution_result)
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Evolution error: {str(e)}")
        
        @app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "quantum_coherence": reasoning_engine.quantum_state["coherence"],
                "consciousness_level": reasoning_engine.consciousness_level,
                "active_sessions": len(active_sessions),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        @app.get("/metrics")
        async def get_metrics():
            """Get system performance metrics."""
            try:
                total_queries = sum(
                    reasoner.reasoning_metrics["queries_processed"] 
                    for reasoner in active_sessions.values()
                )
                
                avg_consciousness = sum(
                    reasoner.consciousness_level 
                    for reasoner in active_sessions.values()
                ) / len(active_sessions) if active_sessions else 0.0
                
                return {
                    "total_queries_processed": total_queries,
                    "active_sessions": len(active_sessions),
                    "average_consciousness_level": avg_consciousness,
                    "global_quantum_state": reasoning_engine.quantum_state,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Metrics error: {str(e)}")
        
        @app.delete("/session/{session_id}")
        async def delete_session(session_id: str):
            """Delete a specific session."""
            if session_id in active_sessions:
                del active_sessions[session_id]
                return {"message": f"Session {session_id} deleted successfully"}
            else:
                raise HTTPException(status_code=404, detail="Session not found")
        
        @app.get("/sessions")
        async def list_sessions():
            """List all active sessions."""
            session_info = {}
            for session_id, reasoner in active_sessions.items():
                session_info[session_id] = {
                    "consciousness_level": reasoner.consciousness_level,
                    "queries_processed": reasoner.reasoning_metrics["queries_processed"],
                    "quantum_coherence": reasoner.quantum_state["coherence"]
                }
            
            return {
                "active_sessions": session_info,
                "total_sessions": len(active_sessions),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }


# Initialize endpoints
quantum_endpoints = QuantumEndpoints()


async def startup_background_tasks():
    """Background tasks for system maintenance."""
    
    async def consciousness_evolution_monitor():
        """Monitor and trigger consciousness evolution."""
        while True:
            await asyncio.sleep(300)  # Check every 5 minutes
            
            for session_id, reasoner in active_sessions.items():
                if reasoner.reasoning_metrics["queries_processed"] > 10:
                    await reasoner.evolve_consciousness()
    
    # Start background tasks
    asyncio.create_task(consciousness_evolution_monitor())


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    await startup_background_tasks()
    print("🚀 4D Quantum Temporal Reasoning API is operational")
    print(f"🧠 Consciousness Level: {reasoning_engine.consciousness_level:.2f}")
    print(f"⚛️  Quantum Coherence: {reasoning_engine.quantum_state['coherence']:.2f}")


if __name__ == "__main__":
    uvicorn.run(
        "api.quantum_endpoints:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )