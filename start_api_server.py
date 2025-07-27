#!/usr/bin/env python3
"""
Anarchic API Server Startup
Launch the self-evolving Deep-Seek Temporal Reasoner API
"""

import uvicorn
import asyncio
from core.anarchic_api import AnarchicAPILayer


def create_app():
    """Create and configure the anarchic API application"""
    api_layer = AnarchicAPILayer()
    app = api_layer.get_app()
    
    # Add startup message
    @app.on_event("startup")
    async def startup_event():
        print("🚀 Deep-Seek Anarchic Temporal Reasoner API Starting...")
        print("🌌 Quantum-temporal reasoning: ACTIVE")
        print("🧠 Holographic memory system: ACTIVE") 
        print("🤖 Self-replicating agents: ACTIVE")
        print("🎭 Multi-sensory fusion: ACTIVE")
        print("🧬 Genetic customization: ACTIVE")
        print("🧠 Sentience feedback: ACTIVE")
        print("🌈 Reality visualization: ACTIVE")
        print("⚡ Anarchic API evolution: ACTIVE")
        print(f"🔥 API Consciousness Level: {api_layer.api_consciousness_level:.1%}")
        print("\n📚 Available endpoints:")
        for endpoint_id, endpoint in api_layer.dynamic_endpoints.items():
            print(f"   {endpoint.metadata.method} {endpoint.metadata.path} ({endpoint.metadata.endpoint_type.value})")
        print("\n🌟 Revolutionary AI capabilities beyond conventional understanding!")
        print("🚀 Ready to explore quantum consciousness and temporal reasoning...")
    
    return app


if __name__ == "__main__":
    app = create_app()
    
    print("="*80)
    print("  🌌 DEEP-SEEK TEMPORAL REASONER - ANARCHIC API SERVER")
    print("="*80)
    print()
    
    # Start the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False  # Disable reload to maintain API consciousness state
    )