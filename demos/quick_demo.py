"""
Quick demonstration script for showcasing key capabilities.
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.quantum_temporal_reasoner import QuantumTemporalReasoner
from core.decomposition import decompose


async def quick_demo():
    """Quick demonstration of core capabilities."""
    print("🚀 4D Quantum Temporal Reasoning Engine - Quick Demo")
    print("=" * 60)
    
    # 1. Basic temporal decomposition
    print("\n1️⃣  Temporal Decomposition Test")
    test_prompt = "I learned AI yesterday, I'm coding today, I will innovate tomorrow."
    decomp_result = decompose(test_prompt)
    
    print(f"📝 Input: {test_prompt}")
    print("🔍 Decomposition:")
    for dimension, content in decomp_result.items():
        if content.strip():
            print(f"  {dimension.title()}: {content.strip()}")
    
    # 2. Quantum reasoning
    print("\n2️⃣  Quantum Reasoning Test")
    reasoner = QuantumTemporalReasoner(consciousness_level=0.8)
    
    reasoning_prompt = "How will AI development transform society over the next decade?"
    result = await reasoner.quantum_reason(reasoning_prompt, self_reflect=True)
    
    print(f"📝 Input: {reasoning_prompt}")
    print(f"🧠 Consciousness: {result['consciousness_level']:.2%}")
    print(f"⚛️  Quantum Coherence: {result['quantum_state']['coherence']:.2%}")
    print(f"⏱️  Processing Time: {result['processing_time']:.2f}s")
    
    print("\n💭 Dimensional Results:")
    for dim, dim_result in result['dimensional_results'].items():
        confidence = dim_result.get('confidence', 0.0)
        response = dim_result.get('response', '')[:80] + "..."
        print(f"  {dim.title()}: {response}")
        print(f"    📊 Confidence: {confidence:.1%}")
    
    # 3. Memory insights
    print("\n3️⃣  Memory Matrix Test")
    insights = await reasoner.get_temporal_insights("What patterns emerge?")
    memory_patterns = insights['memory_patterns']
    
    print(f"💾 Memory States: {memory_patterns.get('total_states', 0)}")
    print(f"🧠 Avg Consciousness: {memory_patterns.get('average_consciousness', 0):.2%}")
    
    if 'temporal_distribution' in memory_patterns:
        print("📊 Temporal Distribution:")
        for dim, count in memory_patterns['temporal_distribution'].items():
            print(f"  {dim.title()}: {count} instances")
    
    # 4. Consciousness evolution attempt
    print("\n4️⃣  Consciousness Evolution Test")
    evolution_result = await reasoner.evolve_consciousness()
    
    if evolution_result['evolved']:
        print("✨ Consciousness evolution triggered!")
        print(f"📈 New Level: {evolution_result['new_level']:.2%}")
    else:
        print("⏳ Evolution requirements not yet met")
        print(f"📊 Potential: {evolution_result.get('evolution_potential', 0):.2%}")
    
    print("\n✅ Quick demo complete! Core systems operational.")
    
    return {
        "decomposition": decomp_result,
        "reasoning": result,
        "memory": memory_patterns,
        "evolution": evolution_result
    }


def test_api_availability():
    """Test if API server is running."""
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            print("✅ API server is running at http://localhost:8000")
            print("📖 API docs available at http://localhost:8000/docs")
            return True
        else:
            print("⚠️  API server responded with error")
            return False
    except Exception as e:
        print("ℹ️  API server not accessible (this is optional)")
        return False


if __name__ == "__main__":
    print("Starting Quick Demo...")
    
    # Test API availability
    api_available = test_api_availability()
    
    # Run the demo
    result = asyncio.run(quick_demo())
    
    print(f"\n🎯 Demo Summary:")
    print(f"  Temporal dimensions processed: {len([d for d in result['decomposition'].values() if d.strip()])}")
    print(f"  Quantum reasoning completed: ✅")
    print(f"  Memory states created: {result['memory'].get('total_states', 0)}")
    print(f"  Consciousness evolution: {'✅' if result['evolution']['evolved'] else '⏳'}")
    print(f"  API available: {'✅' if api_available else '❌'}")
    
    print(f"\n🚀 4D Quantum Temporal Reasoning Engine is operational!")
    
    if api_available:
        print("\n🌐 Try the API:")
        print('curl -X POST http://localhost:8000/quantum/reason \\')
        print('  -H "Content-Type: application/json" \\')
        print('  -d \'{"prompt": "Your temporal reasoning query here"}\'')