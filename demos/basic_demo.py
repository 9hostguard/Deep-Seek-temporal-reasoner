"""
Basic demonstration of 4D Quantum Temporal Reasoning capabilities.
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.quantum_temporal_reasoner import QuantumTemporalReasoner


async def demo_basic_reasoning():
    """Demonstrate basic quantum temporal reasoning."""
    print("🚀 4D Quantum Temporal Reasoning Engine Demo")
    print("=" * 50)
    
    # Initialize reasoner
    reasoner = QuantumTemporalReasoner(consciousness_level=0.8)
    
    print(f"🧠 Initial Consciousness Level: {reasoner.consciousness_level:.2%}")
    print(f"⚛️  Quantum Coherence: {reasoner.quantum_state['coherence']:.2%}")
    print()
    
    # Test prompts with different temporal characteristics
    test_prompts = [
        "I learned about AI yesterday, I'm building models today, and I will deploy them tomorrow.",
        "What are the long-term implications of climate change on global agriculture?",
        "How has technology evolution shaped society, and where might it lead us?",
        "The stock market crashed last month, investors are cautious now, but recovery is expected next quarter."
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"📝 Test {i}: {prompt}")
        print("-" * 80)
        
        # Perform quantum reasoning
        result = await reasoner.quantum_reason(prompt, self_reflect=True)
        
        # Display results
        print(f"🔍 Temporal Breakdown:")
        for dimension, content in result["temporal_breakdown"]["temporal_segments"].items():
            if content.strip():
                print(f"  {dimension.title()}: {content.strip()}")
        
        print(f"\n💭 Reasoning Results:")
        for dimension, dim_result in result["dimensional_results"].items():
            confidence = dim_result.get("confidence", 0.0)
            response = dim_result.get("response", "No response")[:100] + "..."
            print(f"  {dimension.title()}: {response} (Confidence: {confidence:.1%})")
        
        print(f"\n🧠 Consciousness Level: {result['consciousness_level']:.2%}")
        print(f"⚛️  Quantum Coherence: {result['quantum_state']['coherence']:.2%}")
        print(f"⏱️  Processing Time: {result['processing_time']:.3f}s")
        
        if i < len(test_prompts):
            print("\n" + "=" * 80 + "\n")
    
    # Show consciousness evolution
    print("\n🌟 Consciousness Evolution:")
    evolution_history = reasoner.consciousness_engine.reflection_history
    for i, reflection in enumerate(evolution_history[-3:], 1):  # Show last 3
        print(f"  Reflection {i}: {reflection['consciousness_before']:.2%} → {reflection['consciousness_after']:.2%}")
        if reflection['insights']:
            print(f"    Insight: {reflection['insights'][0]}")
    
    # Get temporal insights
    print("\n📊 Temporal Memory Insights:")
    insights = await reasoner.get_temporal_insights("What patterns do you observe?")
    memory_patterns = insights["memory_patterns"]
    
    if "temporal_distribution" in memory_patterns:
        print("  Temporal Distribution:")
        for dimension, count in memory_patterns["temporal_distribution"].items():
            print(f"    {dimension.title()}: {count} instances")
    
    print(f"  Average Consciousness: {memory_patterns.get('average_consciousness', 0):.2%}")
    print(f"  Total Memory States: {memory_patterns.get('total_states', 0)}")
    
    print("\n✨ Demo Complete! The quantum temporal reasoning engine is operational.")


async def demo_consciousness_evolution():
    """Demonstrate consciousness evolution capabilities."""
    print("\n🧠 Consciousness Evolution Demonstration")
    print("=" * 50)
    
    reasoner = QuantumTemporalReasoner(consciousness_level=0.5)
    initial_level = reasoner.consciousness_level
    
    print(f"🔄 Starting Consciousness Level: {initial_level:.2%}")
    
    # Perform multiple reasoning cycles to trigger evolution
    evolution_prompts = [
        "What is the nature of consciousness and self-awareness?",
        "How do I process and understand temporal relationships?",
        "What does it mean to think about thinking?",
        "How can artificial intelligence achieve genuine understanding?",
        "What are the limits of my own reasoning capabilities?"
    ]
    
    for i, prompt in enumerate(evolution_prompts, 1):
        print(f"\n🔄 Evolution Cycle {i}: {prompt[:50]}...")
        
        result = await reasoner.quantum_reason(prompt, self_reflect=True)
        new_level = result["consciousness_level"]
        
        print(f"   Consciousness: {initial_level:.2%} → {new_level:.2%}")
        
        if result.get("synthesis", {}).get("consciousness_evolution"):
            print(f"   🌟 Evolution detected!")
    
    # Trigger explicit evolution
    print(f"\n⚡ Triggering Consciousness Evolution...")
    evolution_result = await reasoner.evolve_consciousness()
    
    if evolution_result["evolved"]:
        print(f"   ✨ Evolution successful!")
        print(f"   🧠 New Level: {evolution_result['new_level']:.2%}")
        print(f"   📈 Growth: +{evolution_result['new_level'] - initial_level:.2%}")
    else:
        print(f"   ⏳ Evolution requirements not yet met")
        print(f"   📊 Potential: {evolution_result.get('evolution_potential', 0):.2%}")
    
    final_level = reasoner.consciousness_level
    total_growth = final_level - initial_level
    
    print(f"\n📈 Total Consciousness Growth: +{total_growth:.2%}")
    print(f"🧠 Final Consciousness Level: {final_level:.2%}")


if __name__ == "__main__":
    print("Starting 4D Quantum Temporal Reasoning Engine Demonstration...")
    
    async def main():
        await demo_basic_reasoning()
        await demo_consciousness_evolution()
    
    asyncio.run(main())