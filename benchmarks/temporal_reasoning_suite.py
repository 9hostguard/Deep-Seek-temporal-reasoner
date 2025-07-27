"""
Benchmarking suite for temporal reasoning performance and capabilities.
"""

import asyncio
import time
import statistics
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List
import numpy as np
from datetime import datetime, timezone

from core.quantum_temporal_reasoner import QuantumTemporalReasoner
from core.decomposition import decompose, quantum_decompose


class TemporalReasoningSuite:
    """
    Comprehensive benchmark suite for temporal reasoning validation.
    """
    
    def __init__(self):
        """Initialize benchmark suite."""
        self.results = {}
        self.test_prompts = {
            "simple": [
                "I learned yesterday, I work today, I will succeed tomorrow.",
                "The past was difficult, the present is challenging, the future looks bright.",
                "Markets fell last week, recovery started today, growth expected next month."
            ],
            "complex": [
                "Throughout history, technological revolutions have fundamentally transformed human societies, and today we witness the emergence of artificial intelligence as the next transformative force, which will likely reshape every aspect of human civilization in the coming decades.",
                "The financial crisis of 2008 demonstrated systemic vulnerabilities in global markets, current regulatory frameworks attempt to address these issues, while emerging technologies like blockchain and AI are creating new paradigms that may fundamentally alter financial systems by 2030.",
                "Climate patterns observed over the past century indicate accelerating changes, present-day extreme weather events reflect these trends, and future projections suggest transformative impacts on agriculture, infrastructure, and human migration patterns."
            ],
            "temporal_heavy": [
                "Before the internet revolution of the 1990s, information access was limited, while today's digital connectivity enables instant global communication, and tomorrow's quantum internet will transcend current technological boundaries entirely.",
                "Economic cycles historically follow predictable patterns of boom and bust, the current period shows signs of structural shifts driven by automation and globalization, and future economic models will need to account for unprecedented technological disruption and environmental constraints.",
                "Previous pandemics like the 1918 flu shaped public health responses, the COVID-19 pandemic revealed both strengths and weaknesses in modern health systems, and future pandemic preparedness will integrate AI-driven early warning systems and personalized medicine approaches."
            ]
        }
    
    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        print("🚀 Running Comprehensive Temporal Reasoning Benchmark")
        print("=" * 60)
        
        # Initialize reasoner
        reasoner = QuantumTemporalReasoner(consciousness_level=0.8)
        
        # Run all benchmark categories
        results = {}
        
        results["decomposition_benchmark"] = await self._benchmark_decomposition()
        results["reasoning_performance"] = await self._benchmark_reasoning_performance(reasoner)
        results["consciousness_evolution"] = await self._benchmark_consciousness_evolution(reasoner)
        results["memory_efficiency"] = await self._benchmark_memory_efficiency(reasoner)
        results["concurrent_processing"] = await self._benchmark_concurrent_processing()
        results["accuracy_validation"] = await self._benchmark_accuracy_validation(reasoner)
        
        # Calculate overall scores
        results["overall_performance"] = self._calculate_overall_performance(results)
        
        return results
    
    async def _benchmark_decomposition(self) -> Dict[str, Any]:
        """Benchmark temporal decomposition accuracy and speed."""
        print("\n📋 Benchmarking Temporal Decomposition...")
        
        decomp_times = []
        accuracy_scores = []
        
        for category, prompts in self.test_prompts.items():
            for prompt in prompts:
                # Time decomposition
                start_time = time.time()
                result = decompose(prompt)
                decomp_time = time.time() - start_time
                decomp_times.append(decomp_time)
                
                # Assess accuracy (simple heuristic)
                has_past = bool(result["past"].strip())
                has_present = bool(result["present"].strip())
                has_future = bool(result["future"].strip())
                
                # Check for temporal indicators in prompt
                prompt_lower = prompt.lower()
                expects_past = any(word in prompt_lower for word in ["yesterday", "before", "was", "had", "past", "previously", "earlier"])
                expects_future = any(word in prompt_lower for word in ["tomorrow", "will", "future", "next", "later", "coming"])
                expects_present = any(word in prompt_lower for word in ["today", "now", "current", "present", "is", "are"])
                
                accuracy = 0
                if expects_past and has_past:
                    accuracy += 0.33
                if expects_future and has_future:
                    accuracy += 0.33
                if expects_present and has_present:
                    accuracy += 0.33
                
                # Bonus for completeness
                if has_past and has_present and has_future:
                    accuracy += 0.1
                
                accuracy_scores.append(min(1.0, accuracy))
        
        avg_time = statistics.mean(decomp_times)
        avg_accuracy = statistics.mean(accuracy_scores)
        
        return {
            "average_decomposition_time": avg_time,
            "decomposition_accuracy": avg_accuracy,
            "total_tests": len(decomp_times),
            "times_distribution": {
                "min": min(decomp_times),
                "max": max(decomp_times),
                "std": statistics.stdev(decomp_times) if len(decomp_times) > 1 else 0
            }
        }
    
    async def _benchmark_reasoning_performance(self, reasoner: QuantumTemporalReasoner) -> Dict[str, Any]:
        """Benchmark quantum reasoning performance."""
        print("🧠 Benchmarking Quantum Reasoning Performance...")
        
        processing_times = []
        confidence_scores = []
        consciousness_levels = []
        coherence_scores = []
        
        for category, prompts in self.test_prompts.items():
            print(f"  Testing {category} prompts...")
            
            for prompt in prompts:
                result = await reasoner.quantum_reason(prompt, self_reflect=True)
                
                processing_times.append(result["processing_time"])
                consciousness_levels.append(result["consciousness_level"])
                
                # Collect confidence scores
                if result["confidence_matrix"]:
                    avg_confidence = statistics.mean(result["confidence_matrix"].values())
                    confidence_scores.append(avg_confidence)
                
                # Collect coherence scores
                if "synthesis" in result and "coherence_score" in result["synthesis"]:
                    coherence_scores.append(result["synthesis"]["coherence_score"])
        
        return {
            "average_processing_time": statistics.mean(processing_times),
            "average_confidence": statistics.mean(confidence_scores) if confidence_scores else 0.0,
            "average_consciousness": statistics.mean(consciousness_levels),
            "average_coherence": statistics.mean(coherence_scores) if coherence_scores else 0.0,
            "performance_distribution": {
                "processing_time": {
                    "min": min(processing_times),
                    "max": max(processing_times),
                    "std": statistics.stdev(processing_times) if len(processing_times) > 1 else 0
                },
                "confidence": {
                    "min": min(confidence_scores) if confidence_scores else 0,
                    "max": max(confidence_scores) if confidence_scores else 0,
                    "std": statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0
                }
            }
        }
    
    async def _benchmark_consciousness_evolution(self, reasoner: QuantumTemporalReasoner) -> Dict[str, Any]:
        """Benchmark consciousness evolution capabilities."""
        print("🌟 Benchmarking Consciousness Evolution...")
        
        initial_consciousness = reasoner.consciousness_level
        evolution_prompts = [
            "What is the nature of self-awareness and metacognition?",
            "How do I understand my own reasoning processes?",
            "What are the limits of artificial consciousness?",
            "How can I improve my cognitive capabilities?",
            "What does it mean to think about thinking?"
        ]
        
        consciousness_trajectory = [initial_consciousness]
        reflection_qualities = []
        
        for prompt in evolution_prompts:
            result = await reasoner.quantum_reason(prompt, self_reflect=True)
            consciousness_trajectory.append(result["consciousness_level"])
            
            # Assess reflection quality
            if reasoner.consciousness_engine.reflection_history:
                last_reflection = reasoner.consciousness_engine.reflection_history[-1]
                insight_count = len(last_reflection.get("insights", []))
                reflection_qualities.append(insight_count)
        
        consciousness_growth = consciousness_trajectory[-1] - consciousness_trajectory[0]
        
        # Try explicit evolution
        evolution_result = await reasoner.evolve_consciousness()
        
        return {
            "initial_consciousness": initial_consciousness,
            "final_consciousness": consciousness_trajectory[-1],
            "consciousness_growth": consciousness_growth,
            "evolution_trajectory": consciousness_trajectory,
            "average_reflection_quality": statistics.mean(reflection_qualities) if reflection_qualities else 0,
            "evolution_triggered": evolution_result.get("evolved", False),
            "evolution_potential": evolution_result.get("evolution_potential", 0.0)
        }
    
    async def _benchmark_memory_efficiency(self, reasoner: QuantumTemporalReasoner) -> Dict[str, Any]:
        """Benchmark temporal memory matrix efficiency."""
        print("💾 Benchmarking Memory Efficiency...")
        
        initial_memory_count = len(reasoner.memory_matrix.memory_states)
        
        # Add many memory states to test efficiency
        test_memories = []
        storage_times = []
        
        for i in range(50):
            prompt = f"Test memory efficiency scenario {i} with temporal elements yesterday, today, tomorrow."
            
            start_time = time.time()
            await reasoner.quantum_reason(prompt, self_reflect=False)  # Faster without reflection
            storage_time = time.time() - start_time
            storage_times.append(storage_time)
        
        final_memory_count = len(reasoner.memory_matrix.memory_states)
        memory_growth = final_memory_count - initial_memory_count
        
        # Test memory retrieval
        memory_patterns = await reasoner.memory_matrix.analyze_patterns()
        
        return {
            "initial_memory_states": initial_memory_count,
            "final_memory_states": final_memory_count,
            "memory_growth": memory_growth,
            "average_storage_time": statistics.mean(storage_times),
            "memory_efficiency": memory_growth / len(storage_times),  # States created per operation
            "pattern_analysis_quality": len(memory_patterns) / 10.0  # Normalize by expected patterns
        }
    
    async def _benchmark_concurrent_processing(self) -> Dict[str, Any]:
        """Benchmark concurrent reasoning capabilities."""
        print("⚡ Benchmarking Concurrent Processing...")
        
        # Create multiple reasoner instances
        reasoners = [QuantumTemporalReasoner(consciousness_level=0.7) for _ in range(3)]
        
        concurrent_prompts = [
            "Analyze market trends across temporal dimensions",
            "Evaluate technological innovation impacts over time",
            "Assess climate change implications for the future"
        ]
        
        # Test sequential processing
        start_time = time.time()
        sequential_results = []
        for i, prompt in enumerate(concurrent_prompts):
            result = await reasoners[i].quantum_reason(prompt)
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        
        # Test concurrent processing
        start_time = time.time()
        tasks = [
            reasoner.quantum_reason(prompt) 
            for reasoner, prompt in zip(reasoners, concurrent_prompts)
        ]
        concurrent_results = await asyncio.gather(*tasks)
        concurrent_time = time.time() - start_time
        
        speedup_factor = sequential_time / concurrent_time if concurrent_time > 0 else 1.0
        
        return {
            "sequential_time": sequential_time,
            "concurrent_time": concurrent_time,
            "speedup_factor": speedup_factor,
            "concurrent_efficiency": speedup_factor / len(reasoners),
            "tasks_completed": len(concurrent_results)
        }
    
    async def _benchmark_accuracy_validation(self, reasoner: QuantumTemporalReasoner) -> Dict[str, Any]:
        """Benchmark reasoning accuracy with known scenarios."""
        print("🎯 Benchmarking Accuracy Validation...")
        
        validation_scenarios = [
            {
                "prompt": "The stock market crashed in 1929, today's economy shows resilience, future markets will likely have better safeguards.",
                "expected_dimensions": ["past", "present", "future"],
                "expected_confidence_order": ["past", "present", "future"]  # Past events more certain
            },
            {
                "prompt": "Climate change impacts are evident now and will worsen in the future.",
                "expected_dimensions": ["present", "future"],
                "expected_confidence_order": ["present", "future"]
            },
            {
                "prompt": "AI development accelerated rapidly in recent years.",
                "expected_dimensions": ["past"],
                "expected_confidence_order": ["past"]
            }
        ]
        
        accuracy_scores = []
        
        for scenario in validation_scenarios:
            result = await reasoner.quantum_reason(scenario["prompt"])
            
            # Check dimensional coverage
            dimensions_found = list(result["dimensional_results"].keys())
            expected_dims = scenario["expected_dimensions"]
            
            dimension_accuracy = len(set(dimensions_found) & set(expected_dims)) / len(expected_dims)
            
            # Check confidence ordering (simplified)
            confidence_scores_ordered = []
            for dim in scenario["expected_confidence_order"]:
                if dim in result["confidence_matrix"]:
                    confidence_scores_ordered.append(result["confidence_matrix"][dim])
            
            confidence_accuracy = 1.0 if len(confidence_scores_ordered) == len(set(confidence_scores_ordered)) else 0.5
            
            overall_accuracy = (dimension_accuracy + confidence_accuracy) / 2
            accuracy_scores.append(overall_accuracy)
        
        return {
            "average_accuracy": statistics.mean(accuracy_scores),
            "accuracy_scores": accuracy_scores,
            "validation_scenarios": len(validation_scenarios)
        }
    
    def _calculate_overall_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall performance metrics."""
        
        # Extract key metrics
        decomp_score = min(1.0, results["decomposition_benchmark"]["decomposition_accuracy"])
        
        performance_score = min(1.0, 
            results["reasoning_performance"]["average_confidence"] * 
            (1.0 / max(0.1, results["reasoning_performance"]["average_processing_time"]))
        )
        
        consciousness_score = min(1.0, 
            results["consciousness_evolution"]["consciousness_growth"] * 10  # Scale growth
        )
        
        memory_score = min(1.0, results["memory_efficiency"]["memory_efficiency"] / 10.0)
        
        concurrent_score = min(1.0, results["concurrent_processing"]["speedup_factor"] / 3.0)
        
        accuracy_score = results["accuracy_validation"]["average_accuracy"]
        
        # Calculate weighted overall score
        overall_score = (
            decomp_score * 0.2 +
            performance_score * 0.25 +
            consciousness_score * 0.15 +
            memory_score * 0.1 +
            concurrent_score * 0.1 +
            accuracy_score * 0.2
        )
        
        return {
            "overall_score": overall_score,
            "component_scores": {
                "decomposition": decomp_score,
                "performance": performance_score,
                "consciousness": consciousness_score,
                "memory": memory_score,
                "concurrency": concurrent_score,
                "accuracy": accuracy_score
            },
            "grade": self._score_to_grade(overall_score)
        }
    
    def _score_to_grade(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 0.9:
            return "A+"
        elif score >= 0.85:
            return "A"
        elif score >= 0.8:
            return "A-"
        elif score >= 0.75:
            return "B+"
        elif score >= 0.7:
            return "B"
        elif score >= 0.65:
            return "B-"
        elif score >= 0.6:
            return "C+"
        elif score >= 0.55:
            return "C"
        else:
            return "C-"


async def run_benchmarks():
    """Run the complete benchmark suite."""
    suite = TemporalReasoningSuite()
    results = await suite.run_comprehensive_benchmark()
    
    print("\n" + "=" * 60)
    print("🏆 BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    
    overall = results["overall_performance"]
    print(f"📊 Overall Score: {overall['overall_score']:.2%} ({overall['grade']})")
    
    print("\n📋 Component Scores:")
    for component, score in overall["component_scores"].items():
        print(f"  {component.title()}: {score:.2%}")
    
    print("\n⚡ Performance Highlights:")
    perf = results["reasoning_performance"]
    print(f"  Average Processing Time: {perf['average_processing_time']:.3f}s")
    print(f"  Average Confidence: {perf['average_confidence']:.2%}")
    print(f"  Average Consciousness: {perf['average_consciousness']:.2%}")
    
    print("\n🧠 Consciousness Evolution:")
    consciousness = results["consciousness_evolution"]
    print(f"  Growth: {consciousness['consciousness_growth']:.2%}")
    print(f"  Evolution Triggered: {consciousness['evolution_triggered']}")
    
    print("\n💾 Memory Efficiency:")
    memory = results["memory_efficiency"]
    print(f"  States Created: {memory['memory_growth']}")
    print(f"  Storage Efficiency: {memory['memory_efficiency']:.2f} states/operation")
    
    print("\n⚡ Concurrency Performance:")
    concurrent = results["concurrent_processing"]
    print(f"  Speedup Factor: {concurrent['speedup_factor']:.2f}x")
    print(f"  Efficiency: {concurrent['concurrent_efficiency']:.2%}")
    
    return results


if __name__ == "__main__":
    asyncio.run(run_benchmarks())