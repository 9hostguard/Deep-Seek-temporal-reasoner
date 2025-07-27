"""
Comprehensive test suite for 4D Quantum Temporal Reasoning Engine.
"""

import pytest
import asyncio
from typing import Dict, Any
import numpy as np

from core.quantum_temporal_reasoner import QuantumTemporalReasoner
from core.decomposition import decompose, quantum_decompose
from core.temporal_memory_matrix import TemporalMemoryMatrix
from core.consciousness_engine import ConsciousnessEngine
from core.plugins.deepseek_quantum_plugin import DeepSeekQuantumPlugin


class TestQuantumTemporalReasoning:
    """Test suite for quantum temporal reasoning functionality."""
    
    @pytest.fixture
    def reasoner(self):
        """Create reasoner instance for testing."""
        return QuantumTemporalReasoner(consciousness_level=0.8)
    
    @pytest.fixture
    def memory_matrix(self):
        """Create memory matrix instance for testing."""
        return TemporalMemoryMatrix(dimensions=4)
    
    @pytest.fixture
    def consciousness_engine(self):
        """Create consciousness engine instance for testing."""
        return ConsciousnessEngine(initial_level=0.7)
    
    def test_decomposition_basic(self):
        """Test basic temporal decomposition."""
        prompt = "I learned Python yesterday, I am coding today, and I will deploy tomorrow."
        result = decompose(prompt)
        
        assert isinstance(result, dict)
        assert "past" in result
        assert "present" in result
        assert "future" in result
        
        # Check that temporal elements are correctly assigned
        assert "learned" in result["past"] or "yesterday" in result["past"]
        assert "coding" in result["present"] or "today" in result["present"]
        assert "deploy" in result["future"] or "tomorrow" in result["future"]
    
    def test_quantum_decomposition(self):
        """Test quantum decomposition functionality."""
        prompt = "The market crashed last week, prices are volatile now, and recovery is expected next month."
        result = quantum_decompose(prompt)
        
        assert isinstance(result, dict)
        assert "temporal_segments" in result
        assert "confidence_matrix" in result
        assert "quantum_coherence" in result
        assert "dimensional_depth" in result
        
        # Check confidence matrix has all dimensions
        confidence_matrix = result["confidence_matrix"]
        assert all(dim in confidence_matrix for dim in ["past", "present", "future"])
        assert all(0 <= conf <= 1 for conf in confidence_matrix.values())
    
    @pytest.mark.asyncio
    async def test_quantum_reasoning_basic(self, reasoner):
        """Test basic quantum reasoning functionality."""
        prompt = "What are the implications of AI development?"
        result = await reasoner.quantum_reason(prompt)
        
        assert isinstance(result, dict)
        assert "session_id" in result
        assert "query" in result
        assert "temporal_breakdown" in result
        assert "dimensional_results" in result
        assert "synthesis" in result
        assert "consciousness_level" in result
        
        # Check consciousness level is valid
        assert 0 <= result["consciousness_level"] <= 1
    
    @pytest.mark.asyncio
    async def test_memory_matrix_storage(self, memory_matrix):
        """Test temporal memory storage and retrieval."""
        prompt = "Test memory storage functionality"
        temporal_breakdown = {
            "temporal_segments": {"past": "", "present": "Test memory", "future": ""},
            "confidence_matrix": {"past": 0.1, "present": 0.9, "future": 0.1},
            "quantum_coherence": 0.8
        }
        
        # Store memory
        memory_key = await memory_matrix.store_temporal_state(prompt, temporal_breakdown, 0.8)
        assert isinstance(memory_key, str)
        assert len(memory_key) > 0
        
        # Retrieve memory
        retrieved_state = await memory_matrix.retrieve_temporal_state(memory_key)
        assert retrieved_state is not None
        assert retrieved_state["prompt"] == prompt
        assert retrieved_state["consciousness_level"] == 0.8
    
    @pytest.mark.asyncio
    async def test_consciousness_reflection(self, consciousness_engine):
        """Test consciousness self-reflection functionality."""
        prompt = "Test self-reflection capabilities"
        dimensional_results = {
            "present": {
                "response": "Testing reflection mechanisms",
                "confidence": 0.8,
                "coherence": 0.7
            }
        }
        quantum_state = {"coherence": 0.85, "entanglement": 0.9}
        
        result = await consciousness_engine.self_reflect(prompt, dimensional_results, quantum_state)
        
        assert isinstance(result, dict)
        assert "new_consciousness_level" in result
        assert "consciousness_evolution" in result
        assert "reasoning_analysis" in result
        assert "awareness_insights" in result
        
        # Check consciousness level is valid
        assert 0 <= result["new_consciousness_level"] <= 1
    
    @pytest.mark.asyncio
    async def test_deepseek_plugin(self):
        """Test DeepSeek quantum plugin functionality."""
        plugin = DeepSeekQuantumPlugin()
        
        result = await plugin.quantum_inference(
            "Test quantum inference",
            dimension="present",
            consciousness_level=0.8
        )
        
        assert isinstance(result, dict)
        assert "response" in result
        assert "confidence" in result
        assert "coherence" in result
        assert "quantum_metrics" in result
        
        # Check confidence is valid
        assert 0 <= result["confidence"] <= 1


class TestIntegration:
    """Integration tests for complete system functionality."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_reasoning(self):
        """Test complete end-to-end reasoning workflow."""
        reasoner = QuantumTemporalReasoner(consciousness_level=0.8)
        
        prompt = "How will climate change affect global agriculture in the next decade?"
        result = await reasoner.quantum_reason(prompt, self_reflect=True)
        
        # Verify complete response structure
        assert "session_id" in result
        assert "temporal_breakdown" in result
        assert "dimensional_results" in result
        assert "synthesis" in result
        assert "consciousness_level" in result
        assert "confidence_matrix" in result
        
        # Check that reasoning produced meaningful results
        dimensional_results = result["dimensional_results"]
        assert len(dimensional_results) > 0
        
        # Verify each dimensional result has required fields
        for dimension, dim_result in dimensional_results.items():
            assert "response" in dim_result
            assert "confidence" in dim_result
            assert isinstance(dim_result["response"], str)
            assert len(dim_result["response"]) > 0
    
    @pytest.mark.asyncio
    async def test_consciousness_evolution_cycle(self):
        """Test consciousness evolution through multiple reasoning cycles."""
        reasoner = QuantumTemporalReasoner(consciousness_level=0.5)
        initial_consciousness = reasoner.consciousness_level
        
        # Perform multiple reasoning cycles
        prompts = [
            "What is the nature of consciousness?",
            "How do we understand temporal relationships?",
            "What are the implications of quantum reasoning?",
            "How can AI achieve self-awareness?"
        ]
        
        for prompt in prompts:
            await reasoner.quantum_reason(prompt, self_reflect=True)
        
        # Check if consciousness has evolved
        final_consciousness = reasoner.consciousness_level
        
        # Consciousness should show some change through reflection
        assert reasoner.reasoning_metrics["queries_processed"] == len(prompts)
        assert len(reasoner.consciousness_engine.reflection_history) == len(prompts)
    
    @pytest.mark.asyncio
    async def test_temporal_insights(self):
        """Test temporal insights functionality."""
        reasoner = QuantumTemporalReasoner()
        
        # Perform some reasoning to build up memory
        await reasoner.quantum_reason("Past: Learning, Present: Applying, Future: Innovating")
        
        # Get temporal insights
        insights = await reasoner.get_temporal_insights("What patterns do you see?")
        
        assert isinstance(insights, dict)
        assert "memory_patterns" in insights
        assert "consciousness_state" in insights
        assert "quantum_metrics" in insights
        assert "performance_metrics" in insights


class TestPerformance:
    """Performance and benchmarking tests."""
    
    @pytest.mark.asyncio
    async def test_reasoning_performance(self):
        """Test reasoning performance meets requirements."""
        reasoner = QuantumTemporalReasoner()
        
        import time
        start_time = time.time()
        
        result = await reasoner.quantum_reason("Quick performance test")
        
        processing_time = time.time() - start_time
        
        # Should complete within reasonable time (< 5 seconds for demo)
        assert processing_time < 5.0
        assert "processing_time" in result
        assert result["processing_time"] < 5.0
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test memory usage efficiency."""
        memory_matrix = TemporalMemoryMatrix(memory_capacity=100)
        
        # Fill memory to capacity
        for i in range(120):  # Exceed capacity
            prompt = f"Test prompt {i}"
            temporal_breakdown = {
                "temporal_segments": {"past": "", "present": f"Test {i}", "future": ""},
                "confidence_matrix": {"past": 0.1, "present": 0.9, "future": 0.1},
                "quantum_coherence": 0.8
            }
            await memory_matrix.store_temporal_state(prompt, temporal_breakdown, 0.8)
        
        # Memory should not exceed capacity
        assert len(memory_matrix.memory_states) <= memory_matrix.memory_capacity
    
    @pytest.mark.asyncio
    async def test_concurrent_reasoning(self):
        """Test concurrent reasoning capabilities."""
        reasoner = QuantumTemporalReasoner()
        
        # Create multiple concurrent reasoning tasks
        tasks = []
        for i in range(5):
            task = reasoner.quantum_reason(f"Concurrent test {i}")
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # All tasks should complete successfully
        assert len(results) == 5
        for result in results:
            assert "session_id" in result
            assert "consciousness_level" in result


class TestEdgeCases:
    """Edge case and boundary testing."""
    
    def test_empty_prompt(self):
        """Test handling of empty prompts."""
        result = decompose("")
        assert result["present"] == ""  # Should handle gracefully
    
    def test_very_long_prompt(self):
        """Test handling of very long prompts."""
        long_prompt = "test " * 1000  # 1000 words
        result = decompose(long_prompt)
        
        assert isinstance(result, dict)
        assert all(key in result for key in ["past", "present", "future"])
    
    @pytest.mark.asyncio
    async def test_extreme_consciousness_levels(self):
        """Test extreme consciousness level values."""
        # Test very low consciousness
        reasoner_low = QuantumTemporalReasoner(consciousness_level=0.01)
        result_low = await reasoner_low.quantum_reason("Test low consciousness")
        assert 0 <= result_low["consciousness_level"] <= 1
        
        # Test very high consciousness
        reasoner_high = QuantumTemporalReasoner(consciousness_level=0.99)
        result_high = await reasoner_high.quantum_reason("Test high consciousness")
        assert 0 <= result_high["consciousness_level"] <= 1
    
    @pytest.mark.asyncio
    async def test_malformed_inputs(self):
        """Test handling of malformed inputs."""
        reasoner = QuantumTemporalReasoner()
        
        # Test with special characters
        result = await reasoner.quantum_reason("Test with @#$%^&*() special chars!")
        assert "dimensional_results" in result
        
        # Test with unicode
        result = await reasoner.quantum_reason("Test with unicode: 測試 🚀 ∞")
        assert "dimensional_results" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])