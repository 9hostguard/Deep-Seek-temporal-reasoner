"""
Self-Replicating Reasoning Agents
Agents that can self-replicate and mutate based on performance metrics
"""
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import random
import uuid
import json
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from datetime import datetime


class AgentType(Enum):
    EXPLORER = "explorer"
    ANALYZER = "analyzer" 
    SYNTHESIZER = "synthesizer"
    CRITIC = "critic"
    CREATIVE = "creative"
    LOGICAL = "logical"


class MutationType(Enum):
    TRAIT_SHIFT = "trait_shift"
    BEHAVIOR_CHANGE = "behavior_change"
    MEMORY_REORGANIZATION = "memory_reorganization"
    REASONING_STYLE = "reasoning_style"


@dataclass
class AgentGenome:
    """Genetic representation of agent characteristics"""
    agent_id: str
    reasoning_genes: Dict[str, float]  # 0.0 to 1.0 values
    behavior_genes: Dict[str, float]
    memory_genes: Dict[str, float]
    performance_genes: Dict[str, float]
    generation: int
    parent_ids: List[str]
    mutation_count: int


@dataclass
class PerformanceMetrics:
    """Performance tracking for agent evolution"""
    accuracy: float
    speed: float
    creativity: float
    consistency: float
    adaptability: float
    user_satisfaction: float
    energy_efficiency: float
    
    def fitness_score(self) -> float:
        """Calculate overall fitness score"""
        return (
            self.accuracy * 0.25 +
            self.creativity * 0.20 +
            self.adaptability * 0.20 +
            self.user_satisfaction * 0.15 +
            self.consistency * 0.10 +
            self.speed * 0.05 +
            self.energy_efficiency * 0.05
        )


class ReasoningAgent:
    """Self-evolving reasoning agent with cellular automata behavior"""
    
    def __init__(self, agent_type: AgentType, genome: Optional[AgentGenome] = None):
        self.agent_type = agent_type
        self.agent_id = str(uuid.uuid4())
        self.genome = genome or self._generate_random_genome()
        self.performance_history: List[PerformanceMetrics] = []
        self.offspring_count = 0
        self.age = 0
        self.energy = 100.0
        self.current_task = None
        self.reasoning_strategy = self._determine_reasoning_strategy()
        
    def _generate_random_genome(self) -> AgentGenome:
        """Generate random genome for new agent"""
        return AgentGenome(
            agent_id=self.agent_id,
            reasoning_genes={
                "logical_strength": random.uniform(0.3, 0.9),
                "creative_factor": random.uniform(0.2, 0.8),
                "analytical_depth": random.uniform(0.4, 0.95),
                "intuitive_leap": random.uniform(0.1, 0.7),
                "pattern_recognition": random.uniform(0.5, 0.9)
            },
            behavior_genes={
                "exploration_drive": random.uniform(0.2, 0.8),
                "collaboration_tendency": random.uniform(0.3, 0.9),
                "risk_tolerance": random.uniform(0.1, 0.7),
                "persistence": random.uniform(0.4, 0.9),
                "adaptability": random.uniform(0.5, 0.9)
            },
            memory_genes={
                "retention_strength": random.uniform(0.6, 0.95),
                "association_ability": random.uniform(0.4, 0.8),
                "forgetting_rate": random.uniform(0.05, 0.3),
                "compression_efficiency": random.uniform(0.5, 0.9)
            },
            performance_genes={
                "processing_speed": random.uniform(0.4, 0.9),
                "energy_efficiency": random.uniform(0.3, 0.8),
                "accuracy_bias": random.uniform(0.6, 0.95),
                "optimization_drive": random.uniform(0.5, 0.9)
            },
            generation=0,
            parent_ids=[],
            mutation_count=0
        )
        
    def _determine_reasoning_strategy(self) -> Dict[str, Any]:
        """Determine reasoning strategy based on genome"""
        genes = self.genome.reasoning_genes
        
        if genes["logical_strength"] > 0.7:
            primary_strategy = "logical_deduction"
        elif genes["creative_factor"] > 0.6:
            primary_strategy = "creative_synthesis"
        elif genes["analytical_depth"] > 0.8:
            primary_strategy = "deep_analysis"
        elif genes["intuitive_leap"] > 0.6:
            primary_strategy = "intuitive_reasoning"
        else:
            primary_strategy = "balanced_approach"
            
        return {
            "primary": primary_strategy,
            "logical_weight": genes["logical_strength"],
            "creative_weight": genes["creative_factor"],
            "analytical_weight": genes["analytical_depth"],
            "intuitive_weight": genes["intuitive_leap"]
        }
        
    def reason(self, prompt: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Perform reasoning using agent's evolved strategy"""
        self.age += 1
        self.energy -= random.uniform(1, 5)  # Reasoning costs energy
        
        strategy = self.reasoning_strategy
        
        # Apply reasoning strategy
        if strategy["primary"] == "logical_deduction":
            result = self._logical_reasoning(prompt, context)
        elif strategy["primary"] == "creative_synthesis":
            result = self._creative_reasoning(prompt, context)
        elif strategy["primary"] == "deep_analysis":
            result = self._analytical_reasoning(prompt, context)
        elif strategy["primary"] == "intuitive_reasoning":
            result = self._intuitive_reasoning(prompt, context)
        else:
            result = self._balanced_reasoning(prompt, context)
            
        # Apply genetic modifiers
        result["confidence"] *= self.genome.performance_genes["accuracy_bias"]
        result["processing_time"] = random.uniform(0.5, 2.0) / self.genome.performance_genes["processing_speed"]
        
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "reasoning_result": result,
            "energy_remaining": self.energy,
            "age": self.age,
            "strategy_used": strategy["primary"]
        }
        
    def _logical_reasoning(self, prompt: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Logical reasoning approach"""
        logical_strength = self.genome.reasoning_genes["logical_strength"]
        
        return {
            "approach": "logical_deduction",
            "reasoning": f"Applying logical analysis to '{prompt}' with systematic deduction and evidence-based conclusions.",
            "confidence": 0.7 + (logical_strength * 0.25),
            "logical_steps": [
                "Premise identification",
                "Evidence evaluation", 
                "Logical inference",
                "Conclusion validation"
            ]
        }
        
    def _creative_reasoning(self, prompt: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Creative reasoning approach"""
        creative_factor = self.genome.reasoning_genes["creative_factor"]
        
        return {
            "approach": "creative_synthesis",
            "reasoning": f"Exploring '{prompt}' through creative ideation and innovative perspective generation.",
            "confidence": 0.6 + (creative_factor * 0.3),
            "creative_elements": [
                "Analogical thinking",
                "Lateral connections",
                "Novel combinations",
                "Imaginative scenarios"
            ]
        }
        
    def _analytical_reasoning(self, prompt: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Deep analytical reasoning"""
        analytical_depth = self.genome.reasoning_genes["analytical_depth"]
        
        return {
            "approach": "deep_analysis",
            "reasoning": f"Conducting thorough analytical examination of '{prompt}' with multi-layered investigation.",
            "confidence": 0.75 + (analytical_depth * 0.2),
            "analysis_layers": [
                "Surface-level examination",
                "Structural analysis",
                "Causal investigation",
                "Systems-level perspective"
            ]
        }
        
    def _intuitive_reasoning(self, prompt: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Intuitive reasoning approach"""
        intuitive_leap = self.genome.reasoning_genes["intuitive_leap"]
        
        return {
            "approach": "intuitive_reasoning",
            "reasoning": f"Applying intuitive insights to '{prompt}' through pattern recognition and holistic understanding.",
            "confidence": 0.5 + (intuitive_leap * 0.4),
            "intuitive_insights": [
                "Pattern recognition",
                "Holistic perception",
                "Gut feeling integration",
                "Subconscious processing"
            ]
        }
        
    def _balanced_reasoning(self, prompt: str, context: Optional[Dict]) -> Dict[str, Any]:
        """Balanced reasoning combining multiple approaches"""
        genes = self.genome.reasoning_genes
        
        return {
            "approach": "balanced_reasoning",
            "reasoning": f"Applying balanced multi-modal reasoning to '{prompt}' integrating logical, creative, analytical, and intuitive elements.",
            "confidence": 0.65 + (sum(genes.values()) / len(genes) * 0.25),
            "reasoning_components": {
                "logical": genes["logical_strength"],
                "creative": genes["creative_factor"],
                "analytical": genes["analytical_depth"],
                "intuitive": genes["intuitive_leap"]
            }
        }
        
    def mutate(self, mutation_type: MutationType, intensity: float = 0.1) -> bool:
        """Mutate agent genome"""
        success = False
        
        if mutation_type == MutationType.TRAIT_SHIFT:
            # Mutate reasoning genes
            gene_key = random.choice(list(self.genome.reasoning_genes.keys()))
            old_value = self.genome.reasoning_genes[gene_key]
            mutation = (random.random() - 0.5) * intensity * 2
            new_value = np.clip(old_value + mutation, 0.0, 1.0)
            self.genome.reasoning_genes[gene_key] = new_value
            success = True
            
        elif mutation_type == MutationType.BEHAVIOR_CHANGE:
            # Mutate behavior genes
            gene_key = random.choice(list(self.genome.behavior_genes.keys()))
            old_value = self.genome.behavior_genes[gene_key]
            mutation = (random.random() - 0.5) * intensity * 2
            new_value = np.clip(old_value + mutation, 0.0, 1.0)
            self.genome.behavior_genes[gene_key] = new_value
            success = True
            
        elif mutation_type == MutationType.MEMORY_REORGANIZATION:
            # Mutate memory genes
            gene_key = random.choice(list(self.genome.memory_genes.keys()))
            old_value = self.genome.memory_genes[gene_key]
            mutation = (random.random() - 0.5) * intensity * 2
            new_value = np.clip(old_value + mutation, 0.0, 1.0)
            self.genome.memory_genes[gene_key] = new_value
            success = True
            
        if success:
            self.genome.mutation_count += 1
            self.reasoning_strategy = self._determine_reasoning_strategy()
            
        return success
        
    def reproduce(self, partner: 'ReasoningAgent', mutation_rate: float = 0.1) -> 'ReasoningAgent':
        """Reproduce with another agent to create offspring"""
        offspring_genome = self._crossover_genomes(self.genome, partner.genome)
        
        # Apply mutations
        if random.random() < mutation_rate:
            mutation_type = random.choice(list(MutationType))
            offspring_agent = ReasoningAgent(self.agent_type, offspring_genome)
            offspring_agent.mutate(mutation_type, intensity=0.15)
        else:
            offspring_agent = ReasoningAgent(self.agent_type, offspring_genome)
            
        self.offspring_count += 1
        partner.offspring_count += 1
        
        return offspring_agent
        
    def _crossover_genomes(self, genome1: AgentGenome, genome2: AgentGenome) -> AgentGenome:
        """Create offspring genome through crossover"""
        offspring_reasoning_genes = {}
        offspring_behavior_genes = {}
        offspring_memory_genes = {}
        offspring_performance_genes = {}
        
        # Crossover reasoning genes
        for gene_key in genome1.reasoning_genes.keys():
            if random.random() < 0.5:
                offspring_reasoning_genes[gene_key] = genome1.reasoning_genes[gene_key]
            else:
                offspring_reasoning_genes[gene_key] = genome2.reasoning_genes[gene_key]
                
        # Crossover behavior genes
        for gene_key in genome1.behavior_genes.keys():
            if random.random() < 0.5:
                offspring_behavior_genes[gene_key] = genome1.behavior_genes[gene_key]
            else:
                offspring_behavior_genes[gene_key] = genome2.behavior_genes[gene_key]
                
        # Crossover memory genes
        for gene_key in genome1.memory_genes.keys():
            if random.random() < 0.5:
                offspring_memory_genes[gene_key] = genome1.memory_genes[gene_key]
            else:
                offspring_memory_genes[gene_key] = genome2.memory_genes[gene_key]
                
        # Crossover performance genes
        for gene_key in genome1.performance_genes.keys():
            if random.random() < 0.5:
                offspring_performance_genes[gene_key] = genome1.performance_genes[gene_key]
            else:
                offspring_performance_genes[gene_key] = genome2.performance_genes[gene_key]
                
        return AgentGenome(
            agent_id=str(uuid.uuid4()),
            reasoning_genes=offspring_reasoning_genes,
            behavior_genes=offspring_behavior_genes,
            memory_genes=offspring_memory_genes,
            performance_genes=offspring_performance_genes,
            generation=max(genome1.generation, genome2.generation) + 1,
            parent_ids=[genome1.agent_id, genome2.agent_id],
            mutation_count=0
        )
        
    def evaluate_performance(self, task_results: List[Dict]) -> PerformanceMetrics:
        """Evaluate agent performance and update metrics"""
        if not task_results:
            return PerformanceMetrics(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
            
        # Calculate performance metrics from results
        accuracy = np.mean([r.get("accuracy", 0.5) for r in task_results])
        mean_processing_time = np.mean([r.get("processing_time", 1.0) for r in task_results])
        if mean_processing_time < 1e-8:
            speed = 0.0  # or set to a default value, e.g., 1.0 if desired
        else:
            speed = 1.0 / mean_processing_time
        creativity = np.mean([r.get("creativity_score", 0.5) for r in task_results])
        consistency = 1.0 - np.std([r.get("confidence", 0.5) for r in task_results])
        adaptability = self.genome.behavior_genes["adaptability"]
        user_satisfaction = np.mean([r.get("user_rating", 0.5) for r in task_results])
        energy_efficiency = self.energy / 100.0
        
        metrics = PerformanceMetrics(
            accuracy=accuracy,
            speed=min(speed, 1.0),
            creativity=creativity,
            consistency=max(consistency, 0.0),
            adaptability=adaptability,
            user_satisfaction=user_satisfaction,
            energy_efficiency=energy_efficiency
        )
        
        self.performance_history.append(metrics)
        return metrics


class SelfReplicatingAgentSwarm:
    """Swarm of self-replicating and evolving reasoning agents"""
    
    def __init__(self, initial_population: int = 20, max_population: int = 100):
        self.agents: Dict[str, ReasoningAgent] = {}
        self.max_population = max_population
        self.generation_count = 0
        self.evolution_history = []
        
        # Create initial population
        for _ in range(initial_population):
            agent_type = random.choice(list(AgentType))
            agent = ReasoningAgent(agent_type)
            self.agents[agent.agent_id] = agent
            
    def collective_reasoning(self, prompt: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Perform collective reasoning using multiple agents"""
        agent_results = []
        
        # Select subset of agents for reasoning
        selected_agents = random.sample(list(self.agents.values()), min(5, len(self.agents)))
        
        for agent in selected_agents:
            if agent.energy > 10:  # Only use agents with sufficient energy
                result = agent.reason(prompt, context)
                agent_results.append(result)
                
        # Synthesize results
        if not agent_results:
            return {"error": "No agents available for reasoning"}
            
        collective_result = self._synthesize_collective_reasoning(agent_results)
        
        return {
            "prompt": prompt,
            "individual_agent_results": agent_results,
            "collective_synthesis": collective_result,
            "agents_used": len(agent_results),
            "total_agents": len(self.agents),
            "generation": self.generation_count
        }
        
    def _synthesize_collective_reasoning(self, agent_results: List[Dict]) -> Dict[str, Any]:
        """Synthesize reasoning from multiple agents"""
        if not agent_results:
            return {}
            
        # Collect reasoning approaches
        approaches = [r["reasoning_result"]["approach"] for r in agent_results]
        confidences = [r["reasoning_result"]["confidence"] for r in agent_results]
        
        # Calculate collective confidence
        collective_confidence = np.mean(confidences)
        
        # Determine dominant approach
        approach_counts = {}
        for approach in approaches:
            approach_counts[approach] = approach_counts.get(approach, 0) + 1
            
        dominant_approach = max(approach_counts.keys(), key=lambda x: approach_counts[x])
        
        # Create synthesis
        reasoning_texts = [r["reasoning_result"]["reasoning"] for r in agent_results]
        
        return {
            "dominant_approach": dominant_approach,
            "collective_confidence": collective_confidence,
            "approach_diversity": len(set(approaches)),
            "reasoning_synthesis": f"Collective reasoning combining {len(agent_results)} agents with approaches: {', '.join(set(approaches))}",
            "confidence_range": [min(confidences), max(confidences)],
            "agent_energy_levels": [r["energy_remaining"] for r in agent_results]
        }
        
    def evolve_generation(self, selection_pressure: float = 0.3) -> Dict[str, Any]:
        """Evolve the agent population through selection, reproduction, and mutation"""
        if len(self.agents) < 2:
            return {"error": "Insufficient agents for evolution"}
            
        # Evaluate all agents
        agent_performances = {}
        for agent_id, agent in self.agents.items():
            # Simulate task results for performance evaluation
            simulated_results = [
                {
                    "accuracy": random.uniform(0.3, 0.95),
                    "processing_time": random.uniform(0.5, 3.0),
                    "creativity_score": random.uniform(0.2, 0.8),
                    "confidence": random.uniform(0.4, 0.9),
                    "user_rating": random.uniform(0.3, 0.9)
                }
                for _ in range(3)
            ]
            
            performance = agent.evaluate_performance(simulated_results)
            agent_performances[agent_id] = performance.fitness_score()
            
        # Selection: keep top performers
        sorted_agents = sorted(
            agent_performances.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        survivors_count = max(2, int(len(self.agents) * (1 - selection_pressure)))
        survivors = [agent_id for agent_id, _ in sorted_agents[:survivors_count]]
        
        # Reproduction: create offspring from survivors
        new_agents = {}
        
        # Keep survivors
        for agent_id in survivors:
            new_agents[agent_id] = self.agents[agent_id]
            
        # Create offspring
        while len(new_agents) < self.max_population and len(survivors) >= 2:
            parent1_id = random.choice(survivors)
            parent2_id = random.choice(survivors)
            
            if parent1_id != parent2_id:
                parent1 = self.agents[parent1_id]
                parent2 = self.agents[parent2_id]
                
                offspring = parent1.reproduce(parent2, mutation_rate=0.15)
                new_agents[offspring.agent_id] = offspring
                
        self.agents = new_agents
        self.generation_count += 1
        
        # Record evolution history
        evolution_stats = {
            "generation": self.generation_count,
            "population_size": len(self.agents),
            "survivors_count": len(survivors),
            "new_offspring": len(new_agents) - len(survivors),
            "average_fitness": np.mean(list(agent_performances.values())),
            "max_fitness": max(agent_performances.values()),
            "min_fitness": min(agent_performances.values()),
            "fitness_diversity": np.std(list(agent_performances.values()))
        }
        
        self.evolution_history.append(evolution_stats)
        
        return evolution_stats
        
    def get_swarm_statistics(self) -> Dict[str, Any]:
        """Get comprehensive swarm statistics"""
        if not self.agents:
            return {"error": "No agents in swarm"}
            
        # Agent type distribution
        type_counts = {}
        for agent in self.agents.values():
            agent_type = agent.agent_type.value
            type_counts[agent_type] = type_counts.get(agent_type, 0) + 1
            
        # Generation distribution
        generations = [agent.genome.generation for agent in self.agents.values()]
        
        # Energy levels
        energy_levels = [agent.energy for agent in self.agents.values()]
        
        # Performance history
        if self.evolution_history:
            recent_evolution = self.evolution_history[-5:]  # Last 5 generations
        else:
            recent_evolution = []
            
        return {
            "total_agents": len(self.agents),
            "agent_type_distribution": type_counts,
            "generation_statistics": {
                "current_generation": self.generation_count,
                "min_generation": min(generations) if generations else 0,
                "max_generation": max(generations) if generations else 0,
                "average_generation": np.mean(generations) if generations else 0
            },
            "energy_statistics": {
                "average_energy": np.mean(energy_levels) if energy_levels else 0,
                "min_energy": min(energy_levels) if energy_levels else 0,
                "max_energy": max(energy_levels) if energy_levels else 0
            },
            "evolution_history": recent_evolution,
            "mutation_counts": [agent.genome.mutation_count for agent in self.agents.values()]
        }
        
    async def continuous_evolution(self, evolution_cycles: int = 10, cycle_delay: float = 30.0):
        """Run continuous evolution cycles"""
        for cycle in range(evolution_cycles):
            # Perform collective reasoning tasks to evaluate agents
            test_prompts = [
                "Analyze the future of artificial intelligence",
                "Solve complex environmental challenges",
                "Design innovative educational systems"
            ]
            
            for prompt in test_prompts:
                await asyncio.sleep(0.1)  # Small delay between tasks
                self.collective_reasoning(prompt)
                
            # Evolve generation
            evolution_stats = self.evolve_generation()
            
            print(f"Evolution cycle {cycle + 1}: {evolution_stats}")
            
            await asyncio.sleep(cycle_delay)