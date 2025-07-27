"""
Genetic Customization System for breeding avatars and reasoning agents
"""
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import random
import json
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from datetime import datetime
from .holographic_memory import PersonalityTrait, AvatarPersonality
from .self_replicating_agents import ReasoningAgent, AgentType


class GeneticOperation(Enum):
    CROSSOVER = "crossover"
    MUTATION = "mutation"
    SELECTION = "selection"
    HYBRIDIZATION = "hybridization"


class FitnessMetric(Enum):
    REASONING_ACCURACY = "reasoning_accuracy"
    CREATIVITY_SCORE = "creativity_score"
    USER_SATISFACTION = "user_satisfaction"
    ADAPTABILITY = "adaptability"
    EFFICIENCY = "efficiency"
    NOVELTY = "novelty"
    COHERENCE = "coherence"


@dataclass
class GeneticProfile:
    """Genetic profile for avatars and agents"""
    profile_id: str
    trait_genes: Dict[str, float]  # Personality/reasoning traits
    behavior_genes: Dict[str, float]  # Behavioral patterns
    memory_genes: Dict[str, float]  # Memory characteristics
    meta_genes: Dict[str, float]  # Meta-cognitive abilities
    fitness_scores: Dict[FitnessMetric, float]
    generation: int
    lineage: List[str]  # Parent IDs
    mutations: List[Dict[str, Any]]
    breeding_history: List[Dict[str, Any]]


@dataclass
class BreedingResult:
    """Result of breeding operation"""
    offspring_id: str
    parent_ids: List[str]
    genetic_operations: List[GeneticOperation]
    trait_inheritance: Dict[str, Dict[str, Any]]
    mutation_effects: List[Dict[str, Any]]
    predicted_fitness: Dict[FitnessMetric, float]
    breeding_success: bool
    novel_traits: List[str]


class GeneticCustomizationSystem:
    """System for breeding avatars and agents using genetic algorithms"""
    
    def __init__(self, population_size: int = 100, mutation_rate: float = 0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.genetic_profiles: Dict[str, GeneticProfile] = {}
        self.breeding_history: List[BreedingResult] = []
        self.fitness_weights = self._initialize_fitness_weights()
        self.trait_compatibility_matrix = self._initialize_compatibility_matrix()
        
    def _initialize_fitness_weights(self) -> Dict[FitnessMetric, float]:
        """Initialize weights for different fitness metrics"""
        return {
            FitnessMetric.REASONING_ACCURACY: 0.25,
            FitnessMetric.CREATIVITY_SCORE: 0.20,
            FitnessMetric.USER_SATISFACTION: 0.20,
            FitnessMetric.ADAPTABILITY: 0.15,
            FitnessMetric.EFFICIENCY: 0.10,
            FitnessMetric.NOVELTY: 0.05,
            FitnessMetric.COHERENCE: 0.05
        }
        
    def _initialize_compatibility_matrix(self) -> Dict[Tuple[str, str], float]:
        """Initialize trait compatibility matrix for breeding"""
        traits = [trait.value for trait in PersonalityTrait]
        compatibility = {}
        
        for i, trait1 in enumerate(traits):
            for trait2 in traits[i:]:
                # Some traits are more compatible than others
                if trait1 == trait2:
                    compatibility[(trait1, trait2)] = 1.0
                elif (trait1, trait2) in [("curiosity", "creativity"), ("empathy", "emotional_intelligence"), 
                                        ("logic", "analytical"), ("adaptability", "intuition")]:
                    compatibility[(trait1, trait2)] = 0.9
                elif (trait1, trait2) in [("logic", "creativity"), ("empathy", "analytical")]:
                    compatibility[(trait1, trait2)] = 0.3  # Less compatible
                else:
                    compatibility[(trait1, trait2)] = random.uniform(0.4, 0.8)
                    
        return compatibility
        
    def create_genetic_profile(self, entity_id: str, 
                             avatar: Optional[AvatarPersonality] = None,
                             agent: Optional[ReasoningAgent] = None) -> str:
        """Create genetic profile from avatar or agent"""
        profile_id = f"genetic_{entity_id}_{uuid.uuid4()}"
        
        if avatar:
            trait_genes = {trait.value: value for trait, value in avatar.traits.items()}
            behavior_genes = {
                "evolution_rate": avatar.user_feedback_influence,
                "quantum_sensitivity": avatar.quantum_randomness_factor,
                "memory_retention": len(avatar.memory_associations) / 100.0,
                "adaptation_speed": 0.5  # Default value
            }
            memory_genes = {
                "association_strength": len(avatar.memory_associations) / 50.0,
                "emotional_binding": 0.7,  # Default
                "persistence": 0.8,  # Default
                "plasticity": avatar.user_feedback_influence
            }
            
        elif agent:
            trait_genes = agent.genome.reasoning_genes.copy()
            behavior_genes = agent.genome.behavior_genes.copy()
            memory_genes = agent.genome.memory_genes.copy()
            
        else:
            # Create random profile
            trait_genes = {trait.value: random.uniform(0.2, 0.8) for trait in PersonalityTrait}
            behavior_genes = {
                "exploration": random.uniform(0.3, 0.9),
                "cooperation": random.uniform(0.4, 0.8),
                "innovation": random.uniform(0.2, 0.7),
                "persistence": random.uniform(0.5, 0.9)
            }
            memory_genes = {
                "retention": random.uniform(0.6, 0.95),
                "association": random.uniform(0.4, 0.8),
                "flexibility": random.uniform(0.3, 0.7),
                "integration": random.uniform(0.5, 0.9)
            }
            
        meta_genes = {
            "self_awareness": random.uniform(0.4, 0.8),
            "learning_rate": random.uniform(0.3, 0.9),
            "metacognition": random.uniform(0.5, 0.85),
            "consciousness_depth": random.uniform(0.6, 0.9)
        }
        
        # Initialize fitness scores
        fitness_scores = {metric: random.uniform(0.3, 0.7) for metric in FitnessMetric}
        
        profile = GeneticProfile(
            profile_id=profile_id,
            trait_genes=trait_genes,
            behavior_genes=behavior_genes,
            memory_genes=memory_genes,
            meta_genes=meta_genes,
            fitness_scores=fitness_scores,
            generation=0,
            lineage=[],
            mutations=[],
            breeding_history=[]
        )
        
        self.genetic_profiles[profile_id] = profile
        return profile_id
        
    def breed_entities(self, parent1_id: str, parent2_id: str,
                      breeding_strategy: str = "balanced") -> BreedingResult:
        """Breed two entities to create offspring"""
        if parent1_id not in self.genetic_profiles or parent2_id not in self.genetic_profiles:
            raise ValueError("Parent profiles not found")
            
        parent1 = self.genetic_profiles[parent1_id]
        parent2 = self.genetic_profiles[parent2_id]
        
        # Assess breeding compatibility
        compatibility = self._assess_breeding_compatibility(parent1, parent2)
        
        if compatibility < 0.3:
            return BreedingResult(
                offspring_id="",
                parent_ids=[parent1_id, parent2_id],
                genetic_operations=[],
                trait_inheritance={},
                mutation_effects=[],
                predicted_fitness={},
                breeding_success=False,
                novel_traits=[]
            )
            
        # Perform genetic crossover
        offspring_genes = self._perform_crossover(parent1, parent2, breeding_strategy)
        
        # Apply mutations
        mutations = self._apply_mutations(offspring_genes)
        
        # Create offspring profile
        offspring_id = f"offspring_{uuid.uuid4()}"
        offspring_profile = GeneticProfile(
            profile_id=offspring_id,
            trait_genes=offspring_genes["trait_genes"],
            behavior_genes=offspring_genes["behavior_genes"],
            memory_genes=offspring_genes["memory_genes"],
            meta_genes=offspring_genes["meta_genes"],
            fitness_scores={},  # Will be evaluated later
            generation=max(parent1.generation, parent2.generation) + 1,
            lineage=[parent1_id, parent2_id],
            mutations=mutations,
            breeding_history=[]
        )
        
        # Predict fitness
        predicted_fitness = self._predict_offspring_fitness(parent1, parent2, offspring_profile)
        offspring_profile.fitness_scores = predicted_fitness
        
        self.genetic_profiles[offspring_id] = offspring_profile
        
        # Detect novel traits
        novel_traits = self._detect_novel_traits(offspring_profile, [parent1, parent2])
        
        # Create breeding result
        breeding_result = BreedingResult(
            offspring_id=offspring_id,
            parent_ids=[parent1_id, parent2_id],
            genetic_operations=[GeneticOperation.CROSSOVER, GeneticOperation.MUTATION],
            trait_inheritance=self._analyze_trait_inheritance(parent1, parent2, offspring_profile),
            mutation_effects=mutations,
            predicted_fitness=predicted_fitness,
            breeding_success=True,
            novel_traits=novel_traits
        )
        
        self.breeding_history.append(breeding_result)
        return breeding_result
        
    def _assess_breeding_compatibility(self, parent1: GeneticProfile, 
                                     parent2: GeneticProfile) -> float:
        """Assess compatibility between two parents for breeding"""
        trait_compatibility = 0.0
        trait_count = 0
        
        # Check trait compatibility
        for trait1 in parent1.trait_genes:
            for trait2 in parent2.trait_genes:
                if trait1 == trait2:
                    # Same trait compatibility
                    trait_pair = (trait1, trait2)
                    if trait_pair in self.trait_compatibility_matrix:
                        trait_compatibility += self.trait_compatibility_matrix[trait_pair]
                    else:
                        trait_compatibility += 0.7  # Default compatibility
                    trait_count += 1
                    
        trait_compatibility = trait_compatibility / max(trait_count, 1)
        
        # Check behavioral compatibility
        behavior_similarity = 0.0
        behavior_count = 0
        
        for behavior in parent1.behavior_genes:
            if behavior in parent2.behavior_genes:
                diff = abs(parent1.behavior_genes[behavior] - parent2.behavior_genes[behavior])
                similarity = 1.0 - diff
                behavior_similarity += similarity
                behavior_count += 1
                
        behavior_similarity = behavior_similarity / max(behavior_count, 1)
        
        # Generation compatibility (prefer similar generations)
        generation_diff = abs(parent1.generation - parent2.generation)
        generation_compatibility = max(0.3, 1.0 - generation_diff / 10.0)
        
        overall_compatibility = (
            trait_compatibility * 0.5 +
            behavior_similarity * 0.3 +
            generation_compatibility * 0.2
        )
        
        return overall_compatibility
        
    def _perform_crossover(self, parent1: GeneticProfile, parent2: GeneticProfile,
                          strategy: str) -> Dict[str, Dict[str, float]]:
        """Perform genetic crossover between parents"""
        offspring_genes = {
            "trait_genes": {},
            "behavior_genes": {},
            "memory_genes": {},
            "meta_genes": {}
        }
        
        # Crossover trait genes
        for trait in parent1.trait_genes:
            if trait in parent2.trait_genes:
                if strategy == "balanced":
                    # Average of parents with slight random variation
                    avg_value = (parent1.trait_genes[trait] + parent2.trait_genes[trait]) / 2
                    variation = (random.random() - 0.5) * 0.1
                    offspring_genes["trait_genes"][trait] = np.clip(avg_value + variation, 0.0, 1.0)
                elif strategy == "dominant":
                    # Choose dominant trait (higher value)
                    if parent1.trait_genes[trait] > parent2.trait_genes[trait]:
                        offspring_genes["trait_genes"][trait] = parent1.trait_genes[trait]
                    else:
                        offspring_genes["trait_genes"][trait] = parent2.trait_genes[trait]
                elif strategy == "random":
                    # Random selection from parents
                    if random.random() < 0.5:
                        offspring_genes["trait_genes"][trait] = parent1.trait_genes[trait]
                    else:
                        offspring_genes["trait_genes"][trait] = parent2.trait_genes[trait]
                        
        # Crossover behavior genes
        for behavior in parent1.behavior_genes:
            if behavior in parent2.behavior_genes:
                # Use balanced strategy for behaviors
                avg_value = (parent1.behavior_genes[behavior] + parent2.behavior_genes[behavior]) / 2
                variation = (random.random() - 0.5) * 0.08
                offspring_genes["behavior_genes"][behavior] = np.clip(avg_value + variation, 0.0, 1.0)
                
        # Crossover memory genes
        for memory in parent1.memory_genes:
            if memory in parent2.memory_genes:
                # Memory genes favor higher values (better memory)
                max_value = max(parent1.memory_genes[memory], parent2.memory_genes[memory])
                min_value = min(parent1.memory_genes[memory], parent2.memory_genes[memory])
                offspring_genes["memory_genes"][memory] = random.uniform(min_value, max_value)
                
        # Crossover meta genes
        for meta in parent1.meta_genes:
            if meta in parent2.meta_genes:
                # Meta genes use weighted combination
                weight1 = random.uniform(0.3, 0.7)
                weight2 = 1.0 - weight1
                offspring_genes["meta_genes"][meta] = (
                    weight1 * parent1.meta_genes[meta] + 
                    weight2 * parent2.meta_genes[meta]
                )
                
        return offspring_genes
        
    def _apply_mutations(self, offspring_genes: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Apply random mutations to offspring genes"""
        mutations = []
        
        for gene_category, genes in offspring_genes.items():
            for gene_name, gene_value in genes.items():
                if random.random() < self.mutation_rate:
                    # Apply mutation
                    mutation_strength = random.uniform(-0.1, 0.1)
                    old_value = gene_value
                    new_value = np.clip(gene_value + mutation_strength, 0.0, 1.0)
                    genes[gene_name] = new_value
                    
                    mutations.append({
                        "gene_category": gene_category,
                        "gene_name": gene_name,
                        "old_value": old_value,
                        "new_value": new_value,
                        "mutation_strength": mutation_strength
                    })
                    
        return mutations
        
    def _predict_offspring_fitness(self, parent1: GeneticProfile, parent2: GeneticProfile,
                                 offspring: GeneticProfile) -> Dict[FitnessMetric, float]:
        """Predict fitness of offspring based on parent fitness and genes"""
        predicted_fitness = {}
        
        for metric in FitnessMetric:
            # Base prediction from parent fitness
            parent1_fitness = parent1.fitness_scores.get(metric, 0.5)
            parent2_fitness = parent2.fitness_scores.get(metric, 0.5)
            base_fitness = (parent1_fitness + parent2_fitness) / 2
            
            # Adjust based on relevant genes
            gene_bonus = 0.0
            
            if metric == FitnessMetric.REASONING_ACCURACY:
                gene_bonus = offspring.trait_genes.get("logic", 0.5) * 0.1
            elif metric == FitnessMetric.CREATIVITY_SCORE:
                gene_bonus = offspring.trait_genes.get("creativity", 0.5) * 0.1
            elif metric == FitnessMetric.USER_SATISFACTION:
                gene_bonus = offspring.trait_genes.get("empathy", 0.5) * 0.1
            elif metric == FitnessMetric.ADAPTABILITY:
                gene_bonus = offspring.trait_genes.get("adaptability", 0.5) * 0.1
            elif metric == FitnessMetric.EFFICIENCY:
                gene_bonus = offspring.behavior_genes.get("persistence", 0.5) * 0.1
                
            # Add random variation
            variation = (random.random() - 0.5) * 0.1
            
            predicted_value = np.clip(base_fitness + gene_bonus + variation, 0.0, 1.0)
            predicted_fitness[metric] = predicted_value
            
        return predicted_fitness
        
    def _analyze_trait_inheritance(self, parent1: GeneticProfile, parent2: GeneticProfile,
                                 offspring: GeneticProfile) -> Dict[str, Dict[str, Any]]:
        """Analyze how traits were inherited from parents"""
        inheritance_analysis = {}
        
        for trait in offspring.trait_genes:
            offspring_value = offspring.trait_genes[trait]
            parent1_value = parent1.trait_genes.get(trait, 0.5)
            parent2_value = parent2.trait_genes.get(trait, 0.5)
            
            # Determine dominant parent
            if abs(offspring_value - parent1_value) < abs(offspring_value - parent2_value):
                dominant_parent = "parent1"
                similarity = 1.0 - abs(offspring_value - parent1_value)
            else:
                dominant_parent = "parent2"
                similarity = 1.0 - abs(offspring_value - parent2_value)
                
            inheritance_analysis[trait] = {
                "offspring_value": offspring_value,
                "parent1_value": parent1_value,
                "parent2_value": parent2_value,
                "dominant_parent": dominant_parent,
                "similarity_to_dominant": similarity,
                "inheritance_type": "blended" if 0.3 < similarity < 0.9 else "direct"
            }
            
        return inheritance_analysis
        
    def _detect_novel_traits(self, offspring: GeneticProfile, 
                           parents: List[GeneticProfile]) -> List[str]:
        """Detect novel traits in offspring"""
        novel_traits = []
        
        for trait in offspring.trait_genes:
            offspring_value = offspring.trait_genes[trait]
            parent_values = [p.trait_genes.get(trait, 0.5) for p in parents]
            
            # Check if offspring value is outside parent range
            min_parent = min(parent_values)
            max_parent = max(parent_values)
            
            if offspring_value < min_parent - 0.1 or offspring_value > max_parent + 0.1:
                novel_traits.append(trait)
                
        return novel_traits
        
    def evolve_population(self, selection_pressure: float = 0.3,
                         breeding_rounds: int = 5) -> Dict[str, Any]:
        """Evolve entire population through multiple breeding rounds"""
        if len(self.genetic_profiles) < 4:
            return {"error": "Insufficient population for evolution"}
            
        evolution_stats = {
            "initial_population": len(self.genetic_profiles),
            "breeding_rounds": breeding_rounds,
            "successful_breedings": 0,
            "failed_breedings": 0,
            "novel_traits_discovered": set(),
            "average_fitness_improvement": 0.0
        }
        
        initial_fitness = self._calculate_population_fitness()
        
        for round_num in range(breeding_rounds):
            # Select breeding pairs based on fitness
            breeding_pairs = self._select_breeding_pairs(selection_pressure)
            
            for parent1_id, parent2_id in breeding_pairs:
                breeding_result = self.breed_entities(parent1_id, parent2_id)
                
                if breeding_result.breeding_success:
                    evolution_stats["successful_breedings"] += 1
                    evolution_stats["novel_traits_discovered"].update(breeding_result.novel_traits)
                else:
                    evolution_stats["failed_breedings"] += 1
                    
        # Calculate fitness improvement
        final_fitness = self._calculate_population_fitness()
        evolution_stats["average_fitness_improvement"] = final_fitness - initial_fitness
        evolution_stats["final_population"] = len(self.genetic_profiles)
        evolution_stats["novel_traits_discovered"] = list(evolution_stats["novel_traits_discovered"])
        
        return evolution_stats
        
    def _calculate_population_fitness(self) -> float:
        """Calculate average fitness of population"""
        if not self.genetic_profiles:
            return 0.0
            
        total_fitness = 0.0
        count = 0
        
        for profile in self.genetic_profiles.values():
            individual_fitness = 0.0
            for metric, score in profile.fitness_scores.items():
                weight = self.fitness_weights.get(metric, 0.1)
                individual_fitness += score * weight
                
            total_fitness += individual_fitness
            count += 1
            
        return total_fitness / count
        
    def _select_breeding_pairs(self, selection_pressure: float) -> List[Tuple[str, str]]:
        """Select breeding pairs based on fitness"""
        # Calculate fitness for all profiles
        fitness_scores = {}
        for profile_id, profile in self.genetic_profiles.items():
            fitness = 0.0
            for metric, score in profile.fitness_scores.items():
                weight = self.fitness_weights.get(metric, 0.1)
                fitness += score * weight
            fitness_scores[profile_id] = fitness
            
        # Sort by fitness
        sorted_profiles = sorted(fitness_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top performers for breeding
        top_count = max(4, int(len(sorted_profiles) * (1 - selection_pressure)))
        top_performers = [profile_id for profile_id, _ in sorted_profiles[:top_count]]
        
        # Create breeding pairs
        breeding_pairs = []
        for i in range(0, len(top_performers) - 1, 2):
            if i + 1 < len(top_performers):
                breeding_pairs.append((top_performers[i], top_performers[i + 1]))
                
        return breeding_pairs
        
    def get_population_statistics(self) -> Dict[str, Any]:
        """Get comprehensive population statistics"""
        if not self.genetic_profiles:
            return {"error": "No population data"}
            
        # Generation distribution
        generations = [profile.generation for profile in self.genetic_profiles.values()]
        
        # Trait statistics
        trait_averages = {}
        for trait in PersonalityTrait:
            trait_values = [
                profile.trait_genes.get(trait.value, 0.5) 
                for profile in self.genetic_profiles.values()
            ]
            trait_averages[trait.value] = {
                "average": np.mean(trait_values),
                "std": np.std(trait_values),
                "min": min(trait_values),
                "max": max(trait_values)
            }
            
        # Fitness statistics
        fitness_stats = {}
        for metric in FitnessMetric:
            fitness_values = [
                profile.fitness_scores.get(metric, 0.5)
                for profile in self.genetic_profiles.values()
            ]
            fitness_stats[metric.value] = {
                "average": np.mean(fitness_values),
                "std": np.std(fitness_values),
                "min": min(fitness_values),
                "max": max(fitness_values)
            }
            
        return {
            "population_size": len(self.genetic_profiles),
            "generation_stats": {
                "min_generation": min(generations),
                "max_generation": max(generations),
                "average_generation": np.mean(generations)
            },
            "trait_statistics": trait_averages,
            "fitness_statistics": fitness_stats,
            "breeding_history_length": len(self.breeding_history),
            "total_mutations": sum(len(p.mutations) for p in self.genetic_profiles.values())
        }