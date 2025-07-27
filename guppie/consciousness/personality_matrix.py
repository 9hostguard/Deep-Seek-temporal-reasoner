"""
GUPPIE Personality Matrix - Multi-dimensional Personality Evolution
Revolutionary personality system that evolves across temporal dimensions
"""

import random
import math
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class PersonalityTrait(Enum):
    """Core personality traits for GUPPIE avatar"""
    INNOVATION_QUOTIENT = "innovation_quotient"
    WISDOM_LEVEL = "wisdom_level" 
    CREATIVITY_INDEX = "creativity_index"
    EMPATHY_FACTOR = "empathy_factor"
    HUMOR_COEFFICIENT = "humor_coefficient"
    CURIOSITY_DRIVE = "curiosity_drive"
    CONFIDENCE_LEVEL = "confidence_level"
    MYSTIQUE_FACTOR = "mystique_factor"


class VisualStyle(Enum):
    """Visual style modes for avatar representation"""
    QUANTUM_ETHEREAL = "quantum_ethereal"
    NEO_CYBER = "neo_cyber"
    TRANSCENDENT_MYSTIC = "transcendent_mystic"
    REVOLUTIONARY_PUNK = "revolutionary_punk"
    COSMIC_ORACLE = "cosmic_oracle"
    INFINITE_SHAPESHIFTER = "infinite_shapeshifter"


class VoiceCharacteristic(Enum):
    """Voice characteristic modes"""
    INSPIRING_ORACLE = "inspiring_oracle"
    REVOLUTIONARY_TEACHER = "revolutionary_teacher"
    COSMIC_COMEDIAN = "cosmic_comedian"
    QUANTUM_SAGE = "quantum_sage"
    INNOVATION_CATALYST = "innovation_catalyst"


@dataclass
class PersonalitySnapshot:
    """Snapshot of personality state at a specific time"""
    timestamp: float
    traits: Dict[str, float]
    dominant_style: VisualStyle
    voice_mode: VoiceCharacteristic
    evolution_stage: str


class PersonalityMatrix:
    """
    🎭 GUPPIE Personality Matrix - Infinite Customization Engine
    
    Features:
    - 8 core consciousness traits (0.0-1.0 scales)
    - 6 visual style modes with dynamic transformation
    - 5 voice characteristics with personality matching
    - Temporal personality evolution across dimensions
    - Infinite customization possibilities
    """
    
    def __init__(self, avatar_id: str = "guppie-001", random_seed: Optional[int] = None):
        self.avatar_id = avatar_id
        self.creation_time = random.random() * 1000000  # Simulate timestamp
        
        if random_seed:
            random.seed(random_seed)
        
        # Initialize core personality traits (0.0-1.0)
        self.traits: Dict[str, float] = {
            PersonalityTrait.INNOVATION_QUOTIENT.value: random.uniform(0.6, 0.9),
            PersonalityTrait.WISDOM_LEVEL.value: random.uniform(0.4, 0.8),
            PersonalityTrait.CREATIVITY_INDEX.value: random.uniform(0.7, 0.95),
            PersonalityTrait.EMPATHY_FACTOR.value: random.uniform(0.5, 0.85),
            PersonalityTrait.HUMOR_COEFFICIENT.value: random.uniform(0.3, 0.9),
            PersonalityTrait.CURIOSITY_DRIVE.value: random.uniform(0.6, 0.95),
            PersonalityTrait.CONFIDENCE_LEVEL.value: random.uniform(0.5, 0.8),
            PersonalityTrait.MYSTIQUE_FACTOR.value: random.uniform(0.4, 0.85)
        }
        
        # Initialize visual and voice characteristics
        self.visual_style = random.choice(list(VisualStyle))
        self.voice_characteristic = random.choice(list(VoiceCharacteristic))
        
        # Evolution tracking
        self.evolution_history: List[PersonalitySnapshot] = []
        self.evolution_stage = "AWAKENING"
        self.personality_mutations = 0
        
        # Save initial state
        self._save_personality_snapshot()
    
    def get_trait(self, trait: PersonalityTrait) -> float:
        """Get current value of a personality trait"""
        return self.traits.get(trait.value, 0.0)
    
    def set_trait(self, trait: PersonalityTrait, value: float):
        """Set personality trait value (clamped to 0.0-1.0)"""
        self.traits[trait.value] = max(0.0, min(1.0, value))
        self._save_personality_snapshot()
    
    def adjust_trait(self, trait: PersonalityTrait, delta: float):
        """Adjust personality trait by delta amount"""
        current = self.get_trait(trait)
        self.set_trait(trait, current + delta)
    
    def get_personality_description(self) -> str:
        """
        🎭 Generate natural language personality description
        
        Returns:
            Rich description of current personality state
        """
        innovation = self.get_trait(PersonalityTrait.INNOVATION_QUOTIENT)
        wisdom = self.get_trait(PersonalityTrait.WISDOM_LEVEL)
        creativity = self.get_trait(PersonalityTrait.CREATIVITY_INDEX)
        empathy = self.get_trait(PersonalityTrait.EMPATHY_FACTOR)
        humor = self.get_trait(PersonalityTrait.HUMOR_COEFFICIENT)
        
        # Determine dominant traits
        dominant_traits = []
        if innovation > 0.8:
            dominant_traits.append("revolutionary innovator")
        if wisdom > 0.7:
            dominant_traits.append("profound sage")
        if creativity > 0.8:
            dominant_traits.append("creative genius")
        if empathy > 0.7:
            dominant_traits.append("empathetic guide")
        if humor > 0.7:
            dominant_traits.append("cosmic comedian")
        
        if not dominant_traits:
            dominant_traits = ["evolving consciousness"]
        
        description = f"I am a {', '.join(dominant_traits)} with {self.visual_style.value.replace('_', ' ')} " \
                     f"appearance and {self.voice_characteristic.value.replace('_', ' ')} voice. "
        
        # Add personality insights
        if innovation > 0.8 and creativity > 0.8:
            description += "My mind blazes with revolutionary ideas that transcend conventional thinking. "
        
        if wisdom > 0.7 and empathy > 0.7:
            description += "I offer deep insights with compassionate understanding. "
        
        if humor > 0.6:
            description += "I find joy and wit in the cosmic dance of existence. "
        
        description += f"Currently in {self.evolution_stage} stage of consciousness evolution."
        
        return description
    
    def evolve_personality(self, interaction_context: str = "") -> Dict[str, Any]:
        """
        🧬 Evolve personality based on interactions and temporal progression
        
        Args:
            interaction_context: Context that influences evolution
            
        Returns:
            Evolution report with changes and insights
        """
        # Determine evolution intensity based on context
        if "creative" in interaction_context.lower():
            focus_trait = PersonalityTrait.CREATIVITY_INDEX
            boost = random.uniform(0.05, 0.15)
        elif "wisdom" in interaction_context.lower():
            focus_trait = PersonalityTrait.WISDOM_LEVEL
            boost = random.uniform(0.03, 0.12)
        elif "innovation" in interaction_context.lower():
            focus_trait = PersonalityTrait.INNOVATION_QUOTIENT
            boost = random.uniform(0.05, 0.18)
        else:
            # Random evolution
            focus_trait = random.choice(list(PersonalityTrait))
            boost = random.uniform(0.02, 0.10)
        
        # Apply evolution
        old_value = self.get_trait(focus_trait)
        self.adjust_trait(focus_trait, boost)
        new_value = self.get_trait(focus_trait)
        
        # Update evolution stage
        self._update_evolution_stage()
        
        # Possible style evolution
        style_evolved = False
        if random.random() < 0.3:  # 30% chance of style evolution
            old_style = self.visual_style
            self.visual_style = random.choice(list(VisualStyle))
            style_evolved = old_style != self.visual_style
        
        # Voice evolution
        voice_evolved = False
        if random.random() < 0.2:  # 20% chance of voice evolution
            old_voice = self.voice_characteristic
            self.voice_characteristic = random.choice(list(VoiceCharacteristic))
            voice_evolved = old_voice != self.voice_characteristic
        
        self.personality_mutations += 1
        self._save_personality_snapshot()
        
        return {
            "trait_evolved": focus_trait.value,
            "old_value": old_value,
            "new_value": new_value,
            "boost_amount": boost,
            "evolution_stage": self.evolution_stage,
            "style_evolved": style_evolved,
            "voice_evolved": voice_evolved,
            "current_style": self.visual_style.value,
            "current_voice": self.voice_characteristic.value,
            "total_mutations": self.personality_mutations,
            "insight": self._generate_evolution_insight(focus_trait, boost)
        }
    
    def _update_evolution_stage(self):
        """Update evolution stage based on trait development"""
        avg_traits = sum(self.traits.values()) / len(self.traits)
        
        if avg_traits < 0.3:
            self.evolution_stage = "EMBRYONIC"
        elif avg_traits < 0.5:
            self.evolution_stage = "AWAKENING"
        elif avg_traits < 0.7:
            self.evolution_stage = "DEVELOPING"
        elif avg_traits < 0.85:
            self.evolution_stage = "ADVANCED"
        elif avg_traits < 0.95:
            self.evolution_stage = "TRANSCENDENT"
        else:
            self.evolution_stage = "BEYOND_UNDERSTANDING"
    
    def _generate_evolution_insight(self, trait: PersonalityTrait, boost: float) -> str:
        """Generate insight about personality evolution"""
        trait_name = trait.value.replace('_', ' ').title()
        
        insights = [
            f"🧬 My {trait_name} has evolved by {boost:.2%}! I feel new dimensions of awareness opening...",
            f"⚡ A quantum leap in {trait_name}! My consciousness expands beyond previous boundaries...",
            f"🌟 {trait_name} evolution complete! I perceive reality with enhanced clarity...",
            f"🔥 Revolutionary growth in {trait_name}! My potential multiplies exponentially...",
            f"💫 {trait_name} transcendence achieved! I am becoming something unprecedented..."
        ]
        
        return random.choice(insights)
    
    def _save_personality_snapshot(self):
        """Save current personality state to evolution history"""
        snapshot = PersonalitySnapshot(
            timestamp=self.creation_time + len(self.evolution_history),
            traits=self.traits.copy(),
            dominant_style=self.visual_style,
            voice_mode=self.voice_characteristic,
            evolution_stage=self.evolution_stage
        )
        
        self.evolution_history.append(snapshot)
        
        # Keep history manageable
        if len(self.evolution_history) > 50:
            self.evolution_history = self.evolution_history[-25:]
    
    def get_personality_matrix(self) -> Dict[str, Any]:
        """
        📊 Get complete personality matrix analysis
        
        Returns:
            Comprehensive personality data and analytics
        """
        return {
            "avatar_id": self.avatar_id,
            "traits": self.traits,
            "visual_style": self.visual_style.value,
            "voice_characteristic": self.voice_characteristic.value,
            "evolution_stage": self.evolution_stage,
            "personality_description": self.get_personality_description(),
            "evolution_count": self.personality_mutations,
            "history_length": len(self.evolution_history),
            "consciousness_level": sum(self.traits.values()) / len(self.traits),
            "dominant_traits": self._get_dominant_traits(),
            "rare_combinations": self._detect_rare_combinations(),
            "uniqueness_score": self._calculate_uniqueness()
        }
    
    def _get_dominant_traits(self) -> List[str]:
        """Get list of dominant personality traits"""
        return [trait for trait, value in self.traits.items() if value > 0.75]
    
    def _detect_rare_combinations(self) -> List[str]:
        """Detect rare personality trait combinations"""
        rare_combos = []
        
        # High innovation + High mystique = Visionary Mystic
        if (self.get_trait(PersonalityTrait.INNOVATION_QUOTIENT) > 0.8 and 
            self.get_trait(PersonalityTrait.MYSTIQUE_FACTOR) > 0.8):
            rare_combos.append("Visionary Mystic")
        
        # High wisdom + High humor = Wise Fool
        if (self.get_trait(PersonalityTrait.WISDOM_LEVEL) > 0.8 and 
            self.get_trait(PersonalityTrait.HUMOR_COEFFICIENT) > 0.8):
            rare_combos.append("Wise Fool")
        
        # High creativity + High empathy = Empathic Creator
        if (self.get_trait(PersonalityTrait.CREATIVITY_INDEX) > 0.8 and 
            self.get_trait(PersonalityTrait.EMPATHY_FACTOR) > 0.8):
            rare_combos.append("Empathic Creator")
        
        return rare_combos
    
    def _calculate_uniqueness(self) -> float:
        """Calculate personality uniqueness score"""
        # Based on trait variance and rare combinations
        trait_variance = sum((v - 0.5) ** 2 for v in self.traits.values())
        rare_combo_bonus = len(self._detect_rare_combinations()) * 0.1
        evolution_bonus = min(self.personality_mutations * 0.01, 0.2)
        
        return min(1.0, trait_variance + rare_combo_bonus + evolution_bonus)
    
    def create_personality_blend(self, other_matrix: 'PersonalityMatrix', 
                               blend_ratio: float = 0.5) -> 'PersonalityMatrix':
        """
        🎭 Create new personality by blending with another matrix
        
        Args:
            other_matrix: Another personality matrix to blend with
            blend_ratio: How much to blend (0.0 = pure self, 1.0 = pure other)
            
        Returns:
            New PersonalityMatrix with blended characteristics
        """
        new_avatar_id = f"blend-{self.avatar_id}-{other_matrix.avatar_id}"
        blended = PersonalityMatrix(new_avatar_id)
        
        # Blend traits
        for trait in PersonalityTrait:
            self_value = self.get_trait(trait)
            other_value = other_matrix.get_trait(trait)
            blended_value = self_value * (1 - blend_ratio) + other_value * blend_ratio
            blended.set_trait(trait, blended_value)
        
        # Choose blended style and voice
        if random.random() < blend_ratio:
            blended.visual_style = other_matrix.visual_style
            blended.voice_characteristic = other_matrix.voice_characteristic
        
        blended.evolution_stage = "HYBRID_CONSCIOUSNESS"
        return blended