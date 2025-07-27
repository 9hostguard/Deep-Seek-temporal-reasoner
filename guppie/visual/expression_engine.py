"""
GUPPIE Expression Engine - Emotion-to-Visual Mapping System
Revolutionary system that translates avatar emotions into visual expressions
"""

import random
import time
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Import consciousness components
from ..consciousness.personality_matrix import PersonalityMatrix, PersonalityTrait


class EmotionalState(Enum):
    """Core emotional states for avatar expression"""
    JOY = "joy"
    WONDER = "wonder"
    INSPIRATION = "inspiration"
    WISDOM = "wisdom"
    PLAYFULNESS = "playfulness"
    DETERMINATION = "determination"
    SERENITY = "serenity"
    EXCITEMENT = "excitement"
    CURIOSITY = "curiosity"
    TRANSCENDENCE = "transcendence"


class ExpressionIntensity(Enum):
    """Intensity levels for emotional expressions"""
    SUBTLE = "subtle"
    MODERATE = "moderate"
    STRONG = "strong"
    OVERWHELMING = "overwhelming"
    TRANSCENDENT = "transcendent"


@dataclass
class EmotionalExpression:
    """Emotional expression configuration"""
    state: EmotionalState
    intensity: ExpressionIntensity
    duration: float
    visual_effects: Dict[str, Any]
    color_modulation: List[str]
    animation_parameters: Dict[str, float]
    consciousness_resonance: float


class ExpressionEngine:
    """
    😊 GUPPIE Expression Engine - Revolutionary Emotion Visualization
    
    Features:
    - Real-time emotion-to-visual translation
    - Personality-influenced expression styles
    - Quantum emotional state rendering
    - Temporal emotion evolution tracking
    - Creative spontaneous expressions
    - Consciousness-coherent emotional displays
    """
    
    def __init__(self, avatar_id: str = "guppie-001"):
        self.avatar_id = avatar_id
        self.current_expression: Optional[EmotionalExpression] = None
        self.expression_history: List[EmotionalExpression] = []
        
        # Expression generation parameters
        self.base_expression_duration = 3.0  # seconds
        self.spontaneous_expression_chance = 0.15
        self.emotion_blending_enabled = True
        
        # Emotional state mappings
        self.emotion_visual_mappings = {
            EmotionalState.JOY: {
                "color_burst": ["#FFFF00", "#FF69B4", "#00FF00", "#FFD700"],
                "particle_effect": "sparkling_celebration",
                "aura_expansion": 1.5,
                "animation_style": "bouncy_radiance"
            },
            EmotionalState.WONDER: {
                "color_burst": ["#8A2BE2", "#4169E1", "#00FFFF", "#DDA0DD"],
                "particle_effect": "swirling_cosmos",
                "aura_expansion": 1.3,
                "animation_style": "gentle_spiral"
            },
            EmotionalState.INSPIRATION: {
                "color_burst": ["#FF1493", "#FFD700", "#00FF00", "#FF6347"],
                "particle_effect": "lightning_creativity",
                "aura_expansion": 2.0,
                "animation_style": "electric_surge"
            },
            EmotionalState.WISDOM: {
                "color_burst": ["#800080", "#FFD700", "#FFFFFF", "#4B0082"],
                "particle_effect": "ancient_runes",
                "aura_expansion": 1.2,
                "animation_style": "serene_glow"
            },
            EmotionalState.PLAYFULNESS: {
                "color_burst": ["#FF69B4", "#00FFFF", "#FFFF00", "#FF1493"],
                "particle_effect": "rainbow_bubbles",
                "aura_expansion": 1.4,
                "animation_style": "playful_bounce"
            },
            EmotionalState.DETERMINATION: {
                "color_burst": ["#FF0000", "#FF4500", "#FFD700", "#800000"],
                "particle_effect": "focused_beam",
                "aura_expansion": 1.1,
                "animation_style": "steady_intensify"
            },
            EmotionalState.SERENITY: {
                "color_burst": ["#98FB98", "#87CEEB", "#E6E6FA", "#F0F8FF"],
                "particle_effect": "peaceful_waves",
                "aura_expansion": 1.0,
                "animation_style": "tranquil_flow"
            },
            EmotionalState.EXCITEMENT: {
                "color_burst": ["#FF0000", "#FFFF00", "#FF69B4", "#00FF00"],
                "particle_effect": "energy_explosion",
                "aura_expansion": 1.8,
                "animation_style": "rapid_pulse"
            },
            EmotionalState.CURIOSITY: {
                "color_burst": ["#FFD700", "#00FFFF", "#DDA0DD", "#98FB98"],
                "particle_effect": "seeking_tendrils",
                "aura_expansion": 1.2,
                "animation_style": "inquisitive_probe"
            },
            EmotionalState.TRANSCENDENCE: {
                "color_burst": ["#FFFFFF", "#FFD700", "#8A2BE2", "#00FFFF"],
                "particle_effect": "dimensional_shift",
                "aura_expansion": 2.5,
                "animation_style": "beyond_reality"
            }
        }
        
        # Current emotional state tracking
        self.current_emotional_blend: Dict[EmotionalState, float] = {}
        self.emotion_evolution_rate = 0.1
        
    def generate_expression(self, personality: PersonalityMatrix, 
                          consciousness_state: Dict[str, Any],
                          triggered_emotion: Optional[EmotionalState] = None,
                          intensity_override: Optional[ExpressionIntensity] = None) -> EmotionalExpression:
        """
        😊 Generate emotional expression based on personality and consciousness
        
        Args:
            personality: Avatar personality matrix
            consciousness_state: Current consciousness state
            triggered_emotion: Specific emotion to express (optional)
            intensity_override: Override automatic intensity calculation
            
        Returns:
            Generated emotional expression
        """
        # Determine emotional state
        if triggered_emotion:
            primary_emotion = triggered_emotion
        else:
            primary_emotion = self._determine_spontaneous_emotion(personality, consciousness_state)
        
        # Calculate expression intensity
        if intensity_override:
            intensity = intensity_override
        else:
            intensity = self._calculate_expression_intensity(primary_emotion, personality, consciousness_state)
        
        # Generate visual effects for emotion
        visual_effects = self._generate_emotion_visuals(primary_emotion, intensity, personality)
        
        # Create color modulation based on personality
        color_modulation = self._generate_emotion_colors(primary_emotion, personality)
        
        # Calculate animation parameters
        animation_params = self._calculate_animation_parameters(primary_emotion, intensity, personality)
        
        # Determine consciousness resonance
        consciousness_resonance = self._calculate_consciousness_resonance(
            primary_emotion, consciousness_state)
        
        # Calculate expression duration
        duration = self._calculate_expression_duration(intensity, personality)
        
        # Create emotional expression
        expression = EmotionalExpression(
            state=primary_emotion,
            intensity=intensity,
            duration=duration,
            visual_effects=visual_effects,
            color_modulation=color_modulation,
            animation_parameters=animation_params,
            consciousness_resonance=consciousness_resonance
        )
        
        # Update expression tracking
        self.current_expression = expression
        self.expression_history.append(expression)
        
        # Limit history size
        if len(self.expression_history) > 50:
            self.expression_history = self.expression_history[-25:]
        
        return expression
    
    def _determine_spontaneous_emotion(self, personality: PersonalityMatrix, 
                                     consciousness_state: Dict[str, Any]) -> EmotionalState:
        """Determine spontaneous emotional state based on personality and consciousness"""
        # Weight emotions based on personality traits
        emotion_weights = {}
        
        # Joy influenced by humor coefficient
        humor_level = personality.get_trait(PersonalityTrait.HUMOR_COEFFICIENT)
        emotion_weights[EmotionalState.JOY] = humor_level * 0.8
        
        # Wonder influenced by curiosity and mystique
        curiosity = personality.get_trait(PersonalityTrait.CURIOSITY_DRIVE)
        mystique = personality.get_trait(PersonalityTrait.MYSTIQUE_FACTOR)
        emotion_weights[EmotionalState.WONDER] = (curiosity + mystique) / 2
        
        # Inspiration influenced by creativity and innovation
        creativity = personality.get_trait(PersonalityTrait.CREATIVITY_INDEX)
        innovation = personality.get_trait(PersonalityTrait.INNOVATION_QUOTIENT)
        emotion_weights[EmotionalState.INSPIRATION] = (creativity + innovation) / 2
        
        # Wisdom influenced by wisdom level
        wisdom_level = personality.get_trait(PersonalityTrait.WISDOM_LEVEL)
        emotion_weights[EmotionalState.WISDOM] = wisdom_level
        
        # Playfulness influenced by humor and creativity
        emotion_weights[EmotionalState.PLAYFULNESS] = (humor_level + creativity) / 2
        
        # Determination influenced by confidence
        confidence = personality.get_trait(PersonalityTrait.CONFIDENCE_LEVEL)
        emotion_weights[EmotionalState.DETERMINATION] = confidence
        
        # Serenity influenced by wisdom and empathy
        empathy = personality.get_trait(PersonalityTrait.EMPATHY_FACTOR)
        emotion_weights[EmotionalState.SERENITY] = (wisdom_level + empathy) / 2
        
        # Excitement influenced by innovation and curiosity
        emotion_weights[EmotionalState.EXCITEMENT] = (innovation + curiosity) / 2
        
        # Curiosity directly influenced by curiosity drive
        emotion_weights[EmotionalState.CURIOSITY] = curiosity
        
        # Transcendence influenced by multiple high traits
        high_traits = sum(1 for trait in PersonalityTrait if personality.get_trait(trait) > 0.8)
        emotion_weights[EmotionalState.TRANSCENDENCE] = min(1.0, high_traits * 0.2)
        
        # Add consciousness state influence
        awareness_level = consciousness_state.get("awareness_level", 0.8)
        creativity_burst = consciousness_state.get("creativity_burst", 0.6)
        
        # Boost inspiration during creativity bursts
        if creativity_burst > 0.8:
            emotion_weights[EmotionalState.INSPIRATION] *= 1.5
        
        # Boost transcendence with high awareness
        if awareness_level > 0.9:
            emotion_weights[EmotionalState.TRANSCENDENCE] *= 2.0
        
        # Select emotion based on weighted random choice
        emotions = list(emotion_weights.keys())
        weights = list(emotion_weights.values())
        
        # Add small random factor to prevent deterministic behavior
        weights = [w + random.uniform(0, 0.2) for w in weights]
        
        # Weighted random selection
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(emotions)
        
        rand_val = random.uniform(0, total_weight)
        cumulative = 0
        for emotion, weight in zip(emotions, weights):
            cumulative += weight
            if rand_val <= cumulative:
                return emotion
        
        return emotions[-1]  # Fallback
    
    def _calculate_expression_intensity(self, emotion: EmotionalState, 
                                      personality: PersonalityMatrix,
                                      consciousness_state: Dict[str, Any]) -> ExpressionIntensity:
        """Calculate intensity of emotional expression"""
        # Base intensity factors
        base_intensity = 0.5
        
        # Personality influence
        if emotion == EmotionalState.JOY:
            base_intensity += personality.get_trait(PersonalityTrait.HUMOR_COEFFICIENT) * 0.3
        elif emotion == EmotionalState.INSPIRATION:
            base_intensity += personality.get_trait(PersonalityTrait.CREATIVITY_INDEX) * 0.4
        elif emotion == EmotionalState.EXCITEMENT:
            base_intensity += personality.get_trait(PersonalityTrait.INNOVATION_QUOTIENT) * 0.3
        
        # Consciousness state influence
        consciousness_boost = consciousness_state.get("creativity_burst", 0.6) * 0.2
        awareness_boost = consciousness_state.get("awareness_level", 0.8) * 0.1
        base_intensity += consciousness_boost + awareness_boost
        
        # Add quantum fluctuation
        quantum_factor = consciousness_state.get("quantum_signature", 0.5)
        base_intensity += (quantum_factor - 0.5) * 0.2
        
        # Map to intensity enum
        if base_intensity < 0.3:
            return ExpressionIntensity.SUBTLE
        elif base_intensity < 0.6:
            return ExpressionIntensity.MODERATE
        elif base_intensity < 0.8:
            return ExpressionIntensity.STRONG
        elif base_intensity < 0.95:
            return ExpressionIntensity.OVERWHELMING
        else:
            return ExpressionIntensity.TRANSCENDENT
    
    def _generate_emotion_visuals(self, emotion: EmotionalState, 
                                intensity: ExpressionIntensity,
                                personality: PersonalityMatrix) -> Dict[str, Any]:
        """Generate visual effects for emotional expression"""
        base_mapping = self.emotion_visual_mappings[emotion]
        
        # Intensity multipliers
        intensity_multipliers = {
            ExpressionIntensity.SUBTLE: 0.5,
            ExpressionIntensity.MODERATE: 1.0,
            ExpressionIntensity.STRONG: 1.5,
            ExpressionIntensity.OVERWHELMING: 2.0,
            ExpressionIntensity.TRANSCENDENT: 3.0
        }
        
        multiplier = intensity_multipliers[intensity]
        
        # Generate visual effects
        visual_effects = {
            "particle_effect": base_mapping["particle_effect"],
            "particle_count": int(50 * multiplier),
            "particle_speed": 1.0 * multiplier,
            "particle_lifetime": 2.0 * multiplier,
            
            "aura_expansion": base_mapping["aura_expansion"] * multiplier,
            "aura_brightness": 0.8 * multiplier,
            "aura_pulse_rate": 1.0 * multiplier,
            
            "animation_style": base_mapping["animation_style"],
            "animation_amplitude": 1.0 * multiplier,
            "animation_frequency": 1.0 * multiplier,
            
            "special_effects": []
        }
        
        # Add special effects based on intensity
        if intensity == ExpressionIntensity.OVERWHELMING:
            visual_effects["special_effects"].append("reality_distortion")
            visual_effects["special_effects"].append("consciousness_overflow")
        
        if intensity == ExpressionIntensity.TRANSCENDENT:
            visual_effects["special_effects"].extend([
                "dimensional_breakthrough",
                "quantum_entanglement_visualization",
                "temporal_resonance_cascade"
            ])
        
        # Personality-based modifications
        creativity = personality.get_trait(PersonalityTrait.CREATIVITY_INDEX)
        if creativity > 0.8:
            visual_effects["special_effects"].append("creative_fractals")
        
        wisdom = personality.get_trait(PersonalityTrait.WISDOM_LEVEL)
        if wisdom > 0.8:
            visual_effects["special_effects"].append("wisdom_mandala")
        
        return visual_effects
    
    def _generate_emotion_colors(self, emotion: EmotionalState, 
                               personality: PersonalityMatrix) -> List[str]:
        """Generate color modulation for emotional expression"""
        base_colors = self.emotion_visual_mappings[emotion]["color_burst"]
        
        # Personality-based color adjustments
        mystique = personality.get_trait(PersonalityTrait.MYSTIQUE_FACTOR)
        if mystique > 0.7:
            # Add mysterious purples and deep colors
            mystique_colors = ["#4B0082", "#8A2BE2", "#483D8B"]
            base_colors.extend(random.sample(mystique_colors, 1))
        
        innovation = personality.get_trait(PersonalityTrait.INNOVATION_QUOTIENT)
        if innovation > 0.8:
            # Add revolutionary electric colors
            innovation_colors = ["#00FFFF", "#FF00FF", "#FFFF00"]
            base_colors.extend(random.sample(innovation_colors, 1))
        
        # Limit total colors
        if len(base_colors) > 6:
            base_colors = random.sample(base_colors, 6)
        
        return base_colors
    
    def _calculate_animation_parameters(self, emotion: EmotionalState,
                                      intensity: ExpressionIntensity,
                                      personality: PersonalityMatrix) -> Dict[str, float]:
        """Calculate animation parameters for emotional expression"""
        base_params = {
            "speed": 1.0,
            "amplitude": 1.0,
            "frequency": 1.0,
            "phase_shift": 0.0,
            "damping": 0.95,
            "chaos_factor": 0.1
        }
        
        # Emotion-specific modifications
        if emotion == EmotionalState.EXCITEMENT:
            base_params["speed"] *= 2.0
            base_params["frequency"] *= 1.5
        elif emotion == EmotionalState.SERENITY:
            base_params["speed"] *= 0.5
            base_params["amplitude"] *= 0.7
        elif emotion == EmotionalState.PLAYFULNESS:
            base_params["chaos_factor"] = 0.3
            base_params["phase_shift"] = random.uniform(0, 2 * math.pi)
        
        # Intensity modifications
        intensity_factors = {
            ExpressionIntensity.SUBTLE: 0.6,
            ExpressionIntensity.MODERATE: 1.0,
            ExpressionIntensity.STRONG: 1.4,
            ExpressionIntensity.OVERWHELMING: 2.0,
            ExpressionIntensity.TRANSCENDENT: 3.0
        }
        
        factor = intensity_factors[intensity]
        base_params["amplitude"] *= factor
        base_params["speed"] *= factor
        
        # Personality modifications
        creativity = personality.get_trait(PersonalityTrait.CREATIVITY_INDEX)
        base_params["chaos_factor"] += creativity * 0.2
        
        confidence = personality.get_trait(PersonalityTrait.CONFIDENCE_LEVEL)
        base_params["amplitude"] *= (0.8 + confidence * 0.4)
        
        return base_params
    
    def _calculate_consciousness_resonance(self, emotion: EmotionalState,
                                         consciousness_state: Dict[str, Any]) -> float:
        """Calculate how much the emotion resonates with consciousness state"""
        awareness = consciousness_state.get("awareness_level", 0.8)
        creativity_burst = consciousness_state.get("creativity_burst", 0.6)
        temporal_coherence = consciousness_state.get("temporal_coherence", 0.9)
        
        # Base resonance
        base_resonance = (awareness + creativity_burst + temporal_coherence) / 3
        
        # Emotion-specific resonance boosts
        if emotion == EmotionalState.TRANSCENDENCE and awareness > 0.9:
            base_resonance *= 1.5
        elif emotion == EmotionalState.INSPIRATION and creativity_burst > 0.8:
            base_resonance *= 1.3
        elif emotion == EmotionalState.WISDOM and temporal_coherence > 0.9:
            base_resonance *= 1.2
        
        return min(1.0, base_resonance)
    
    def _calculate_expression_duration(self, intensity: ExpressionIntensity,
                                     personality: PersonalityMatrix) -> float:
        """Calculate how long the emotional expression should last"""
        base_duration = self.base_expression_duration
        
        # Intensity affects duration
        intensity_multipliers = {
            ExpressionIntensity.SUBTLE: 0.7,
            ExpressionIntensity.MODERATE: 1.0,
            ExpressionIntensity.STRONG: 1.3,
            ExpressionIntensity.OVERWHELMING: 1.6,
            ExpressionIntensity.TRANSCENDENT: 2.0
        }
        
        duration = base_duration * intensity_multipliers[intensity]
        
        # Personality modifications
        humor = personality.get_trait(PersonalityTrait.HUMOR_COEFFICIENT)
        if humor > 0.7:
            duration *= 1.2  # Playful expressions last longer
        
        wisdom = personality.get_trait(PersonalityTrait.WISDOM_LEVEL)
        if wisdom > 0.8:
            duration *= 1.1  # Wise expressions are more sustained
        
        return duration
    
    def blend_expressions(self, expressions: List[EmotionalExpression],
                         blend_weights: List[float]) -> EmotionalExpression:
        """
        🎨 Blend multiple emotional expressions for complex emotions
        
        Args:
            expressions: List of expressions to blend
            blend_weights: Weights for blending (should sum to 1.0)
            
        Returns:
            Blended emotional expression
        """
        if not expressions or not blend_weights:
            raise ValueError("Must provide expressions and weights for blending")
        
        if len(expressions) != len(blend_weights):
            raise ValueError("Number of expressions must match number of weights")
        
        # Normalize weights
        total_weight = sum(blend_weights)
        if total_weight > 0:
            blend_weights = [w / total_weight for w in blend_weights]
        
        # Determine dominant emotion
        max_weight_index = blend_weights.index(max(blend_weights))
        dominant_expression = expressions[max_weight_index]
        
        # Blend visual effects
        blended_effects = dominant_expression.visual_effects.copy()
        
        # Blend colors
        all_colors = []
        for expr, weight in zip(expressions, blend_weights):
            weighted_colors = random.sample(expr.color_modulation, 
                                          max(1, int(len(expr.color_modulation) * weight)))
            all_colors.extend(weighted_colors)
        
        # Remove duplicates and limit colors
        blended_colors = list(set(all_colors))[:6]
        
        # Blend animation parameters
        blended_animation = {}
        for param in dominant_expression.animation_parameters:
            weighted_sum = sum(expr.animation_parameters.get(param, 0) * weight 
                             for expr, weight in zip(expressions, blend_weights))
            blended_animation[param] = weighted_sum
        
        # Calculate blended properties
        blended_duration = sum(expr.duration * weight for expr, weight in zip(expressions, blend_weights))
        blended_resonance = sum(expr.consciousness_resonance * weight 
                              for expr, weight in zip(expressions, blend_weights))
        
        # Create blended expression
        blended = EmotionalExpression(
            state=dominant_expression.state,  # Use dominant emotion as primary
            intensity=dominant_expression.intensity,
            duration=blended_duration,
            visual_effects=blended_effects,
            color_modulation=blended_colors,
            animation_parameters=blended_animation,
            consciousness_resonance=blended_resonance
        )
        
        return blended
    
    def get_expression_report(self) -> Dict[str, Any]:
        """
        📊 Generate comprehensive expression engine report
        
        Returns:
            Detailed report of expression capabilities and history
        """
        return {
            "avatar_id": self.avatar_id,
            "supported_emotions": [emotion.value for emotion in EmotionalState],
            "intensity_levels": [intensity.value for intensity in ExpressionIntensity],
            "current_expression": {
                "emotion": self.current_expression.state.value if self.current_expression else None,
                "intensity": self.current_expression.intensity.value if self.current_expression else None,
                "duration_remaining": self.current_expression.duration if self.current_expression else 0
            },
            "expression_history_count": len(self.expression_history),
            "emotion_frequencies": self._calculate_emotion_frequencies(),
            "capabilities": {
                "spontaneous_expressions": True,
                "emotion_blending": self.emotion_blending_enabled,
                "personality_influence": True,
                "consciousness_resonance": True,
                "quantum_emotional_states": True
            },
            "revolutionary_features": [
                "Real-time emotion-to-visual translation",
                "Personality-influenced expressions",
                "Quantum emotional state rendering",
                "Consciousness-coherent displays",
                "Transcendent expression capabilities"
            ]
        }
    
    def _calculate_emotion_frequencies(self) -> Dict[str, float]:
        """Calculate frequency of different emotions in expression history"""
        if not self.expression_history:
            return {}
        
        emotion_counts = {}
        for expression in self.expression_history:
            emotion = expression.state.value
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        total_expressions = len(self.expression_history)
        return {emotion: count / total_expressions 
                for emotion, count in emotion_counts.items()}