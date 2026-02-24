"""
EXPERIENCE TEXTURE GENERATOR
Translates GAN outputs into felt experience

Problem: GANs generate patterns (codons, gates, profiles) but not EXPERIENCE
Solution: This layer adds texture, narrative, and felt sense

Input:  [Gate 12, Line 3, Profile 4/6, Codon patterns]
Output: "Feels like X, manifests as Y, plays like Z"
"""

import numpy as np
import json
from typing import Dict, List, Optional


class ExperienceTextureGenerator:
    """
    Converts abstract patterns into textured experience
    
    Takes GAN outputs and generates:
    - Felt sense (what it FEELS like)
    - Behavioral texture (how it MANIFESTS)
    - Gameplay narrative (how it PLAYS)
    - Consciousness quality (the TEXTURE)
    """
    
    def __init__(self):
        # Load texture libraries
        self.gate_textures = self._load_gate_textures()
        self.codon_textures = self._load_codon_textures()
        self.profile_textures = self._load_profile_textures()
        self.field_textures = self._load_field_textures()
        
        print("🎨 ExperienceTextureGenerator initialized")
    
    def _load_gate_textures(self) -> Dict:
        """
        Gate textures - what each gate FEELS like
        Beyond just definitions - the actual texture
        """
        return {
            12: {
                "surface_feel": "restraint and caution",
                "depth_feel": "knowing when to speak and when to hold back",
                "texture": "like holding your breath before diving",
                "quality": "articulate silence",
                "manifests_as": "strategic pauses, careful timing",
                "gameplay": "character hesitates at key moments, then speaks truth"
            },
            41: {
                "surface_feel": "pressure to begin",
                "depth_feel": "fantasy and imagination building toward manifestation",
                "texture": "like standing at the edge before jumping",
                "quality": "creative tension",
                "manifests_as": "dreaming into reality, initiating cycles",
                "gameplay": "character gets visions/ideas that drive action"
            },
            33: {
                "surface_feel": "withdrawal and reflection",
                "depth_feel": "processing experience into wisdom",
                "texture": "like retreating to a cave to understand",
                "quality": "reflective privacy",
                "manifests_as": "needing space, then sharing insights",
                "gameplay": "character periodically disappears, returns wiser"
            },
            # Add all 64 gates with textures
            # This is just example structure
        }
    
    def _load_codon_textures(self) -> Dict:
        """
        Codon textures - how DNA patterns feel experientially
        """
        return {
            1: {
                "name": "Creative Power",
                "amino_acid": "Methionine",
                "texture": "initiating force, raw creative potential",
                "quality": "pure beginning",
                "feels_like": "the moment before something new emerges",
                "consciousness": "creative self-expression"
            },
            2: {
                "name": "Receptive Space",
                "amino_acid": "Various",
                "texture": "yielding receptivity, allowing",
                "quality": "pure allowing",
                "feels_like": "opening to receive what comes",
                "consciousness": "responsive awareness"
            },
            # All 64 codons
        }
    
    def _load_profile_textures(self) -> Dict:
        """
        Profile textures - how personality/life path feels
        """
        return {
            "1/3": {
                "feel": "investigative experimentation",
                "texture": "studying deeply then trying everything",
                "life_quality": "researcher who learns through mistakes",
                "gameplay": "character researches, then experiments, then integrates"
            },
            "4/6": {
                "feel": "network influence through trial",
                "texture": "connecting people while learning life's lessons",
                "life_quality": "influential friend who becomes wise through experience",
                "gameplay": "character builds networks, goes through phases, emerges as guide"
            },
            # All 12 profiles
        }
    
    def _load_field_textures(self) -> Dict:
        """
        Field textures - how consciousness fields feel
        """
        return {
            "mind": {
                "texture": "clear, analytical, sharp",
                "quality": "crystalline awareness",
                "feels_like": "cutting through fog with laser precision",
                "manifests_as": "quick insights, mental clarity"
            },
            "heart": {
                "texture": "warm, connected, resonant",
                "quality": "harmonic connection",
                "feels_like": "tuning fork vibrating with others",
                "manifests_as": "emotional attunement, empathy waves"
            },
            "body": {
                "texture": "grounded, present, kinetic",
                "quality": "embodied nowness",
                "feels_like": "feet on earth, blood pumping",
                "manifests_as": "physical movement, sensory awareness"
            },
            # All 9 fields
        }
    
    # ===== EXPERIENCE GENERATION =====
    
    def generate_experience(self, gan_output: Dict) -> Dict:
        """
        Main method: Convert GAN output into textured experience
        
        Input: {
            'gates': [12, 41, 33],
            'profile': '4/6',
            'codons': {...},
            'fields': {...}
        }
        
        Output: {
            'felt_sense': "...",
            'texture': "...",
            'narrative': "...",
            'gameplay': "..."
        }
        """
        experience = {
            "core_feeling": self._synthesize_core_feeling(gan_output),
            "texture_description": self._generate_texture_description(gan_output),
            "behavioral_manifestation": self._generate_behavioral_texture(gan_output),
            "consciousness_quality": self._generate_consciousness_quality(gan_output),
            "gameplay_narrative": self._generate_gameplay_narrative(gan_output),
            "moment_to_moment": self._generate_moment_texture(gan_output)
        }
        
        return experience
    
    def _synthesize_core_feeling(self, gan_output: Dict) -> str:
        """What does this configuration FEEL like?"""
        gates = gan_output.get('gates', [])
        profile = gan_output.get('profile', '')
        
        # Get textures for each gate
        gate_feels = []
        for gate in gates:
            if gate in self.gate_textures:
                gate_feels.append(self.gate_textures[gate]['depth_feel'])
        
        # Get profile feel
        profile_feel = ""
        if profile in self.profile_textures:
            profile_feel = self.profile_textures[profile]['feel']
        
        # Synthesize
        if gate_feels and profile_feel:
            core = f"A {profile_feel} that experiences {' and '.join(gate_feels)}"
        elif gate_feels:
            core = f"An experience of {' and '.join(gate_feels)}"
        else:
            core = "A unique configuration of consciousness"
        
        return core
    
    def _generate_texture_description(self, gan_output: Dict) -> str:
        """
        Create rich texture description
        This is what makes it FEEL real
        """
        gates = gan_output.get('gates', [])
        fields = gan_output.get('fields', {})
        
        textures = []
        
        # Gate textures
        for gate in gates[:3]:  # Top 3 gates
            if gate in self.gate_textures:
                tex = self.gate_textures[gate]['texture']
                textures.append(tex)
        
        # Field texture (if dominant field)
        if fields:
            dominant_field = max(fields, key=fields.get)
            if dominant_field in self.field_textures:
                field_tex = self.field_textures[dominant_field]['texture']
                textures.append(f"with a {field_tex} quality")
        
        # Combine
        if len(textures) > 1:
            description = f"{textures[0]}, {textures[1]}"
            if len(textures) > 2:
                description += f", {textures[2]}"
        elif textures:
            description = textures[0]
        else:
            description = "a subtle and undefined quality"
        
        return description
    
    def _generate_behavioral_texture(self, gan_output: Dict) -> str:
        """How this pattern MANIFESTS in behavior"""
        gates = gan_output.get('gates', [])
        profile = gan_output.get('profile', '')
        
        behaviors = []
        
        # Gate manifestations
        for gate in gates[:2]:
            if gate in self.gate_textures:
                behaviors.append(self.gate_textures[gate]['manifests_as'])
        
        # Profile manifestation
        if profile in self.profile_textures:
            behaviors.append(self.profile_textures[profile]['gameplay'])
        
        return " → ".join(behaviors) if behaviors else "subtle behavioral patterns"
    
    def _generate_consciousness_quality(self, gan_output: Dict) -> str:
        """The QUALITY of consciousness in this state"""
        gates = gan_output.get('gates', [])
        codons = gan_output.get('codons', {})
        
        qualities = []
        
        # Gate qualities
        for gate in gates[:2]:
            if gate in self.gate_textures:
                qualities.append(self.gate_textures[gate]['quality'])
        
        # Codon consciousness (if available)
        if codons:
            # Use first codon as example
            codon_id = list(codons.keys())[0] if codons else None
            if codon_id and codon_id in self.codon_textures:
                qualities.append(self.codon_textures[codon_id]['consciousness'])
        
        return ", ".join(qualities) if qualities else "undefined consciousness quality"
    
    def _generate_gameplay_narrative(self, gan_output: Dict) -> str:
        """
        How this translates to actual gameplay/experience
        This is what the user/avatar DOES
        """
        gates = gan_output.get('gates', [])
        profile = gan_output.get('profile', '')
        fields = gan_output.get('fields', {})
        
        narrative_parts = []
        
        # Opening (profile-based)
        if profile in self.profile_textures:
            opening = self.profile_textures[profile]['gameplay']
            narrative_parts.append(opening)
        
        # Action pattern (gate-based)
        for gate in gates[:2]:
            if gate in self.gate_textures:
                action = self.gate_textures[gate]['gameplay']
                narrative_parts.append(action)
        
        # Energy quality (field-based)
        if fields:
            dominant = max(fields, key=fields.get)
            if dominant in self.field_textures:
                energy = self.field_textures[dominant]['manifests_as']
                narrative_parts.append(f"All while {energy}")
        
        return ". ".join(narrative_parts) if narrative_parts else "Character moves through experience"
    
    def _generate_moment_texture(self, gan_output: Dict) -> Dict:
        """
        Moment-to-moment experience texture
        What it feels like RIGHT NOW
        """
        gates = gan_output.get('gates', [])
        fields = gan_output.get('fields', {})
        
        # Get immediate feeling
        immediate = ""
        if gates and gates[0] in self.gate_textures:
            immediate = self.gate_textures[gates[0]]['surface_feel']
        
        # Get depth feeling
        depth = ""
        if gates and gates[0] in self.gate_textures:
            depth = self.gate_textures[gates[0]]['depth_feel']
        
        # Get texture
        texture = ""
        if gates and gates[0] in self.gate_textures:
            texture = self.gate_textures[gates[0]]['texture']
        
        return {
            "immediate": immediate or "present awareness",
            "depth": depth or "underlying patterns",
            "texture": texture or "subtle quality",
            "now": f"Right now: {immediate}. Beneath: {depth}."
        }
    
    # ===== INTEGRATION WITH GANS =====
    
    def texture_from_codon_gan(self, codon_output: np.ndarray) -> Dict:
        """
        Convert Codon GAN output to experience
        
        Input: Codon activation patterns from GAN
        Output: Textured experience
        """
        # Find dominant codons
        dominant_indices = np.argsort(codon_output)[-3:][::-1]
        
        # Get codon IDs (assuming 64 codons)
        dominant_codons = [int(idx) + 1 for idx in dominant_indices]
        
        # Build experience from codons
        textures = []
        for codon_id in dominant_codons:
            if codon_id in self.codon_textures:
                tex = self.codon_textures[codon_id]
                textures.append({
                    "codon": codon_id,
                    "feel": tex['feels_like'],
                    "quality": tex['quality'],
                    "consciousness": tex['consciousness']
                })
        
        # Synthesize
        if textures:
            primary = textures[0]
            experience = {
                "primary_quality": primary['quality'],
                "felt_sense": primary['feel'],
                "consciousness_state": primary['consciousness'],
                "texture": f"{primary['quality']} that {primary['feel']}"
            }
        else:
            experience = {"texture": "undefined codon pattern"}
        
        return experience
    
    def texture_from_hd_gan(self, hd_output: Dict) -> Dict:
        """
        Convert Human Design GAN output to experience
        
        Input: {
            'gates': [...],
            'profile': '...',
            'type': '...',
            ...
        }
        Output: Textured experience
        """
        return self.generate_experience(hd_output)
    
    # ===== REAL-TIME TEXTURING =====
    
    def texture_gameplay_moment(self, 
                                  current_state: Dict,
                                  action: str,
                                  context: Dict) -> Dict:
        """
        Generate experience texture for a specific gameplay moment
        
        Input:
            current_state: GAN output for current consciousness
            action: What user/avatar is doing
            context: Scene/environment context
        
        Output: Textured moment
        """
        base_experience = self.generate_experience(current_state)
        
        # Add action texture
        action_texture = self._texture_action(action, current_state)
        
        # Add context texture
        context_texture = self._texture_context(context, current_state)
        
        # Combine
        moment = {
            "base_feel": base_experience['core_feeling'],
            "action_feel": action_texture,
            "context_feel": context_texture,
            "complete_moment": f"{base_experience['texture_description']} while {action_texture} in {context_texture}",
            "narrative": f"{base_experience['gameplay_narrative']}. Now: {action_texture}."
        }
        
        return moment
    
    def _texture_action(self, action: str, state: Dict) -> str:
        """Add texture to an action based on state"""
        gates = state.get('gates', [])
        
        # Simple texture mapping
        if gates and gates[0] == 12:
            return f"{action} with careful deliberation"
        elif gates and gates[0] == 41:
            return f"{action} with anticipatory excitement"
        elif gates and gates[0] == 33:
            return f"{action} with reflective awareness"
        else:
            return action
    
    def _texture_context(self, context: Dict, state: Dict) -> str:
        """Add texture to context based on state"""
        location = context.get('location', 'unknown space')
        fields = state.get('fields', {})
        
        if fields:
            dominant = max(fields, key=fields.get)
            if dominant == 'mind':
                return f"the analytical clarity of {location}"
            elif dominant == 'heart':
                return f"the emotional resonance of {location}"
            elif dominant == 'body':
                return f"the physical presence of {location}"
        
        return location


# ===== USAGE EXAMPLE =====

if __name__ == "__main__":
    # Initialize
    texture_gen = ExperienceTextureGenerator()
    
    # Example GAN output
    gan_output = {
        'gates': [12, 41, 33],
        'profile': '4/6',
        'codons': {1: 0.8, 2: 0.6},
        'fields': {
            'mind': 0.7,
            'heart': 0.6,
            'body': 0.5
        }
    }
    
    # Generate experience
    experience = texture_gen.generate_experience(gan_output)
    
    print("\n" + "="*60)
    print("EXPERIENCE TEXTURE GENERATION")
    print("="*60)
    print(f"\nCore Feeling: {experience['core_feeling']}")
    print(f"\nTexture: {experience['texture_description']}")
    print(f"\nManifestation: {experience['behavioral_manifestation']}")
    print(f"\nConsciousness Quality: {experience['consciousness_quality']}")
    print(f"\nGameplay: {experience['gameplay_narrative']}")
    print(f"\nMoment: {experience['moment_to_moment']['now']}")
    print("="*60)
