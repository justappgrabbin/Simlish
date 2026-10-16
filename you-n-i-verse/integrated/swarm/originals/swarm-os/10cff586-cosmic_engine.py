#!/usr/bin/env python3
"""
13-LAYER COSMIC-MATERIAL TRANSLATION ENGINE
============================================
Translates 360° zodiacal coordinates through 13-layer fractal matrix.
Outputs: BANTU encoding + Air Physics + Hex Colors + Sonic Vowels

Usage:
  python cosmic_engine.py --coordinate 136.75 [--altitude 2000]
  python cosmic_engine.py --batch [0,45.5,90,136.75,180,225.25,270,315.5]
"""

import math
import argparse
import json
from typing import Dict, List

# CONSTANTS
R_SPECIFIC_AIR = 287.058     # J/(kg·K)
GAMMA_AIR = 1.4              # Heat capacity ratio
M_AIR = 0.0289647            # kg/mol
G_EARTH = 9.80665            # m/s²
P0_STANDARD = 101325.0       # Pa
T0_STANDARD = 288.15         # K

# 13-LAYER MATRIX SPECIFICATION
LAYER_13_SPEC = {
    1:  {"name": "Planetary_Dimension", "width_deg": 360.0,      "output": "frequency_band+density"},
    2:  {"name": "Gate_Line",           "width_deg": 5.625,      "output": "64_data_blocks"},
    3:  {"name": "Color_Spectrum",      "width_deg": 0.15625,    "output": "base6_color_index"},
    4:  {"name": "Tone_Structure",      "width_deg": 0.0260417,  "output": "subharmonic_freq"},
    5:  {"name": "Base_Anchor",         "width_deg": 0.0052083,  "output": "ground_state"},
    6:  {"name": "Degree_Mapping",      "width_deg": 1.0,        "output": "macroposition_0_359"},
    7:  {"name": "Minute_FineTuning",   "width_deg": 1/60,       "output": "base60_modulo"},
    8:  {"name": "Second_Precision",    "width_deg": 1/3600,     "output": "micro_targeting"},
    9:  {"name": "ArcSecond_MicroGeometry", "width_deg": 1/3600, "output": "vector_stabilizer"},
    10: {"name": "Zodiac_House_Sector", "width_deg": 30.0,       "output": "contextual_filter"},
    11: {"name": "Five_Ws_Matrix",      "width_deg": None,       "output": "semantic_intent"},
    12: {"name": "Sonic_Vowel_Keys",    "width_deg": 72.0,       "output": "phonetic_identifier"},
    13: {"name": "AX_Realignment",      "width_deg": None,       "output": "validation_script"},
}

# HEX COLOR DICTIONARIES
TIER_A_PERSONALITY = {
    1: {"name": "Fear_Survival",     "hex": "#FF4B4B", "rgb": [255, 75, 75]},
    2: {"name": "Hope_Possibility",  "hex": "#FFB84D", "rgb": [255, 184, 77]},
    3: {"name": "Desire_Action",     "hex": "#FFE600", "rgb": [255, 230, 0]},
    4: {"name": "Need_Structure",    "hex": "#2ECC71", "rgb": [46, 204, 113]},
    5: {"name": "Guilt_Realignment", "hex": "#3498DB", "rgb": [52, 152, 219]},
    6: {"name": "Innocence_Observer","hex": "#9B59B6", "rgb": [155, 89, 182]},
}

TIER_B_DESIGN = {
    1: {"name": "Caves_SolidAnchor",   "hex": "#3E2723", "rgb": [62, 39, 35]},
    2: {"name": "Markets_Exchange",    "hex": "#D35400", "rgb": [211, 84, 0]},
    3: {"name": "Kitchens_Transform",  "hex": "#E74C3C", "rgb": [231, 76, 60]},
    4: {"name": "Mountains_Altitude",  "hex": "#BDC3C7", "rgb": [189, 195, 199]},
    5: {"name": "Valleys_FlowBasins",  "hex": "#16A085", "rgb": [22, 160, 133]},
    6: {"name": "Shores_Thresholds",   "hex": "#E1BEE7", "rgb": [225, 190, 231]},
}

# 5 AIR FORMULAS
def air_density(P, T): return P / (R_SPECIFIC_AIR * T)
def speed_of_sound(T): return math.sqrt(GAMMA_AIR * R_SPECIFIC_AIR * T)
def air_drag(rho, v, Cd, A): return 0.5 * rho * v**2 * Cd * A
def pressure_at_altitude(h, T=T0_STANDARD, P0=P0_STANDARD):
    return P0 * math.exp(-M_AIR * G_EARTH * h / (8.314 * T))


class CosmicTranslationEngine:
    """13-Layer Cosmic-Material Translation Engine."""

    NATO = ['Alpha','Bravo','Charlie','Delta','Echo','Foxtrot','Golf','Hotel','India',
            'Juliett','Kilo','Lima','Mike','November','Oscar','Papa','Quebec','Romeo',
            'Sierra','Tango','Uniform','Victor','Whiskey','X-ray','Yankee','Zulu']
    VOWELS = ['A', 'E', 'I', 'O', 'U']
    ZODIAC = ['Aries','Taurus','Gemini','Cancer','Leo','Virgo',
              'Libra','Scorpio','Sagittarius','Capricorn','Aquarius','Pisces']
    FIVE_WS = ['Who','What','Where','When','Why']

    def coordinate_to_13_layers(self, degrees: float) -> Dict:
        coord = degrees % 360.0
        result = {"input_degrees": coord}
        result[1] = self._layer1_planetary(coord)
        result[2] = self._layer2_gate_line(coord)
        result[3] = self._layer3_color(coord)
        result[4] = self._layer4_tone(coord)
        result[5] = self._layer5_anchor(coord)
        result[6] = self._layer6_degree(coord)
        result[7] = self._layer7_minute(coord)
        result[8] = self._layer8_second(coord)
        result[9] = self._layer9_arcsecond(coord)
        result[10] = self._layer10_zodiac(coord)
        result[11] = self._layer11_five_ws(coord)
        result[12] = self._layer12_vowel(coord)
        result[13] = self._layer13_ax(result)
        return result

    def _layer1_planetary(self, coord):
        band = int(coord / 30) + 1
        freq = 20 * (1000 ** (coord / 360))
        density = 0.5 + (coord / 360) * 2.0
        return {"band": band, "frequency_hz": round(freq, 2), "density_kg_m3": round(density, 3)}

    def _layer2_gate_line(self, coord):
        gate = int(coord / 5.625) + 1
        remainder = coord % 5.625
        line = int(remainder / 0.9375) + 1
        return {"gate": gate, "line": line, "gate_name": self.NATO[(gate-1) % 26]}

    def _layer3_color(self, coord):
        line_remainder = coord % 0.9375
        color_index = int(line_remainder / 0.15625) + 1
        next_color = (color_index % 6) + 1
        return {
            "color_index": color_index,
            "next_color": next_color,
            "tier_a_personality": TIER_A_PERSONALITY[color_index],
            "tier_b_design": TIER_B_DESIGN[color_index],
            "hex_personality": TIER_A_PERSONALITY[color_index]["hex"],
            "hex_design": TIER_B_DESIGN[color_index]["hex"]
        }

    def _layer4_tone(self, coord):
        tone_idx = int(coord / 0.0260417) % 216
        frequency = 432 * (2 ** (tone_idx / 216))
        return {"tone_index": tone_idx, "frequency_hz": round(frequency, 2)}

    def _layer5_anchor(self, coord):
        anchor = int(coord / 0.0052083) % 1080
        return {"anchor_index": anchor, "ground_state": f"GS-{anchor:04d}"}

    def _layer6_degree(self, coord):
        return {"degree": int(coord), "fraction": coord - int(coord)}

    def _layer7_minute(self, coord):
        frac = coord - int(coord)
        minutes = int(frac * 60)
        return {"minutes": minutes, "base60_value": minutes}

    def _layer8_second(self, coord):
        frac = coord - int(coord)
        total_seconds = frac * 3600
        seconds = int(total_seconds) % 60
        return {"seconds": seconds, "micro_target": round(total_seconds, 4)}

    def _layer9_arcsecond(self, coord):
        frac = coord - int(coord)
        arcsec = frac * 3600
        return {"arcseconds": round(arcsec, 2), "vector_stability": round(1.0 / (1.0 + arcsec * 0.001), 6)}

    def _layer10_zodiac(self, coord):
        house = int(coord / 30)
        sign = self.ZODIAC[house]
        degree_in_sign = coord % 30
        return {"house": house + 1, "sign": sign, "degree_in_sign": round(degree_in_sign, 4)}

    def _layer11_five_ws(self, coord):
        idx = int(coord / 72) % 5
        w = self.FIVE_WS[idx]
        intent_strength = (coord % 72) / 72.0
        return {"primary_w": w, "intent_strength": round(intent_strength, 4)}

    def _layer12_vowel(self, coord):
        vowel_idx = int(coord / 72) % 5
        vowel = self.VOWELS[vowel_idx]
        harmonic = 1 + (coord % 72) / 72.0
        return {"vowel": vowel, "vowel_index": vowel_idx, "harmonic": round(harmonic, 4)}

    def _layer13_ax(self, layers):
        gate = layers[2]["gate"]
        color = layers[3]["color_index"]
        zodiac = layers[10]["house"]
        vowel = layers[12]["vowel_index"]
        alignment = (gate * color * zodiac * (vowel + 1)) % 360
        validated = alignment < 180
        return {
            "alignment_score": alignment,
            "validated": validated,
            "triggers": f"G{gate}-C{color}-Z{zodiac}-V{vowel}",
            "movement_vector": [gate % 8, color % 6, zodiac % 12]
        }

    def compute_air_physics(self, layers: Dict, altitude: float = 0) -> Dict:
        T = T0_STANDARD + (layers[1]["frequency_hz"] / 1000) * 10
        P = P0_STANDARD * (1 - altitude / 50000)
        V = 1.0 / max(layers[1]["density_kg_m3"], 0.1)
        return {
            "temperature_k": round(T, 2),
            "pressure_pa": round(P, 2),
            "volume_m3": round(V, 4),
            "density_kg_m3": round(air_density(P, T), 4),
            "sound_speed_m_s": round(speed_of_sound(T), 2),
            "drag_force_n": round(air_drag(air_density(P, T), 10, 0.47, 0.01), 6),
            "pressure_at_altitude_pa": round(pressure_at_altitude(altitude, T), 2)
        }


def main():
    parser = argparse.ArgumentParser(description='13-Layer Cosmic Translation Engine')
    parser.add_argument('--coordinate', type=float, help='Degrees (0-360)')
    parser.add_argument('--batch', type=str, help='Comma-separated degrees')
    parser.add_argument('--altitude', type=float, default=0, help='Meters')
    parser.add_argument('--output', type=str, default=None, help='JSON output file')
    args = parser.parse_args()

    engine = CosmicTranslationEngine()

    coords = []
    if args.coordinate is not None:
        coords = [args.coordinate]
    elif args.batch:
        coords = [float(x.strip()) for x in args.batch.strip('[]').split(',')]
    else:
        coords = [0, 45.5, 90, 136.75, 180, 225.25, 270, 315.5]

    results = {}
    for coord in coords:
        r = engine.coordinate_to_13_layers(coord)
        air = engine.compute_air_physics(r, args.altitude)
        results[str(coord)] = {"layers": r, "air_physics": air}

        print(f"\n{'='*60}")
        print(f"INPUT: {coord}°")
        print(f"{'='*60}")
        print(f"L2 Gate/Line:  {r[2]['gate']} ({r[2]['gate_name']}) / Line {r[2]['line']}")
        print(f"L3 Color:      {r[3]['hex_personality']} / {r[3]['hex_design']}")
        print(f"L10 Zodiac:    {r[10]['sign']} (House {r[10]['house']})")
        print(f"L12 Vowel:     {r[12]['vowel']}")
        print(f"L13 AX:        {r[13]['triggers']} | Validated: {r[13]['validated']}")
        print(f"Air:           T={air['temperature_k']}K, c={air['sound_speed_m_s']}m/s")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {args.output}")


if __name__ == '__main__':
    main()
