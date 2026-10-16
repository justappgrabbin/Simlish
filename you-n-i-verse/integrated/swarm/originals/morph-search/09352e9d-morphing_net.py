
"""
MORPHING NET — UNIVERSAL EXECUTION ENGINE v1.0
Backend: FastAPI + Substrate Pipeline + Supabase + Neo4j

Pipeline: INGEST → DISSOLVE → ANALYZE → SYNTHESIZE → EXECUTE → PERSIST → RESPOND
"""

import hashlib
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

# ============================================================================
# CONFIGURATION
# ============================================================================

SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_ANON_KEY = "your-anon-key"
SUPABASE_SERVICE_KEY = "your-service-key"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

# ============================================================================
# ENUMS
# ============================================================================

class ContentType(Enum):
    WEB_APP = "web_app"
    GAME = "game"
    API = "api"
    DATA_SCHEMA = "data_schema"
    CONFIG = "config"
    TOOL = "tool"
    DOCUMENT = "document"
    MEDIA = "media"
    UNKNOWN = "unknown"

class Purpose(Enum):
    DISPLAY = "display"
    COLLECT = "collect"
    TRANSFORM = "transform"
    COMMUNICATE = "communicate"
    ORGANIZE = "organize"
    CALCULATE = "calculate"
    VISUALIZE = "visualize"
    CONTROL = "control"
    ENTERTAIN = "entertain"
    EDUCATE = "educate"

class Dialect(Enum):
    STANDARD = "standard"
    REVERSED = "reversed"
    COMPLEMENTARY = "complementary"
    KARNAUGH = "karnaugh"
    TERNARY = "ternary"

class Center(Enum):
    HEAD = "Head"
    AJNA = "Ajna"
    THROAT = "Throat"
    G = "G"
    HEART = "Heart"
    SPLEEN = "Spleen"
    SOLAR_PLEXUS = "Solar Plexus"
    SACRAL = "Sacral"
    ROOT = "Root"

class Circuit(Enum):
    UNDERSTANDING = "Understanding"
    SENSING = "Sensing"
    KNOWING = "Knowing"
    CENTERING = "Centering"
    INTEGRATION = "Integration"

class Family(Enum):
    INITIATION = "Initiation"
    ACCEPTANCE = "Acceptance"
    MONEY = "Money"
    MUTATION = "Mutation"
    INSPIRATION = "Inspiration"
    AWAKENING = "Awakening"
    WAVELENGTH = "Wavelength"
    TRANSITORINESS = "Transitoriness"

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class GateActivation:
    gate: int
    strength: float
    dialect: Dialect
    layer: int
    byte_position: int
    center: Center = None
    family: Family = None
    circuit: Circuit = None

@dataclass
class Trigram:
    name: str
    binary: int
    operator: str
    center: Center

@dataclass
class EmergentState:
    id: str
    tetragram: int
    source_gates: Tuple[int, int]
    crystallized_at: float
    resonance_pattern: List[int]
    bridges_to: List[int]

@dataclass
class DissolvedState:
    source_name: str
    source_type: str
    source_size: int
    source_hash: str
    activations: List[GateActivation]
    trigrams: List[Trigram]
    dialects: List[Dict]
    emergent_seeds: List[int]
    dominant_centers: List[Center]
    dominant_families: List[Family]
    morph_vectors: List[Dict]

@dataclass
class AnalysisResult:
    content_type: ContentType
    purpose: Purpose
    confidence: float
    detected_features: List[str]
    entry_points: List[str]
    dependencies: List[str]
    structure: Dict
    scores: Dict[str, float] = field(default_factory=dict)

@dataclass
class ComponentSpec:
    id: str
    type: str
    gate: int
    center: Center
    purpose: str
    props: Dict[str, Any]
    children: List[str]
    logic: List[str]
    style: Dict[str, str]
    weight: float

@dataclass
class AppArchitecture:
    content_type: ContentType
    purpose: Purpose
    components: List[ComponentSpec]
    layout: str
    data_flow: str
    entry_component: str
    features: List[str]
    emergent_injections: List[str]
    substrate_graph: Dict

@dataclass
class Artifact:
    id: str
    type: str
    python: Optional[str]
    html: Optional[str]
    js: Optional[str]
    css: Optional[str]
    json_config: Optional[str]
    metadata: Dict

@dataclass
class MorphResponse:
    request_id: str
    status: str
    dissolved: Dict
    analysis: Dict
    synthesis: Dict
    artifact: Dict
    supabase_ref: Optional[str]
    neo4j_ref: Optional[str]
    execution_time_ms: float

# ============================================================================
# CONSTANTS
# ============================================================================

TRIGRAMS = [
    Trigram("Qian", 7, "OR", Center.HEAD),
    Trigram("Kun", 0, "AND", Center.AJNA),
    Trigram("Kan", 2, "NOT", Center.SPLEEN),
    Trigram("Li", 5, "XOR", Center.SOLAR_PLEXUS),
    Trigram("Zhen", 4, "NAND", Center.SACRAL),
    Trigram("Xun", 3, "NOR", Center.HEART),
    Trigram("Gen", 1, "XNOR", Center.G),
    Trigram("Dui", 6, "IMPLY", Center.THROAT)
]

CENTER_GATES = {
    Center.HEAD: [64, 61, 63],
    Center.AJNA: [47, 24, 4, 43, 11, 17],
    Center.THROAT: [62, 23, 56, 35, 12, 45, 33, 20, 8, 31, 16],
    Center.G: [1, 13, 25, 15, 10, 7, 2, 46],
    Center.HEART: [21, 40, 51, 26],
    Center.SPLEEN: [48, 57, 44, 50, 32, 28, 18, 10],
    Center.SOLAR_PLEXUS: [55, 30, 36, 22, 37, 6, 49],
    Center.SACRAL: [34, 5, 14, 29, 59, 9, 3, 42, 27, 52],
    Center.ROOT: [53, 60, 41, 19, 58, 38, 39, 54]
}

FAMILY_CLUSTERS = {
    Family.INITIATION: [25, 51, 17, 62, 21, 45, 24, 61],
    Family.ACCEPTANCE: [17, 62, 47, 24, 43, 11, 4, 63],
    Family.MONEY: [21, 45, 26, 40, 37, 6, 49, 55],
    Family.MUTATION: [42, 53, 3, 60, 9, 52, 14, 5],
    Family.INSPIRATION: [8, 1, 13, 33, 56, 12, 20, 16],
    Family.AWAKENING: [20, 10, 34, 57, 44, 32, 50, 28],
    Family.WAVELENGTH: [16, 48, 18, 58, 38, 54, 39, 41],
    Family.TRANSITORINESS: [35, 36, 22, 30, 29, 46, 27, 2]
}

INTEGRATION_GATES = [34, 20, 10, 57]
SENSING_GATES = [42, 53, 29, 46, 36, 35]

HEXAGRAM_NAMES = [
    "The Creative", "The Receptive", "Difficulty at the Beginning", "Youthful Folly",
    "Waiting", "Conflict", "The Army", "Holding Together", "Small Taming", "Treading",
    "Peace", "Standstill", "Fellowship", "Great Possession", "Modesty", "Enthusiasm",
    "Following", "Work on Decay", "Approach", "Contemplation", "Biting Through", "Grace",
    "Splitting Apart", "Return", "Innocence", "Great Taming", "Nourishment",
    "Great Preponderance", "The Abysmal", "The Clinging", "Influence", "Duration",
    "Retreat", "Great Power", "Progress", "Darkening of the Light", "The Family",
    "Opposition", "Obstruction", "Deliverance", "Decrease", "Increase", "Breakthrough",
    "Coming to Meet", "Gathering Together", "Pushing Upward", "Oppression", "The Well",
    "Revolution", "The Cauldron", "The Arousing", "Keeping Still", "Development",
    "The Marrying Maiden", "Abundance", "The Wanderer", "The Gentle", "The Joyous",
    "Dispersion", "Limitation", "Inner Truth", "Preponderance of the Small",
    "After Completion", "Before Completion"
]

# ============================================================================
# BINARY MORPH KERNEL
# ============================================================================

class BinaryMorph:
    @staticmethod
    def gate_to_binary(gate: int) -> int:
        return gate - 1

    @staticmethod
    def binary_to_gate(binary: int) -> int:
        return (binary & 0x3F) + 1

    @staticmethod
    def inverse(gate: int) -> int:
        return BinaryMorph.binary_to_gate((~(gate - 1)) & 0x3F)

    @staticmethod
    def xor(a: int, b: int) -> int:
        return BinaryMorph.binary_to_gate((a - 1) ^ (b - 1))

    @staticmethod
    def and_op(a: int, b: int) -> int:
        return BinaryMorph.binary_to_gate((a - 1) & (b - 1))

    @staticmethod
    def or_op(a: int, b: int) -> int:
        return BinaryMorph.binary_to_gate((a - 1) | (b - 1))

    @staticmethod
    def upper_trigram(gate: int) -> int:
        return ((gate - 1) >> 3) & 0x07

    @staticmethod
    def lower_trigram(gate: int) -> int:
        return (gate - 1) & 0x07

    @staticmethod
    def get_trigram(binary: int) -> Trigram:
        return next((t for t in TRIGRAMS if t.binary == (binary & 0x07)), TRIGRAMS[0])

    @staticmethod
    def gate_to_karnaugh(gate: int) -> Tuple[int, int, int]:
        bin_val = gate - 1
        bits = [(bin_val >> i) & 1 for i in range(6)]
        return (bits[0] + 2 * bits[1], bits[2] + 2 * bits[3], bits[4] + 2 * bits[5])

    @staticmethod
    def karnaugh_distance(g1: int, g2: int) -> int:
        x1, y1, z1 = BinaryMorph.gate_to_karnaugh(g1)
        x2, y2, z2 = BinaryMorph.gate_to_karnaugh(g2)
        return abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2)

# ============================================================================
# MORPHING NET ENGINE
# ============================================================================

class MorphingNetEngine:
    def __init__(self):
        self.nodes: Dict[str, Any] = {}
        self.counter = 0
        self.emergent_states: Dict[str, EmergentState] = {}
        self.emergent_counter = 0
        self.cycle_state: Dict[str, int] = {}
        self.memory: List[Dict] = []
        self.initialize_gates()

    def initialize_gates(self):
        for gate in range(1, 65):
            center = self._get_center(gate)
            family = self._get_family(gate)
            circuit = self._get_circuit(center)
            binary = BinaryMorph.gate_to_binary(gate)

            self.nodes[f"gate_{gate}"] = {
                "id": f"gate_{gate}", "gate": gate, "binary": binary,
                "inverse": BinaryMorph.inverse(gate), "nuclear": ((binary >> 1) & 0x0F) + 1,
                "upper_trigram": BinaryMorph.get_trigram(BinaryMorph.upper_trigram(gate)),
                "lower_trigram": BinaryMorph.get_trigram(BinaryMorph.lower_trigram(gate)),
                "center": center, "circuit": circuit, "family": family,
                "weight": self._fibonacci((gate % 8) + 1) / 100,
                "usage_count": 0, "connections": [], "activation_level": 0.0,
                "cycle_position": "begin", "message_queue": []
            }

        self._connect_gates()

    def _get_center(self, gate: int) -> Center:
        for center, gates in CENTER_GATES.items():
            if gate in gates:
                return center
        return Center.ROOT

    def _get_family(self, gate: int) -> Family:
        for family, gates in FAMILY_CLUSTERS.items():
            if gate in gates:
                return family
        return Family.INITIATION

    def _get_circuit(self, center: Center) -> Circuit:
        mapping = {
            Center.HEAD: Circuit.UNDERSTANDING, Center.AJNA: Circuit.UNDERSTANDING,
            Center.THROAT: Circuit.CENTERING, Center.G: Circuit.CENTERING,
            Center.HEART: Circuit.CENTERING, Center.SPLEEN: Circuit.SENSING,
            Center.SOLAR_PLEXUS: Circuit.KNOWING, Center.SACRAL: Circuit.KNOWING,
            Center.ROOT: Circuit.CENTERING
        }
        return mapping.get(center, Circuit.CENTERING)

    def _fibonacci(self, n: int) -> int:
        if n <= 1:
            return n
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

    def _connect_gates(self):
        all_nodes = list(self.nodes.values())

        for node in all_nodes:
            same_family = [n for n in all_nodes if n["family"] == node["family"] and n["id"] != node["id"]]
            for sibling in same_family[:3]:
                if sibling["id"] not in node["connections"]:
                    node["connections"].append(sibling["id"])

            inverse_node = next((n for n in all_nodes if n["gate"] == node["inverse"]), None)
            if inverse_node:
                same_fam = inverse_node["family"] == node["family"]
                is_integration = node["gate"] in INTEGRATION_GATES or inverse_node["gate"] in INTEGRATION_GATES
                if (same_fam or is_integration) and inverse_node["id"] not in node["connections"]:
                    node["connections"].append(inverse_node["id"])

            if node["gate"] in INTEGRATION_GATES:
                for other in all_nodes:
                    if other["id"] != node["id"] and other["id"] not in node["connections"]:
                        node["connections"].append(other["id"])

    def morph(self, start_gate: int, target_gate: Optional[int] = None, operation: str = "XOR") -> Dict:
        start_node = self.nodes.get(f"gate_{start_gate}")
        if not start_node:
            raise ValueError(f"Gate {start_gate} not found")

        start_node["usage_count"] += 1
        start_node["activation_level"] = min(1.0, start_node["activation_level"] + 0.2)

        messages = []
        trigram_path = [start_node["upper_trigram"], start_node["lower_trigram"]]
        families_visited = [start_node["family"]]
        integration_nodes_used = []
        emergent_states = []
        layers_traversed = []
        cycle_switches = 0

        current_gate = start_gate
        current_node = start_node
        current_family = start_node["family"]

        # Layers 1-3: LOCAL
        for layer in range(1, 4):
            layers_traversed.append(layer)
            layer_op = ["XOR", "AND", "OR"][layer - 1]

            family_nodes = [n for n in self.nodes.values() if n["family"] == current_family]
            if family_nodes:
                idx = hash(f"{layer}_{current_gate}") % len(family_nodes)
                next_node = family_nodes[idx]
            else:
                next_node = None

            if target_gate and layer == 3:
                next_node = self.nodes.get(f"gate_{target_gate}")

            if not next_node:
                continue

            if layer_op == "XOR":
                result = BinaryMorph.xor(current_gate, next_node["gate"])
            elif layer_op == "AND":
                result = BinaryMorph.and_op(current_gate, next_node["gate"])
            else:
                result = BinaryMorph.or_op(current_gate, next_node["gate"])

            messages.append({
                "id": f"msg_{self.counter}", "from": f"gate_{current_gate}",
                "to": f"gate_{next_node['gate']}", "operation": layer_op,
                "layer": layer, "is_local": True, "family_hop": False, "gap_resonance": False
            })
            self.counter += 1

            current_gate = result
            current_node = self.nodes.get(f"gate_{result}", current_node)
            current_family = current_node["family"]

            if current_family not in families_visited:
                families_visited.append(current_family)
            trigram_path.extend([current_node["upper_trigram"], current_node["lower_trigram"]])

        # Layer 4: INTEGRATION
        layers_traversed.append(4)
        integration_node = self._find_nearest_integration(current_node)
        if integration_node:
            integration_nodes_used.append(integration_node["gate"])
            messages.append({
                "id": f"msg_{self.counter}", "from": f"gate_{current_gate}",
                "to": integration_node["id"], "operation": "XNOR",
                "layer": 4, "is_local": False, "family_hop": True, "gap_resonance": False
            })
            self.counter += 1

            current_node = integration_node
            current_gate = integration_node["gate"]
            trigram_path.extend([integration_node["upper_trigram"], integration_node["lower_trigram"]])

        # Layer 5: GAP RESONANCE
        layers_traversed.append(5)
        target = target_gate or ((current_gate % 64) + 1)
        target_num = int(target)

        dist = BinaryMorph.karnaugh_distance(current_gate, target_num)
        color_diff = abs((current_gate % 6) - (target_num % 6))
        tone_diff = abs(((current_gate + 2) % 6) - ((target_num + 2) % 6))
        tetragram = ((dist * 9) + (color_diff * 3) + tone_diff) % 81 + 1

        self.emergent_counter += 1
        emergent = EmergentState(
            id=f"emergent_{self.emergent_counter}", tetragram=tetragram,
            source_gates=(current_gate, target_num),
            crystallized_at=datetime.now().timestamp(),
            resonance_pattern=[dist, color_diff, tone_diff],
            bridges_to=[ig for ig in INTEGRATION_GATES if (tetragram + ig) % 7 == 0]
        )
        self.emergent_states[emergent.id] = emergent
        emergent_states.append(emergent)

        bridge_target = emergent.bridges_to[0] if emergent.bridges_to else 34
        bridge_node = self.nodes.get(f"gate_{bridge_target}")

        if bridge_node:
            messages.append({
                "id": f"msg_{self.counter}", "from": f"emergent_{emergent.id}",
                "to": f"gate_{bridge_target}", "operation": "OR",
                "layer": 5, "is_local": False, "family_hop": True, "gap_resonance": True
            })
            self.counter += 1

            current_node = bridge_node
            current_gate = bridge_target
            current_family = bridge_node["family"]

            if current_family not in families_visited:
                families_visited.append(current_family)
            if bridge_target not in integration_nodes_used:
                integration_nodes_used.append(bridge_target)
            trigram_path.extend([bridge_node["upper_trigram"], bridge_node["lower_trigram"]])

        # Cycle switching
        if current_gate in SENSING_GATES:
            cycle_switches += 1
            self._advance_cycle(current_node)

        return {
            "start_gate": start_gate, "end_gate": current_gate,
            "messages": messages,
            "layers_traversed": list(set(layers_traversed)),
            "families_visited": list(set(families_visited)),
            "integration_nodes_used": list(set(integration_nodes_used)),
            "emergent_states": [asdict(e) for e in emergent_states],
            "trigram_path": [{"name": t.name, "operator": t.operator} for t in trigram_path],
            "cycle_switches": cycle_switches
        }

    def _find_nearest_integration(self, node: Dict) -> Optional[Dict]:
        nearest = None
        min_dist = float('inf')

        for ig in INTEGRATION_GATES:
            ig_node = self.nodes.get(f"gate_{ig}")
            if not ig_node:
                continue
            dist = BinaryMorph.karnaugh_distance(node["gate"], ig)
            if dist < min_dist:
                min_dist = dist
                nearest = ig_node

        return nearest

    def _advance_cycle(self, node: Dict):
        cycle_key = node["id"]
        cycle_count = self.cycle_state.get(cycle_key, 0) + 1
        self.cycle_state[cycle_key] = cycle_count

        positions = ["begin", "middle", "end", "transition"]
        node["cycle_position"] = positions[cycle_count % 4]

    def learn(self, morph_result: Dict, user_kept: bool = True):
        self.memory.append({
            "timestamp": datetime.now().isoformat(),
            "start_gate": morph_result.get("start_gate"),
            "end_gate": morph_result.get("end_gate"),
            "layers": morph_result.get("layers_traversed"),
            "families": morph_result.get("families_visited", []),
            "emergent_count": len(morph_result.get("emergent_states", [])),
            "user_kept": user_kept,
            "weight_adjustment": 0.1 if user_kept else -0.05
        })

    def get_stats(self) -> Dict:
        return {
            "total_nodes": len(self.nodes),
            "emergent_states": len(self.emergent_states),
            "cycle_states": dict(self.cycle_state),
            "memory_size": len(self.memory),
            "total_morphs": sum(n["usage_count"] for n in self.nodes.values())
        }

print("Part 1: Core Engine — COMPLETE")
print(f"  - 64 gates initialized")
print(f"  - 8 trigrams loaded")
print(f"  - 5-layer morph protocol ready")
print(f"  - Cycle switching (42/53) active")
print(f"  - Substrate memory enabled")


# ============================================================================
# DISSOLVER
# ============================================================================

class SubstrateDissolver:
    def dissolve(self, content: bytes, name: str, content_type: str) -> DissolvedState:
        activations = []

        # Process in 6-byte windows (hexagram size)
        for i in range(0, len(content), 6):
            window = content[i:i+6]
            if len(window) < 6:
                break

            binary6 = [1 if b > 127 else 0 for b in window]
            val = sum(bit << (5 - idx) for idx, bit in enumerate(binary6))
            gate = (val % 64) + 1

            mean = sum(window) / len(window)
            variance = (sum((b - mean) ** 2 for b in window) / len(window)) ** 0.5
            strength = min(1.0, variance / 128)

            # Base activation (Layer 1: Standard)
            center = self._get_center(gate)
            family = self._get_family(gate)
            circuit = self._get_circuit(center)

            activations.append(GateActivation(
                gate=gate, strength=strength, dialect=Dialect.STANDARD,
                layer=1, byte_position=i, center=center, family=family, circuit=circuit
            ))

            # Apply all 5 dialects
            self._apply_dialects(gate, strength, i, activations)

        # Extract trigrams from strongest activations
        strongest = sorted(activations, key=lambda a: a.strength, reverse=True)[:8]
        trigrams = []
        for act in strongest:
            trigrams.append(BinaryMorph.get_trigram(BinaryMorph.upper_trigram(act.gate)))
            trigrams.append(BinaryMorph.get_trigram(BinaryMorph.lower_trigram(act.gate)))

        # Generate emergent seeds from dialect collisions
        seeds = self._generate_seeds(activations)

        # Build dialect summary
        dialects = []
        for layer in range(1, 6):
            layer_acts = [a for a in activations if a.layer == layer]
            dialects.append({
                "layer": layer,
                "sequence": ["standard", "reversed", "complementary", "karnaugh", "ternary"][layer - 1],
                "gates": list(set(a.gate for a in layer_acts)),
                "operator": ["XOR", "AND", "OR", "XNOR", "NAND"][layer - 1],
                "count": len(layer_acts)
            })

        # Dominant centers and families
        center_counts = {}
        family_counts = {}
        for a in activations:
            center_counts[a.center] = center_counts.get(a.center, 0) + 1
            family_counts[a.family] = family_counts.get(a.family, 0) + 1

        dominant_centers = sorted(center_counts.keys(), key=lambda c: center_counts[c], reverse=True)[:3]
        dominant_families = sorted(family_counts.keys(), key=lambda f: family_counts[f], reverse=True)[:3]

        # Morph vectors (directional flow)
        morph_vectors = self._calculate_morph_vectors(activations)

        return DissolvedState(
            source_name=name, source_type=content_type, source_size=len(content),
            source_hash=hashlib.sha256(content).hexdigest()[:16],
            activations=activations, trigrams=list(dict.fromkeys(trigrams)),
            dialects=dialects, emergent_seeds=seeds,
            dominant_centers=dominant_centers, dominant_families=dominant_families,
            morph_vectors=morph_vectors
        )

    def _apply_dialects(self, gate: int, strength: float, position: int, activations: List[GateActivation]):
        binary = gate - 1

        # Layer 2: Reversed
        rev_binary = int(bin(binary)[2:].zfill(6)[::-1], 2)
        rev_gate = BinaryMorph.binary_to_gate(rev_binary)
        center = self._get_center(rev_gate)
        family = self._get_family(rev_gate)
        circuit = self._get_circuit(center)
        activations.append(GateActivation(
            gate=rev_gate, strength=strength * 0.9, dialect=Dialect.REVERSED,
            layer=2, byte_position=position, center=center, family=family, circuit=circuit
        ))

        # Layer 3: Complementary
        comp_binary = (~binary) & 0x3F
        comp_gate = BinaryMorph.binary_to_gate(comp_binary)
        center = self._get_center(comp_gate)
        family = self._get_family(comp_gate)
        circuit = self._get_circuit(center)
        activations.append(GateActivation(
            gate=comp_gate, strength=strength * 0.8, dialect=Dialect.COMPLEMENTARY,
            layer=3, byte_position=position, center=center, family=family, circuit=circuit
        ))

        # Layer 4: Karnaugh spatial shift
        k = BinaryMorph.gate_to_karnaugh(gate)
        shifted = ((k[0] + 1) % 4, (k[1] + 1) % 4, (k[2] + 1) % 4)
        k_gate = self._karnaugh_to_gate(shifted)
        center = self._get_center(k_gate)
        family = self._get_family(k_gate)
        circuit = self._get_circuit(center)
        activations.append(GateActivation(
            gate=k_gate, strength=strength * 0.7, dialect=Dialect.KARNAUGH,
            layer=4, byte_position=position, center=center, family=family, circuit=circuit
        ))

        # Layer 5: Ternary
        ternary_gate = ((gate * 3) % 64) + 1
        center = self._get_center(ternary_gate)
        family = self._get_family(ternary_gate)
        circuit = self._get_circuit(center)
        activations.append(GateActivation(
            gate=ternary_gate, strength=strength * 0.6, dialect=Dialect.TERNARY,
            layer=5, byte_position=position, center=center, family=family, circuit=circuit
        ))

    def _karnaugh_to_gate(self, k: Tuple[int, int, int]) -> int:
        bits = [k[0] & 1, (k[0] >> 1) & 1, k[1] & 1, (k[1] >> 1) & 1, k[2] & 1, (k[2] >> 1) & 1]
        val = sum(bit << (5 - idx) for idx, bit in enumerate(bits))
        return (val % 64) + 1

    def _generate_seeds(self, activations: List[GateActivation]) -> List[int]:
        by_position = {}
        for a in activations:
            by_position.setdefault(a.byte_position, []).append(a)

        seeds = []
        for pos, acts in by_position.items():
            if len(acts) > 1:
                avg_gate = sum(a.gate for a in acts) // len(acts)
                seeds.append(avg_gate)

        return seeds[:20]

    def _calculate_morph_vectors(self, activations: List[GateActivation]) -> List[Dict]:
        vectors = []
        for i in range(len(activations) - 1):
            a1, a2 = activations[i], activations[i + 1]
            if a1.layer == a2.layer:
                vectors.append({
                    "from_gate": a1.gate, "to_gate": a2.gate,
                    "layer": a1.layer, "dialect": a1.dialect.value,
                    "distance": BinaryMorph.karnaugh_distance(a1.gate, a2.gate),
                    "strength_delta": a2.strength - a1.strength
                })
        return vectors

    def _get_center(self, gate: int) -> Center:
        for center, gates in CENTER_GATES.items():
            if gate in gates:
                return center
        return Center.ROOT

    def _get_family(self, gate: int) -> Family:
        for family, gates in FAMILY_CLUSTERS.items():
            if gate in gates:
                return family
        return Family.INITIATION

    def _get_circuit(self, gate: int) -> Circuit:
        center = self._get_center(gate)
        mapping = {
            Center.HEAD: Circuit.UNDERSTANDING, Center.AJNA: Circuit.UNDERSTANDING,
            Center.THROAT: Circuit.CENTERING, Center.G: Circuit.CENTERING,
            Center.HEART: Circuit.CENTERING, Center.SPLEEN: Circuit.SENSING,
            Center.SOLAR_PLEXUS: Circuit.KNOWING, Center.SACRAL: Circuit.KNOWING,
            Center.ROOT: Circuit.CENTERING
        }
        return mapping.get(center, Circuit.CENTERING)

# ============================================================================
# ANALYZER — Scoring-based, not binary heuristics
# ============================================================================

class SubstrateAnalyzer:
    def analyze(self, dissolved: DissolvedState, raw_text: Optional[str] = None) -> AnalysisResult:
        scores = self._calculate_scores(dissolved, raw_text)

        content_type_name = max(scores, key=scores.get)
        content_type = ContentType(content_type_name) if content_type_name in [ct.value for ct in ContentType] else ContentType.UNKNOWN

        purpose = self._infer_purpose(dissolved.trigrams, scores)
        features = self._detect_features(dissolved, raw_text)
        entry_points = [dissolved.source_name]
        dependencies = self._extract_dependencies(raw_text)
        structure = self._extract_structure(raw_text, content_type)
        confidence = scores.get(content_type_name, 0.5)

        return AnalysisResult(
            content_type=content_type, purpose=purpose, confidence=confidence,
            detected_features=features, entry_points=entry_points,
            dependencies=dependencies, structure=structure, scores=scores
        )

    def _calculate_scores(self, dissolved: DissolvedState, raw_text: Optional[str]) -> Dict[str, float]:
        scores = {ct.value: 0.0 for ct in ContentType}

        # Center-based scoring
        center_weights = {
            Center.HEAD: {"web_app": 0.3, "tool": 0.4, "document": 0.2},
            Center.AJNA: {"web_app": 0.4, "api": 0.3, "data_schema": 0.3},
            Center.THROAT: {"web_app": 0.5, "api": 0.2, "document": 0.2},
            Center.G: {"web_app": 0.3, "config": 0.3, "tool": 0.2},
            Center.HEART: {"game": 0.4, "web_app": 0.2, "media": 0.2},
            Center.SPLEEN: {"web_app": 0.3, "api": 0.3, "tool": 0.2},
            Center.SOLAR_PLEXUS: {"web_app": 0.4, "data_schema": 0.3, "api": 0.2},
            Center.SACRAL: {"game": 0.5, "media": 0.3, "web_app": 0.2},
            Center.ROOT: {"config": 0.4, "tool": 0.3, "document": 0.2}
        }

        for center in dissolved.dominant_centers:
            weights = center_weights.get(center, {})
            for ct, weight in weights.items():
                scores[ct] += weight * 0.3

        # Family-based scoring
        family_weights = {
            Family.INITIATION: {"web_app": 0.3, "tool": 0.3},
            Family.MONEY: {"game": 0.4, "web_app": 0.2},
            Family.MUTATION: {"api": 0.3, "tool": 0.3, "config": 0.2},
            Family.INSPIRATION: {"document": 0.4, "media": 0.3},
            Family.AWAKENING: {"web_app": 0.4, "game": 0.2},
            Family.WAVELENGTH: {"data_schema": 0.4, "api": 0.2},
            Family.TRANSITORINESS: {"config": 0.3, "tool": 0.3}
        }

        for family in dissolved.dominant_families:
            weights = family_weights.get(family, {})
            for ct, weight in weights.items():
                scores[ct] += weight * 0.2

        # Text-based scoring
        if raw_text:
            text_lower = raw_text.lower()

            if "<!doctype html>" in text_lower or "<html" in text_lower:
                scores["web_app"] += 0.5
            if "function" in text_lower or "class" in text_lower or "const" in text_lower:
                scores["web_app"] += 0.3
                scores["tool"] += 0.2
            if "fetch(" in text_lower or "axios" in text_lower:
                scores["api"] += 0.4
            if "canvas" in text_lower or "webgl" in text_lower:
                scores["game"] += 0.4
            if "{" in raw_text and '"' in raw_text and ":" in raw_text:
                scores["config"] += 0.3
                scores["data_schema"] += 0.2
            if "# " in raw_text or "## " in raw_text:
                scores["document"] += 0.4
            if "import " in text_lower or "from " in text_lower:
                scores["web_app"] += 0.2
                scores["tool"] += 0.2

        # File extension scoring
        name_lower = dissolved.source_name.lower()
        ext_scores = {
            ".html": {"web_app": 0.3}, ".js": {"web_app": 0.3, "tool": 0.1},
            ".ts": {"web_app": 0.3, "tool": 0.1}, ".jsx": {"web_app": 0.4},
            ".tsx": {"web_app": 0.4}, ".json": {"config": 0.3, "data_schema": 0.2, "api": 0.1},
            ".py": {"tool": 0.4, "web_app": 0.1}, ".java": {"tool": 0.3},
            ".go": {"tool": 0.3, "api": 0.2}, ".rs": {"tool": 0.3},
            ".md": {"document": 0.4}, ".txt": {"document": 0.3},
            ".zip": {"web_app": 0.2, "game": 0.1, "tool": 0.1}
        }

        for ext, weights in ext_scores.items():
            if name_lower.endswith(ext):
                for ct, weight in weights.items():
                    scores[ct] += weight

        # Normalize
        total = sum(scores.values())
        if total > 0:
            scores = {k: min(1.0, v / total * 2) for k, v in scores.items()}

        return scores

    def _infer_purpose(self, trigrams: List[Trigram], scores: Dict[str, float]) -> Purpose:
        trigram_names = [t.name for t in trigrams]

        purpose_scores = {p.value: 0.0 for p in Purpose}

        if "Li" in trigram_names and "Dui" in trigram_names:
            purpose_scores["display"] += 0.4
        if "Kan" in trigram_names and "Kun" in trigram_names:
            purpose_scores["collect"] += 0.4
        if "Zhen" in trigram_names and "Qian" in trigram_names:
            purpose_scores["transform"] += 0.4
        if "Xun" in trigram_names and "Li" in trigram_names:
            purpose_scores["communicate"] += 0.4
        if "Gen" in trigram_names and "Kun" in trigram_names:
            purpose_scores["organize"] += 0.4

        if scores.get("web_app", 0) > 0.5:
            purpose_scores["display"] += 0.3
        if scores.get("api", 0) > 0.5:
            purpose_scores["communicate"] += 0.3
        if scores.get("game", 0) > 0.5:
            purpose_scores["entertain"] += 0.4
        if scores.get("tool", 0) > 0.5:
            purpose_scores["calculate"] += 0.3
        if scores.get("data_schema", 0) > 0.5:
            purpose_scores["organize"] += 0.3

        purpose_name = max(purpose_scores, key=purpose_scores.get)
        return Purpose(purpose_name) if purpose_name in [p.value for p in Purpose] else Purpose.DISPLAY

    def _detect_features(self, dissolved: DissolvedState, raw_text: Optional[str]) -> List[str]:
        features = []

        center_counts = {}
        for a in dissolved.activations:
            center_counts[a.center] = center_counts.get(a.center, 0) + 1

        if center_counts.get(Center.HEAD, 0) > 5:
            features.append("navigation")
        if center_counts.get(Center.AJNA, 0) > 5:
            features.append("search")
        if center_counts.get(Center.THROAT, 0) > 5:
            features.append("forms")
        if center_counts.get(Center.G, 0) > 5:
            features.append("layout")
        if center_counts.get(Center.HEART, 0) > 5:
            features.append("auth")
        if center_counts.get(Center.SPLEEN, 0) > 5:
            features.append("realtime")
        if center_counts.get(Center.SOLAR_PLEXUS, 0) > 5:
            features.append("data_binding")
        if center_counts.get(Center.SACRAL, 0) > 5:
            features.append("animations")
        if center_counts.get(Center.ROOT, 0) > 5:
            features.append("persistence")

        if raw_text:
            text_lower = raw_text.lower()
            if "fetch(" in text_lower or "axios" in text_lower:
                features.append("api_client")
            if "localstorage" in text_lower or "indexeddb" in text_lower:
                features.append("storage")
            if "canvas" in text_lower or "webgl" in text_lower:
                features.append("graphics")
            if "websocket" in text_lower or "ws://" in text_lower:
                features.append("websocket")
            if "drag" in text_lower or "drop" in text_lower:
                features.append("dnd")
            if "gesture" in text_lower or "touch" in text_lower:
                features.append("mobile")

        return list(set(features))

    def _extract_dependencies(self, raw_text: Optional[str]) -> List[str]:
        if not raw_text:
            return []

        deps = []
        import_matches = raw_text.split("import ")
        for match in import_matches[1:]:
            dep = match.split('from')[0].strip().strip(chr(34)+chr(39)).split('.')[0]
            if dep and not dep.startswith("."):
                deps.append(dep)

        return list(set(deps))

    def _extract_structure(self, raw_text: Optional[str], content_type: ContentType) -> Dict:
        if not raw_text:
            return {}

        if content_type == ContentType.WEB_APP:
            return {
                "has_body": "<body" in raw_text.lower(),
                "has_scripts": "<script" in raw_text.lower(),
                "has_styles": "<style" in raw_text.lower() or "<link" in raw_text.lower(),
                "has_head": "<head" in raw_text.lower()
            }

        try:
            json_data = json.loads(raw_text)
            return {
                "keys": list(json_data.keys())[:10],
                "is_array": isinstance(json_data, list),
                "depth": self._object_depth(json_data)
            }
        except:
            return {"raw_preview": raw_text[:500]}

    def _object_depth(self, obj: Any, depth: int = 0) -> int:
        if not isinstance(obj, dict) or not obj:
            return depth
        return max(depth, max((self._object_depth(v, depth + 1) for v in obj.values()), default=depth))

print("Part 2: Dissolver + Analyzer — COMPLETE")

# ============================================================================
# SYNTHESIZER
# ============================================================================

class ArchitectureSynthesizer:
    def __init__(self, engine):
        self.engine = engine

    def synthesize(self, analysis, dissolved):
        components = self._gates_to_components(dissolved.activations, analysis)
        layout = self._trigrams_to_layout(dissolved.trigrams)
        data_flow = self._circuits_to_data_flow(dissolved.activations)
        entry_component = self._find_entry_component(components)
        injections = self._emergent_to_features(dissolved.emergent_seeds, analysis)
        substrate_graph = self._build_substrate_graph(dissolved, components)
        return AppArchitecture(
            content_type=analysis.content_type, purpose=analysis.purpose,
            components=components, layout=layout, data_flow=data_flow,
            entry_component=entry_component, features=analysis.detected_features,
            emergent_injections=injections, substrate_graph=substrate_graph
        )

    def _gates_to_components(self, activations, analysis):
        components = []
        seen_gates = set()
        sorted_acts = sorted(activations, key=lambda a: a.strength, reverse=True)[:16]
        for act in sorted_acts:
            if act.gate in seen_gates:
                continue
            seen_gates.add(act.gate)
            component_type = self._gate_to_component_type(act.gate, act.center, act.family, analysis)
            components.append(ComponentSpec(
                id=f'comp_{act.gate}_{act.layer}', type=component_type, gate=act.gate,
                center=act.center, purpose=self._center_to_purpose(act.center),
                props=self._generate_props(act.gate, component_type, analysis),
                children=[], logic=self._generate_logic(act.gate, component_type),
                style=self._gate_to_style(act.gate, act.layer), weight=act.strength
            ))
        for comp in components:
            family = self._get_family(comp.gate)
            siblings = [c for c in components if self._get_family(c.gate) == family and c.id != comp.id]
            comp.children = [c.id for c in siblings[:3]]
        return components

    def _gate_to_component_type(self, gate, center, family, analysis):
        type_map = {
            Center.HEAD: {'default': 'navigation', Circuit.UNDERSTANDING: 'search'},
            Center.AJNA: {'default': 'input', Circuit.UNDERSTANDING: 'textarea'},
            Center.THROAT: {'default': 'button', Circuit.CENTERING: 'link'},
            Center.G: {'default': 'container', Circuit.CENTERING: 'layout'},
            Center.HEART: {'default': 'badge', Circuit.CENTERING: 'alert'},
            Center.SPLEEN: {'default': 'list', Circuit.SENSING: 'timeline'},
            Center.SOLAR_PLEXUS: {'default': 'card', Circuit.KNOWING: 'form'},
            Center.SACRAL: {'default': 'media', Circuit.KNOWING: 'animation'},
            Center.ROOT: {'default': 'footer', Circuit.CENTERING: 'header'}
        }
        circuit = self._get_circuit(gate)
        center_map = type_map.get(center, {})
        return center_map.get(circuit, center_map.get('default', 'div'))

    def _center_to_purpose(self, center):
        mapping = {
            Center.HEAD: 'direct', Center.AJNA: 'process', Center.THROAT: 'express',
            Center.G: 'contain', Center.HEART: 'validate', Center.SPLEEN: 'sense',
            Center.SOLAR_PLEXUS: 'feel', Center.SACRAL: 'create', Center.ROOT: 'ground'
        }
        return mapping.get(center, 'contain')

    def _generate_props(self, gate, component_type, analysis):
        props = {
            'id': f'gate_{gate}', 'data_gate': gate,
            'data_center': self._get_center(gate).value,
            'data_family': self._get_family(gate).value
        }
        if component_type == 'button':
            props['variant'] = 'primary' if gate % 2 == 0 else 'secondary'
            props['size'] = 'large' if gate % 3 == 0 else 'medium'
        elif component_type in ['input', 'textarea']:
            props['placeholder'] = HEXAGRAM_NAMES[gate - 1]
            props['required'] = gate % 7 == 0
        elif component_type == 'card':
            props['elevated'] = gate % 4 == 0
            props['collapsible'] = gate % 5 == 0
        elif component_type == 'navigation':
            props['orientation'] = 'horizontal' if gate % 2 == 0 else 'vertical'
            props['sticky'] = gate % 3 == 0
        if 'auth' in analysis.detected_features:
            props['protected'] = True
        if 'realtime' in analysis.detected_features:
            props['live'] = True
        return props

    def _generate_logic(self, gate, component_type):
        logic = []
        upper = BinaryMorph.get_trigram(BinaryMorph.upper_trigram(gate))
        lower = BinaryMorph.get_trigram(BinaryMorph.lower_trigram(gate))
        logic.append(f'// {upper.operator} logic from upper trigram')
        logic.append(f'// {lower.operator} logic from lower trigram')
        if component_type == 'button':
            logic.append(f'on_click: lambda: activate_gate({gate})')
        elif component_type == 'input':
            logic.append('on_change: lambda v: dissolve_input(v)')
        elif component_type == 'card':
            logic.append(f'on_expand: lambda: morph_layer({gate % 5 + 1})')
        return logic

    def _gate_to_style(self, gate, layer):
        colors = ['#ff3333', '#ff8833', '#ffdd33', '#33ff88', '#3388ff']
        base_color = colors[layer - 1] if layer <= 5 else colors[0]
        return {
            '--gate-color': base_color,
            'border-color': base_color + '40',
            'background': f'linear-gradient(135deg, {base_color}10 0%, transparent 100%)'
        }

    def _trigrams_to_layout(self, trigrams):
        names = [t.name for t in trigrams]
        if 'Qian' in names and 'Kun' in names:
            return 'grid'
        if 'Zhen' in names and 'Xun' in names:
            return 'flex'
        if 'Kan' in names and 'Li' in names:
            return 'flow'
        if 'Gen' in names and 'Dui' in names:
            return 'absolute'
        return 'flex'

    def _circuits_to_data_flow(self, activations):
        circuit_counts = {}
        for a in activations:
            circuit_counts[a.circuit] = circuit_counts.get(a.circuit, 0) + 1
        dominant = max(circuit_counts, key=circuit_counts.get) if circuit_counts else Circuit.CENTERING
        flow_map = {
            Circuit.UNDERSTANDING: 'props', Circuit.SENSING: 'events',
            Circuit.KNOWING: 'context', Circuit.CENTERING: 'stream',
            Circuit.INTEGRATION: 'stream'
        }
        return flow_map.get(dominant, 'props')

    def _find_entry_component(self, components):
        head_comp = next((c for c in components if c.center == Center.HEAD), None)
        g_comp = next((c for c in components if c.center == Center.G), None)
        return head_comp.id if head_comp else (g_comp.id if g_comp else (components[0].id if components else 'app'))

    def _emergent_to_features(self, seeds, analysis):
        injections = []
        for seed in seeds[:5]:
            tetragram = (seed % 81) + 1
            if tetragram % 9 == 0:
                injections.append('auto_morph')
            if tetragram % 7 == 0:
                injections.append('voice_input')
            if tetragram % 5 == 0:
                injections.append('gesture_control')
            if tetragram % 4 == 0:
                injections.append('dark_mode')
            if tetragram % 3 == 0:
                injections.append('p2p_sync')
            if tetragram % 2 == 0:
                injections.append('offline_mode')
            if analysis.content_type == ContentType.WEB_APP and tetragram > 40:
                injections.append('pwa_shell')
            if analysis.content_type == ContentType.GAME and tetragram > 60:
                injections.append('procedural_gen')
            if analysis.purpose == Purpose.VISUALIZE and tetragram < 20:
                injections.append('3d_view')
        return list(set(injections))

    def _build_substrate_graph(self, dissolved, components):
        return {
            'nodes': [
                {
                    'id': f'gate_{a.gate}', 'gate': a.gate,
                    'center': a.center.value, 'family': a.family.value,
                    'circuit': a.circuit.value, 'layer': a.layer,
                    'dialect': a.dialect.value, 'strength': a.strength,
                    'byte_position': a.byte_position
                }
                for a in dissolved.activations[:50]
            ],
            'relations': [
                {
                    'from': f'gate_{v["from_gate"]}', 'to': f'gate_{v["to_gate"]}',
                    'layer': v['layer'], 'dialect': v['dialect'],
                    'distance': v['distance'], 'strength_delta': v['strength_delta']
                }
                for v in dissolved.morph_vectors[:30]
            ],
            'components': [
                {
                    'id': c.id, 'type': c.type, 'gate': c.gate,
                    'center': c.center.value, 'weight': c.weight,
                    'children': c.children
                }
                for c in components
            ],
            'dominant_centers': [c.value for c in dissolved.dominant_centers],
            'dominant_families': [f.value for f in dissolved.dominant_families],
            'emergent_seeds': dissolved.emergent_seeds
        }

    def _get_family(self, gate):
        for family, gates in FAMILY_CLUSTERS.items():
            if gate in gates:
                return family
        return Family.INITIATION

    def _get_center(self, gate):
        for center, gates in CENTER_GATES.items():
            if gate in gates:
                return center
        return Center.ROOT

    def _get_circuit(self, gate):
        center = self._get_center(gate)
        mapping = {
            Center.HEAD: Circuit.UNDERSTANDING, Center.AJNA: Circuit.UNDERSTANDING,
            Center.THROAT: Circuit.CENTERING, Center.G: Circuit.CENTERING,
            Center.HEART: Circuit.CENTERING, Center.SPLEEN: Circuit.SENSING,
            Center.SOLAR_PLEXUS: Circuit.KNOWING, Center.SACRAL: Circuit.KNOWING,
            Center.ROOT: Circuit.CENTERING
        }
        return mapping.get(center, Circuit.CENTERING)

# ============================================================================
# EXECUTOR
# ============================================================================

class NativeExecutor:
    def __init__(self, engine):
        self.engine = engine

    def execute(self, architecture, dissolved):
        content_type = architecture.content_type
        if content_type == ContentType.GAME:
            return self._generate_game(architecture, dissolved)
        elif content_type == ContentType.DOCUMENT:
            return self._generate_document(architecture, dissolved)
        elif content_type == ContentType.API:
            return self._generate_api_client(architecture, dissolved)
        elif content_type == ContentType.TOOL:
            return self._generate_tool(architecture, dissolved)
        else:
            return self._generate_web_app(architecture, dissolved)

    def _generate_web_app(self, arch, dissolved):
        artifact_id = str(uuid.uuid4())[:8]
        python_code = self._generate_python_backend(arch, dissolved, artifact_id)
        html_code = self._generate_html_frontend(arch, dissolved, artifact_id)
        js_code = self._generate_js_logic(arch, dissolved, artifact_id)
        css_code = self._generate_styles(arch, dissolved)
        json_config = json.dumps({
            'artifact_id': artifact_id, 'content_type': arch.content_type.value,
            'purpose': arch.purpose.value, 'layout': arch.layout,
            'data_flow': arch.data_flow, 'entry_component': arch.entry_component,
            'features': arch.features, 'emergent_injections': arch.emergent_injections,
            'components': [{'id': c.id, 'type': c.type, 'gate': c.gate, 'center': c.center.value} for c in arch.components],
            'substrate_graph': arch.substrate_graph
        }, indent=2)
        return Artifact(
            id=artifact_id, type='web_app', python=python_code, html=html_code,
            js=js_code, css=css_code, json_config=json_config,
            metadata={
                'generated_at': datetime.now().isoformat(),
                'source': dissolved.source_name,
                'gate_count': len(set(a.gate for a in dissolved.activations)),
                'emergent_states': len(dissolved.emergent_seeds)
            }
        )

    def _generate_python_backend(self, arch, dissolved, artifact_id):
        components_json = json.dumps([{'gate': c.gate, 'center': c.center.value, 'type': c.type} for c in arch.components])
        return f'''#!/usr/bin/env python3
# Morphed Backend — Generated from {dissolved.source_name}
# Artifact ID: {artifact_id}

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
import uuid

app = FastAPI(title='Morphed App — {artifact_id}')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

SUBSTRATE = {{
    'gates': {components_json},
    'features': {json.dumps(arch.features)},
    'emergent': {json.dumps(arch.emergent_injections)},
    'layout': '{arch.layout}'
}}

class MorphRequest(BaseModel):
    input: str
    layer: int = 1

@app.get('/')
def root():
    return {{
        'app': 'Morphed — {artifact_id}',
        'substrate': SUBSTRATE,
        'status': 'active'
    }}

@app.post('/morph')
def morph(request: MorphRequest):
    bytes_input = request.input.encode()
    activations = []
    for i in range(0, len(bytes_input), 6):
        window = bytes_input[i:i+6]
        if len(window) < 6:
            break
        binary6 = [1 if b > 127 else 0 for b in window]
        val = sum(bit << (5 - idx) for idx, bit in enumerate(binary6))
        gate = (val % 64) + 1
        activations.append({{'gate': gate, 'position': i}})
    return {{
        'request_id': str(uuid.uuid4()),
        'activations': activations,
        'layer': request.layer,
        'emergent_count': len(activations) // 3
    }}

@app.get('/gates')
def get_gates():
    return SUBSTRATE['gates']

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
'''

    def _generate_html_frontend(self, arch, dissolved, artifact_id):
        comp_html = '\n'.join([
            f'    <div class="morph-{c.type}" data-gate="{c.gate}" data-center="{c.center.value}">'
            f'      <span class="gate-label">Gate {c.gate}</span>'
            f'      <span class="gate-name">{HEXAGRAM_NAMES[c.gate - 1]}</span>'
            f'    </div>'
            for c in arch.components[:12]
        ])
        return f'''<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='UTF-8'>
  <title>Morphed: {dissolved.source_name}</title>
  <link rel='stylesheet' href='style.css'>
</head>
<body data-layout='{arch.layout}'>
  <div id='app' class='morph-app'>
    <header class='morph-header'>
      <h1>Morphed App</h1>
      <span class='artifact-id'>{artifact_id}</span>
    </header>
    <main class='morph-main'>
{comp_html}
    </main>
    <footer class='morph-footer'>
      <span>Generated from {dissolved.source_name}</span>
      <span id='gate-count'>{len(arch.components)} gates active</span>
    </footer>
  </div>
  <script src='app.js'></script>
</body>
</html>'''

    def _generate_js_logic(self, arch, dissolved, artifact_id):
        injections = arch.emergent_injections
        emergent_code = ''
        if 'auto_morph' in injections:
            emergent_code += '    setInterval(() => {\n      const gates = document.querySelectorAll(\'[data-gate]\');\n      if (gates.length > 0) {\n        const g = gates[Math.floor(Math.random() * gates.length)];\n        g.style.transform = \'scale(1.1)\';\n        setTimeout(() => { g.style.transform = \'\'; }, 300);\n      }\n    }, 5000);\n'
        if 'dark_mode' in injections:
            emergent_code += '    document.addEventListener(\'keydown\', e => {\n      if (e.key === \'d\' && e.ctrlKey) {\n        document.body.classList.toggle(\'dark\');\n      }\n    });\n'
        if 'gesture_control' in injections:
            emergent_code += '    let touchStart = null;\n    document.addEventListener(\'touchstart\', e => { touchStart = e.touches[0]; });\n    document.addEventListener(\'touchend\', e => {\n      if (!touchStart) return;\n      const dx = e.changedTouches[0].clientX - touchStart.clientX;\n      if (Math.abs(dx) > 100) {\n        document.body.setAttribute(\'data-swipe\', dx > 0 ? \'right\' : \'left\');\n      }\n    });\n'
        return f'''// Morphed App — {artifact_id}
const GATE_REGISTRY = {{}};
const MORPH_STATE = {{
  activations: {json.dumps([{'gate': a.gate, 'strength': a.strength} for a in dissolved.activations[:8]])},
  emergent: {json.dumps(dissolved.emergent_seeds[:5])},
  layer: 1
}};

function activateGate(gate) {{
  GATE_REGISTRY[gate] = (GATE_REGISTRY[gate] || 0) + 1;
  MORPH_STATE.layer = (MORPH_STATE.layer % 5) + 1;
  const el = document.querySelector(`[data-gate="${{gate}}"]`);
  if (el) {{
    el.style.borderColor = '#ffdd33';
    setTimeout(() => el.style.borderColor = '', 500);
  }}
}}

function dissolveInput(value) {{
  const bytes = new TextEncoder().encode(value);
  const gates = [];
  for (let i = 0; i < bytes.length; i += 6) {{
    const window = bytes.slice(i, i + 6);
    if (window.length < 6) break;
    const bin = Array.from(window).map(b => b > 127 ? 1 : 0);
    const val = bin.reduce((a, b, idx) => a + (b << (5 - idx)), 0);
    gates.push((val % 64) + 1);
  }}
  gates.forEach(g => activateGate(g));
}}

document.addEventListener('click', e => {{
  const gate = e.target.closest('[data-gate]');
  if (gate) activateGate(parseInt(gate.dataset.gate));
}});

{emergent_code}

console.log('Morphed App — {artifact_id} — initialized');
'''

    def _generate_styles(self, arch, dissolved):
        layout_css = {
            'grid': 'display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 12px;',
            'flex': 'display: flex; flex-wrap: wrap; gap: 12px;',
            'flow': 'display: block;',
            'absolute': 'position: relative;'
        }.get(arch.layout, 'display: flex; flex-wrap: wrap; gap: 12px;')
        return f''':root {{
  --bg: #0a0a0f; --fg: #e0e0e0; --muted: #666;
  --l1: #ff3333; --l2: #ff8833; --l3: #ffdd33; --l4: #33ff88; --l5: #3388ff;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--fg); font-family: 'Courier New', monospace; min-height: 100vh; }}
body.dark {{ filter: invert(1) hue-rotate(180deg); }}
.morph-app {{ {layout_css} padding: 20px; }}
.morph-header {{ width: 100%; padding: 15px; background: rgba(255,255,255,0.02); border-bottom: 1px solid #333; display: flex; justify-content: space-between; }}
.morph-header h1 {{ font-size: 1.2em; color: var(--l3); }}
.artifact-id {{ font-size: 0.8em; color: var(--muted); }}
.morph-main {{ {layout_css} flex: 1; }}
.morph-footer {{ width: 100%; padding: 15px; background: rgba(255,255,255,0.02); border-top: 1px solid #333; display: flex; justify-content: space-between; }}
.morph-button {{ background: linear-gradient(135deg, var(--l1)20, transparent); border: 1px solid #333; border-radius: 6px; padding: 8px 16px; cursor: pointer; color: var(--fg); font-family: inherit; transition: all 0.2s; }}
.morph-button:hover {{ border-color: var(--l3); transform: translateY(-2px); }}
.morph-input {{ background: rgba(255,255,255,0.05); border: 1px solid #333; border-radius: 6px; padding: 10px; color: var(--fg); font-family: inherit; width: 100%; }}
.morph-card {{ background: rgba(255,255,255,0.03); border: 1px solid #333; border-radius: 8px; padding: 15px; transition: all 0.2s; position: relative; }}
.morph-card:hover {{ border-color: var(--l3); transform: translateY(-2px); }}
.morph-card::before {{ content: attr(data-gate); position: absolute; top: -8px; right: 8px; background: var(--bg); padding: 0 6px; font-size: 0.7em; color: var(--muted); }}
.gate-label {{ font-size: 0.8em; color: var(--l3); }}
.gate-name {{ display: block; font-size: 0.7em; color: var(--muted); margin-top: 4px; }}
'''

    def _generate_game(self, arch, dissolved):
        artifact_id = str(uuid.uuid4())[:8]
        gates = list(set(a.gate for a in dissolved.activations if a.layer == 1))[:8]
        entities_js = '\n'.join([
            f'entities.push({{ x: {150 + (i % 4) * 150}, y: {150 + (i // 4) * 150}, gate: {g}, layer: {(i % 5) + 1}, color: colors[{i % 5}], size: {25 + (g % 20)}, vx: {(hash(str(g)) % 10 - 5) / 5}, vy: {(hash(str(g + 1)) % 10 - 5) / 5} }});'
            for i, g in enumerate(gates)
        ])
        html = f'''<!DOCTYPE html>
<html>
<head><title>Game: {dissolved.source_name}</title>
<style>body {{ background: #0a0a0f; color: #e0e0e0; font-family: monospace; margin: 0; overflow: hidden; }}
canvas {{ display: block; }}
#ui {{ position: fixed; top: 10px; left: 10px; background: rgba(0,0,0,0.7); padding: 10px; border-radius: 8px; }}</style>
</head>
<body>
<canvas id='game'></canvas>
<div id='ui'><div>Score: <span id='score'>0</span></div><div>Gate: <span id='gate'>1</span></div></div>
<script>
const canvas = document.getElementById('game');
const ctx = canvas.getContext('2d');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
let score = 0, currentGate = 1;
const entities = [];
const colors = ['#ff3333', '#ff8833', '#ffdd33', '#33ff88', '#3388ff'];
{entities_js}
function gameLoop() {{
  ctx.fillStyle = '#0a0a0f';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  entities.forEach(e => {{
    e.x += e.vx; e.y += e.vy;
    if (e.x < e.size || e.x > canvas.width - e.size) e.vx *= -1;
    if (e.y < e.size || e.y > canvas.height - e.size) e.vy *= -1;
    ctx.fillStyle = e.color;
    ctx.beginPath();
    ctx.arc(e.x, e.y, e.size, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.font = '14px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(e.gate, e.x, e.y + 5);
  }});
  requestAnimationFrame(gameLoop);
}}
canvas.addEventListener('click', e => {{
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  entities.forEach(ent => {{
    const dist = Math.hypot(ent.x - x, ent.y - y);
    if (dist < ent.size) {{
      score += ent.gate;
      currentGate = ent.gate;
      document.getElementById('score').textContent = score;
      document.getElementById('gate').textContent = currentGate;
      ent.size *= 1.1;
      ent.layer = (ent.layer % 5) + 1;
      ent.color = colors[ent.layer - 1];
    }}
  }});
}});
gameLoop();
</script>
</body>
</html>'''
        return Artifact(
            id=artifact_id, type='game', python=None, html=html,
            js=None, css=None, json_config=None,
            metadata={'generated_at': datetime.now().isoformat(), 'source': dissolved.source_name}
        )

    def _generate_document(self, arch, dissolved):
        return self._generate_web_app(arch, dissolved)

    def _generate_api_client(self, arch, dissolved):
        return self._generate_web_app(arch, dissolved)

    def _generate_tool(self, arch, dissolved):
        return self._generate_web_app(arch, dissolved)

# ============================================================================
# FASTAPI SERVER
# ============================================================================

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

app = FastAPI(title='Morphing Net — Universal Execution Engine')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

engine = MorphingNetEngine()
dissolver = SubstrateDissolver()
analyzer = SubstrateAnalyzer()
synthesizer = ArchitectureSynthesizer(engine)
executor = NativeExecutor(engine)

@app.get('/')
def root():
    return {
        'engine': 'Morphing Net v6.0',
        'gates': 64, 'centers': 9, 'layers': 5,
        'dialects': ['standard', 'reversed', 'complementary', 'karnaugh', 'ternary'],
        'status': 'active',
        'stats': engine.get_stats()
    }

@app.post('/morph')
async def morph_endpoint(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    content_type_hint: Optional[str] = Form(None)
):
    start_time = time.time()
    try:
        if file:
            content = await file.read()
            name = file.filename
            content_type = file.content_type or 'application/octet-stream'
        elif text:
            content = text.encode('utf-8')
            name = 'pasted_text.txt'
            content_type = 'text/plain'
        else:
            raise HTTPException(status_code=400, detail='No file or text provided')
        
        dissolved = dissolver.dissolve(content, name, content_type)
        raw_text = content.decode('utf-8', errors='ignore') if content_type.startswith('text/') else None
        analysis = analyzer.analyze(dissolved, raw_text)
        architecture = synthesizer.synthesize(analysis, dissolved)
        artifact = executor.execute(architecture, dissolved)
        
        response = MorphResponse(
            request_id=str(uuid.uuid4()),
            status='success',
            dissolved={
                'source': {'name': dissolved.source_name, 'type': dissolved.source_type, 'size': dissolved.source_size},
                'activations_count': len(dissolved.activations),
                'unique_gates': len(set(a.gate for a in dissolved.activations)),
                'emergent_seeds': len(dissolved.emergent_seeds),
                'dominant_centers': [c.value for c in dissolved.dominant_centers],
                'dominant_families': [f.value for f in dissolved.dominant_families],
                'dialects': dissolved.dialects
            },
            analysis={
                'content_type': analysis.content_type.value,
                'purpose': analysis.purpose.value,
                'confidence': analysis.confidence,
                'features': analysis.detected_features,
                'scores': analysis.scores
            },
            synthesis={
                'layout': architecture.layout,
                'data_flow': architecture.data_flow,
                'entry_component': architecture.entry_component,
                'components_count': len(architecture.components),
                'emergent_injections': architecture.emergent_injections,
                'substrate_graph': architecture.substrate_graph
            },
            artifact={
                'id': artifact.id,
                'type': artifact.type,
                'python': artifact.python,
                'html': artifact.html,
                'js': artifact.js,
                'css': artifact.css,
                'json_config': artifact.json_config,
                'metadata': artifact.metadata
            },
            supabase_ref=None,
            neo4j_ref=None,
            execution_time_ms=int((time.time() - start_time) * 1000)
        )
        
        engine.learn({
            'start_gate': dissolved.activations[0].gate if dissolved.activations else 1,
            'end_gate': dissolved.activations[-1].gate if dissolved.activations else 64,
            'layers_traversed': list(set(a.layer for a in dissolved.activations)),
            'families_visited': list(set(a.family.value for a in dissolved.activations)),
            'emergent_states': dissolved.emergent_seeds
        })
        
        return JSONResponse(content=asdict(response))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/stats')
def get_stats():
    return engine.get_stats()

@app.get('/memory')
def get_memory():
    return {'memory': engine.memory}

@app.post('/remorph')
def remorph_endpoint(artifact_id: str = Form(...)):
    return {'status': 'not_implemented', 'message': 'Recursive remorphing requires persistent storage'}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)

print('Part 3: Synthesizer + Executor + FastAPI Server — COMPLETE')