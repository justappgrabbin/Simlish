// Transcribed from the supplied master document; no inferred dimension assignments.
export const SYNTAX_SOURCE = {"driveId": "1x7xzzzaxg_SdhHM5uteiZ21kGsoRfdK5", "section": "Unified Syntax Field (Full Integration)"};
export const SYNTAX_OPERATORS = Object.freeze([
  {
    "mark": "•",
    "name": "Singularity",
    "linguisticEquivalent": "a / one / this potential",
    "action": "Pre-collapse seed; infinite potential",
    "sourceLine": 42
  },
  {
    "mark": ".",
    "name": "Transitioner",
    "linguisticEquivalent": "then / so / thus",
    "action": "Step inward; descent into next chamber of sequence",
    "sourceLine": 43
  },
  {
    "mark": "°",
    "name": "Collapse",
    "linguisticEquivalent": "at / in / to",
    "action": "Anchors potential into coordinate; fixes a phase or state",
    "sourceLine": 44
  },
  {
    "mark": ":",
    "name": "Portal",
    "linguisticEquivalent": "as / into / through",
    "action": "Opens a threshold; passes between chambers or modes",
    "sourceLine": 45
  },
  {
    "mark": ";",
    "name": "Fork",
    "linguisticEquivalent": "or / while / meanwhile",
    "action": "Divides into divergent streams; creates parallel threads",
    "sourceLine": 46
  },
  {
    "mark": ",",
    "name": "Breath",
    "linguisticEquivalent": "and / with",
    "action": "Pause for collection; gathers fragments before continuation",
    "sourceLine": 47
  },
  {
    "mark": "–",
    "name": "Current",
    "linguisticEquivalent": "is / are / be",
    "action": "Span of continuity; hum of existence; steady-state being",
    "sourceLine": 48
  },
  {
    "mark": "′",
    "name": "Pulse",
    "linguisticEquivalent": "by / per",
    "action": "Rhythmic tick; marks tempo and cadence of events",
    "sourceLine": 49
  },
  {
    "mark": "″",
    "name": "Flicker",
    "linguisticEquivalent": "just / yet",
    "action": "Micro-shift or instant transition; shimmer of awareness",
    "sourceLine": 50
  },
  {
    "mark": "“ ”",
    "name": "Container",
    "linguisticEquivalent": "the name of / known as",
    "action": "Sacred seal of identity; holds a defined essence",
    "sourceLine": 51
  },
  {
    "mark": "( )",
    "name": "Cocoon",
    "linguisticEquivalent": "within / becoming / under",
    "action": "Incubation bubble; protects metamorphosis",
    "sourceLine": 52
  },
  {
    "mark": "[ ]",
    "name": "Index Gate",
    "linguisticEquivalent": "from / where / that which",
    "action": "Lookup key; recalls memory or source reference",
    "sourceLine": 53
  },
  {
    "mark": "{ }",
    "name": "Domain",
    "linguisticEquivalent": "of / about / regarding",
    "action": "Field scope; defines local laws or logic boundaries",
    "sourceLine": 54
  },
  {
    "mark": "/",
    "name": "Blade",
    "linguisticEquivalent": "or else / versus / divide",
    "action": "Cut, separation, or decisive differentiation",
    "sourceLine": 55
  },
  {
    "mark": "\\",
    "name": "Escape",
    "linguisticEquivalent": "except / beyond / aside from",
    "action": "Sideways exit from a pattern; loophole in structure",
    "sourceLine": 56
  },
  {
    "mark": "*",
    "name": "Starburst",
    "linguisticEquivalent": "and also / plus / each",
    "action": "Expansion, multiplication; radiating potential",
    "sourceLine": 57
  },
  {
    "mark": "…",
    "name": "Continuation",
    "linguisticEquivalent": "and then / still / to be continued",
    "action": "Unfinished wave; temporal openness",
    "sourceLine": 58
  },
  {
    "mark": "=",
    "name": "Mirror",
    "linguisticEquivalent": "the / this / that same",
    "action": "Reflective unification; collapse of two into one known instance",
    "sourceLine": 59
  },
  {"mark":"→","name":"Vector","linguisticEquivalent":"becomes / leads to / creates","action":"Direction of transformation; flow of creation","sourceSection":"T12_unified_syntax_field, mark 19, GGM L178","sourceArchive":"Synthia-State-Space-Foundation/vendor/browser/docs/FRAGMENT_ALGEBRA_SPEC.md"}
]);
export function resolveSyntaxMarks(marks) {
  return marks.map(mark => {
    const definition = SYNTAX_OPERATORS.find(item=>item.mark===mark);
    if (!definition) throw Error(`No source definition for syntax mark: ${mark}`);
    return {...definition, source:SYNTAX_SOURCE};
  });
}
