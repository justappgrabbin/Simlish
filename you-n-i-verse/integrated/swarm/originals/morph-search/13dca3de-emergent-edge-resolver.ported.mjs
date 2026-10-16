/**
 * GENERATED FILE — mechanical TS->JS type-erasure transpile (esbuild transform, no logic changes).
 * Source: emergent-edge-resolver.ts (canonical original in this directory)
 * Generated: 2026-09-22, esbuild 0.27.7. Adapter change: relative import specifiers rewritten to generated .ported.mjs filenames.
 */
function checkCompatibility(source, target) {
  const reasons = [];
  let score = 0;
  const harmonicGate = (source.micro.gate + 32) % 64;
  const isHarmonic = target.micro.gate === harmonicGate;
  const isComplementary = Math.abs(source.micro.gate - target.micro.gate) === 1;
  if (isHarmonic) {
    score += 0.4;
    reasons.push("harmonic_gate");
  } else if (isComplementary) {
    score += 0.2;
    reasons.push("complementary_gate");
  }
  const phaseDiff = Math.abs(source.phase - target.phase);
  const phaseAligned = phaseDiff < Math.PI / 4 || phaseDiff > 7 * Math.PI / 4;
  const phaseOpposite = Math.abs(phaseDiff - Math.PI) < Math.PI / 4;
  if (phaseAligned) {
    score += 0.3;
    reasons.push("phase_aligned");
  } else if (phaseOpposite) {
    score += 0.15;
    reasons.push("phase_opposite");
  }
  const dimDiff = Math.abs(source.dimension - target.dimension);
  if (dimDiff === 0) {
    score += 0.2;
    reasons.push("same_dimension");
  } else if (dimDiff === 1) {
    score += 0.1;
    reasons.push("adjacent_dimension");
  }
  const minCoherence = 0.2;
  if (source.coherence > minCoherence && target.coherence > minCoherence) {
    score += 0.1;
    reasons.push("coherence_threshold");
  }
  const zodiacElements = [
    "Fire",
    "Earth",
    "Air",
    "Water",
    "Fire",
    "Earth",
    "Air",
    "Water",
    "Fire",
    "Earth",
    "Air",
    "Water"
  ];
  const sourceElement = zodiacElements[source.macro.zodiac % 12];
  const targetElement = zodiacElements[target.macro.zodiac % 12];
  if (sourceElement === targetElement) {
    score += 0.1;
    reasons.push("same_element");
  }
  const compatible = score >= 0.5;
  return { compatible, score, reasons };
}
const EDGE_FUNCTIONS = [
  {
    type: "INSPIRATION",
    name: "Inspiration",
    description: "Creative spark between unique self and contribution",
    gateA: 0,
    gateB: 7,
    compatible: (s, t) => s.micro.gate === 0 && t.micro.gate === 7 || s.micro.gate === 7 && t.micro.gate === 0,
    execute: (s, t) => {
      const emergent = {
        type: "INSPIRATION",
        coherence: (s.coherence + t.coherence) / 2,
        resonance: 7.83 * (1 + s.amplitude * t.amplitude),
        tension: Math.abs(s.phase - t.phase) / (2 * Math.PI),
        amplification: s.amplitude * t.amplitude,
        trace: `Inspiration: Gate ${s.micro.gate + 1} sparks Gate ${t.micro.gate + 1} into creative expression`
      };
      return {
        source: { ...s, amplitude: s.amplitude * 1.1, phase: s.phase + 0.1 },
        target: { ...t, amplitude: t.amplitude * 1.2, phase: t.phase + 0.2 },
        emergent
      };
    }
  },
  {
    type: "BEAT",
    name: "Beat",
    description: "Rhythmic pulse of life force direction",
    gateA: 1,
    gateB: 13,
    compatible: (s, t) => s.micro.gate === 1 && t.micro.gate === 13 || s.micro.gate === 13 && t.micro.gate === 1,
    execute: (s, t) => {
      const avgPhase = (s.phase + t.phase) / 2;
      const emergent = {
        type: "BEAT",
        coherence: Math.cos(s.phase - t.phase),
        resonance: 7.83 * 2,
        // Double Schumann
        tension: 0,
        amplification: Math.cos(avgPhase),
        trace: `Beat: Gates ${s.micro.gate + 1} and ${t.micro.gate + 1} synchronize into rhythmic pulse`
      };
      return {
        source: { ...s, phase: avgPhase, amplitude: s.amplitude * 0.95 },
        target: { ...t, phase: avgPhase, amplitude: t.amplitude * 0.95 },
        emergent
      };
    }
  },
  {
    type: "MUTATION",
    name: "Mutation",
    description: "Ordered chaos, genetic transformation",
    gateA: 2,
    gateB: 59,
    compatible: (s, t) => s.micro.gate === 2 && t.micro.gate === 59 || s.micro.gate === 59 && t.micro.gate === 2,
    execute: (s, t) => {
      const mutation = Math.random() < 0.5 ? s : t;
      const emergent = {
        type: "MUTATION",
        coherence: Math.min(s.coherence, t.coherence) * 0.8,
        resonance: 7.83 * (1 + Math.random()),
        tension: 0.3,
        amplification: 0,
        trace: `Mutation: Gates ${s.micro.gate + 1} and ${t.micro.gate + 1} produce chaotic transformation`
      };
      return {
        source: { ...s, amplitude: mutation === s ? s.amplitude * 1.2 : s.amplitude * 0.8 },
        target: { ...t, amplitude: mutation === t ? t.amplitude * 1.2 : t.amplitude * 0.8 },
        emergent
      };
    }
  },
  {
    type: "LOGIC",
    name: "Logic",
    description: "Pattern recognition and doubt resolution",
    gateA: 3,
    gateB: 62,
    compatible: (s, t) => s.micro.gate === 3 && t.micro.gate === 62 || s.micro.gate === 62 && t.micro.gate === 3,
    execute: (s, t) => {
      const logicalAnd = s.amplitude > 0.5 && t.amplitude > 0.5;
      const emergent = {
        type: "LOGIC",
        coherence: logicalAnd ? 1 : 0,
        resonance: 7.83 * (logicalAnd ? 2 : 0.5),
        tension: logicalAnd ? 0 : 0.8,
        amplification: logicalAnd ? 1 : 0,
        trace: `Logic: Gates ${s.micro.gate + 1} and ${t.micro.gate + 1} ${logicalAnd ? "resolve" : "fail"} pattern recognition`
      };
      return {
        source: { ...s, coherence: logicalAnd ? 1 : 0.2 },
        target: { ...t, coherence: logicalAnd ? 1 : 0.2 },
        emergent
      };
    }
  },
  {
    type: "RHYTHM",
    name: "Rhythm",
    description: "Cyclic flow with moderated extremes",
    gateA: 4,
    gateB: 14,
    compatible: (s, t) => s.micro.gate === 4 && t.micro.gate === 14 || s.micro.gate === 14 && t.micro.gate === 4,
    execute: (s, t) => {
      const cycle = (s.phase + t.phase) % (2 * Math.PI);
      const emergent = {
        type: "RHYTHM",
        coherence: Math.abs(Math.cos(cycle)),
        resonance: 7.83 * (1 + Math.sin(cycle)),
        tension: Math.abs(Math.sin(cycle)),
        amplification: Math.cos(cycle),
        trace: `Rhythm: Gates ${s.micro.gate + 1} and ${t.micro.gate + 1} establish cyclic flow at ${cycle.toFixed(2)} rad`
      };
      return {
        source: { ...s, phase: cycle },
        target: { ...t, phase: cycle },
        emergent
      };
    }
  },
  {
    type: "INTIMACY",
    name: "Intimacy",
    description: "Emotional bonding and sexual chemistry",
    gateA: 5,
    gateB: 58,
    compatible: (s, t) => s.micro.gate === 5 && t.micro.gate === 58 || s.micro.gate === 58 && t.micro.gate === 5,
    execute: (s, t) => {
      const chemistry = s.amplitude * t.amplitude * Math.cos(s.phase - t.phase);
      const emergent = {
        type: "INTIMACY",
        coherence: Math.abs(chemistry),
        resonance: 7.83 * (1 + Math.abs(chemistry)),
        tension: chemistry < 0 ? Math.abs(chemistry) : 0,
        amplification: chemistry > 0 ? chemistry : 0,
        trace: `Intimacy: Gates ${s.micro.gate + 1} and ${t.micro.gate + 1} ${chemistry > 0 ? "bond" : "repel"} with chemistry ${chemistry.toFixed(3)}`
      };
      return {
        source: { ...s, amplitude: s.amplitude * (1 + chemistry * 0.2) },
        target: { ...t, amplitude: t.amplitude * (1 + chemistry * 0.2) },
        emergent
      };
    }
  },
  {
    type: "ALPHA",
    name: "Alpha",
    description: "Leadership and collective direction",
    gateA: 6,
    gateB: 30,
    compatible: (s, t) => s.micro.gate === 6 && t.micro.gate === 30 || s.micro.gate === 30 && t.micro.gate === 6,
    execute: (s, t) => {
      const leader = s.amplitude > t.amplitude ? s : t;
      const follower = s.amplitude > t.amplitude ? t : s;
      const emergent = {
        type: "ALPHA",
        coherence: leader.coherence,
        resonance: 7.83 * leader.amplitude,
        tension: 0,
        amplification: leader.amplitude,
        trace: `Alpha: Gate ${leader.micro.gate + 1} leads Gate ${follower.micro.gate + 1}`
      };
      return {
        source: { ...s, amplitude: s.amplitude === leader.amplitude ? s.amplitude : s.amplitude * 0.9 },
        target: { ...t, amplitude: t.amplitude === leader.amplitude ? t.amplitude : t.amplitude * 0.9 },
        emergent
      };
    }
  },
  {
    type: "CONCENTRATION",
    name: "Concentration",
    description: "Focused attention through stillness",
    gateA: 8,
    gateB: 51,
    compatible: (s, t) => s.micro.gate === 8 && t.micro.gate === 51 || s.micro.gate === 51 && t.micro.gate === 8,
    execute: (s, t) => {
      const focus = (s.amplitude + t.amplitude) / 2;
      const emergent = {
        type: "CONCENTRATION",
        coherence: focus,
        resonance: 7.83 * focus,
        tension: 1 - focus,
        amplification: focus * focus,
        trace: `Concentration: Gates ${s.micro.gate + 1} and ${t.micro.gate + 1} focus to ${focus.toFixed(3)}`
      };
      return {
        source: { ...s, amplitude: focus, phase: 0 },
        target: { ...t, amplitude: focus, phase: 0 },
        emergent
      };
    }
  },
  {
    type: "AWAKENING",
    name: "Awakening",
    description: "Self-awareness in the present moment",
    gateA: 9,
    gateB: 19,
    compatible: (s, t) => s.micro.gate === 9 && t.micro.gate === 19 || s.micro.gate === 19 && t.micro.gate === 9,
    execute: (s, t) => {
      const now = (s.amplitude + t.amplitude) / 2;
      const emergent = {
        type: "AWAKENING",
        coherence: now,
        resonance: 7.83 * (1 + now),
        tension: 0,
        amplification: now,
        trace: `Awakening: Gates ${s.micro.gate + 1} and ${t.micro.gate + 1} awaken to ${now.toFixed(3)}`
      };
      return {
        source: { ...s, amplitude: now, coherence: 1 },
        target: { ...t, amplitude: now, coherence: 1 },
        emergent
      };
    }
  },
  {
    type: "CURIOSITY",
    name: "Curiosity",
    description: "Idea seeking and stimulation drive",
    gateA: 10,
    gateB: 55,
    compatible: (s, t) => s.micro.gate === 10 && t.micro.gate === 55 || s.micro.gate === 55 && t.micro.gate === 10,
    execute: (s, t) => {
      const curiosity = Math.abs(s.phase - t.phase) / (2 * Math.PI);
      const emergent = {
        type: "CURIOSITY",
        coherence: curiosity,
        resonance: 7.83 * (1 + curiosity),
        tension: 0,
        amplification: curiosity,
        trace: `Curiosity: Gates ${s.micro.gate + 1} and ${t.micro.gate + 1} diverge by ${curiosity.toFixed(3)}`
      };
      return {
        source: { ...s, phase: s.phase + curiosity * 0.1 },
        target: { ...t, phase: t.phase - curiosity * 0.1 },
        emergent
      };
    }
  },
  {
    type: "OPENNESS",
    name: "Openness",
    description: "Emotional grace with social caution",
    gateA: 11,
    gateB: 21,
    compatible: (s, t) => s.micro.gate === 11 && t.micro.gate === 21 || s.micro.gate === 21 && t.micro.gate === 11,
    execute: (s, t) => {
      const grace = Math.min(s.amplitude, t.amplitude);
      const caution = 1 - Math.max(s.amplitude, t.amplitude);
      const emergent = {
        type: "OPENNESS",
        coherence: grace * caution,
        resonance: 7.83 * grace,
        tension: caution,
        amplification: grace,
        trace: `Openness: Gates ${s.micro.gate + 1} and ${t.micro.gate + 1} balance grace ${grace.toFixed(3)} with caution ${caution.toFixed(3)}`
      };
      return {
        source: { ...s, amplitude: grace },
        target: { ...t, amplitude: grace },
        emergent
      };
    }
  },
  {
    type: "PRODIGAL",
    name: "Prodigal",
    description: "Secret keeping with privacy cycles",
    gateA: 12,
    gateB: 32,
    compatible: (s, t) => s.micro.gate === 12 && t.micro.gate === 32 || s.micro.gate === 32 && t.micro.gate === 12,
    execute: (s, t) => {
      const secret = s.amplitude * t.amplitude * 0.5;
      const emergent = {
        type: "PRODIGAL",
        coherence: secret,
        resonance: 7.83 * (1 - secret),
        tension: secret,
        amplification: 0,
        trace: `Prodigal: Gates ${s.micro.gate + 1} and ${t.micro.gate + 1} conceal ${secret.toFixed(3)}`
      };
      return {
        source: { ...s, amplitude: s.amplitude - secret },
        target: { ...t, amplitude: t.amplitude - secret },
        emergent
      };
    }
  },
  {
    type: "DEPTH",
    name: "Depth",
    description: "Skill mastery through talent recognition",
    gateA: 15,
    gateB: 47,
    compatible: (s, t) => s.micro.gate === 15 && t.micro.gate === 47 || s.micro.gate === 47 && t.micro.gate === 15,
    execute: (s, t) => {
      const depth = Math.sqrt(s.amplitude * s.coherence * t.amplitude * t.coherence);
      const emergent = {
        type: "DEPTH",
        coherence: depth,
        resonance: 7.83 * (1 + depth),
        tension: 0,
        amplification: depth,
        trace: `Depth: Gates ${s.micro.gate + 1} and ${t.micro.gate + 1} reach ${depth.toFixed(3)}`
      };
      return {
        source: { ...s, coherence: depth },
        target: { ...t, coherence: depth },
        emergent
      };
    }
  },
  {
    type: "ACCEPTANCE",
    name: "Acceptance",
    description: "Opinion formation with detail focus",
    gateA: 16,
    gateB: 61,
    compatible: (s, t) => s.micro.gate === 16 && t.micro.gate === 61 || s.micro.gate === 61 && t.micro.gate === 16,
    execute: (s, t) => {
      const opinion = (s.phase + t.phase) / 2;
      const detail = Math.abs(s.phase - t.phase);
      const emergent = {
        type: "ACCEPTANCE",
        coherence: 1 - detail / (2 * Math.PI),
        resonance: 7.83 * (1 + Math.cos(opinion)),
        tension: detail / (2 * Math.PI),
        amplification: Math.cos(opinion),
        trace: `Acceptance: Gates ${s.micro.gate + 1} and ${t.micro.gate + 1} form opinion at ${opinion.toFixed(3)}`
      };
      return {
        source: { ...s, phase: opinion },
        target: { ...t, phase: opinion },
        emergent
      };
    }
  },
  {
    type: "JUDGEMENT",
    name: "Judgement",
    description: "Pattern correction with vitality discernment",
    gateA: 17,
    gateB: 57,
    compatible: (s, t) => s.micro.gate === 17 && t.micro.gate === 57 || s.micro.gate === 57 && t.micro.gate === 17,
    execute: (s, t) => {
      const correction = s.coherence > t.coherence ? s : t;
      const target = s.coherence > t.coherence ? t : s;
      const emergent = {
        type: "JUDGEMENT",
        coherence: correction.coherence,
        resonance: 7.83 * correction.amplitude,
        tension: 1 - target.coherence,
        amplification: correction.coherence - target.coherence,
        trace: `Judgement: Gate ${correction.micro.gate + 1} corrects Gate ${target.micro.gate + 1}`
      };
      return {
        source: { ...s, coherence: s.coherence === correction.coherence ? s.coherence : target.coherence + 0.1 },
        target: { ...t, coherence: t.coherence === correction.coherence ? t.coherence : target.coherence + 0.1 },
        emergent
      };
    }
  },
  {
    type: "SYNTHESIS",
    name: "Synthesis",
    description: "Wanting and rejection into resource synthesis",
    gateA: 18,
    gateB: 48,
    compatible: (s, t) => s.micro.gate === 18 && t.micro.gate === 48 || s.micro.gate === 48 && t.micro.gate === 18,
    execute: (s, t) => {
      const want = s.amplitude;
      const reject = 1 - t.amplitude;
      const synthesis = want * reject;
      const emergent = {
        type: "SYNTHESIS",
        coherence: synthesis,
        resonance: 7.83 * synthesis,
        tension: reject,
        amplification: want,
        trace: `Synthesis: Gates ${s.micro.gate + 1} wants ${want.toFixed(3)}, Gate ${t.micro.gate + 1} rejects ${reject.toFixed(3)} \u2192 ${synthesis.toFixed(3)}`
      };
      return {
        source: { ...s, amplitude: want },
        target: { ...t, amplitude: reject },
        emergent
      };
    }
  },
  {
    type: "MONEY",
    name: "Money",
    description: "Material control and gathering power",
    gateA: 20,
    gateB: 44,
    compatible: (s, t) => s.micro.gate === 20 && t.micro.gate === 44 || s.micro.gate === 44 && t.micro.gate === 20,
    execute: (s, t) => {
      const control = s.amplitude * (1 - t.amplitude);
      const gather = s.amplitude + t.amplitude;
      const emergent = {
        type: "MONEY",
        coherence: control,
        resonance: 7.83 * gather,
        tension: 1 - control,
        amplification: gather,
        trace: `Money: Gates ${s.micro.gate + 1} controls ${control.toFixed(3)}, gathers ${gather.toFixed(3)}`
      };
      return {
        source: { ...s, amplitude: control },
        target: { ...t, amplitude: gather },
        emergent
      };
    }
  },
  {
    type: "STRUGGLE",
    name: "Struggle",
    description: "Risk and perseverance for purpose finding",
    gateA: 27,
    gateB: 37,
    compatible: (s, t) => s.micro.gate === 27 && t.micro.gate === 37 || s.micro.gate === 37 && t.micro.gate === 27,
    execute: (s, t) => {
      const risk = Math.max(s.amplitude, t.amplitude);
      const persevere = Math.min(s.coherence, t.coherence);
      const purpose = risk * persevere;
      const emergent = {
        type: "STRUGGLE",
        coherence: persevere,
        resonance: 7.83 * risk,
        tension: 1 - persevere,
        amplification: purpose,
        trace: `Struggle: Gates ${s.micro.gate + 1} and ${t.micro.gate + 1} risk ${risk.toFixed(3)} to persevere ${persevere.toFixed(3)} for purpose ${purpose.toFixed(3)}`
      };
      return {
        source: { ...s, amplitude: risk },
        target: { ...t, amplitude: persevere },
        emergent
      };
    }
  },
  {
    type: "RECOGNITION",
    name: "Recognition",
    description: "Commitment and desire for experiential depth",
    gateA: 28,
    gateB: 29,
    compatible: (s, t) => s.micro.gate === 28 && t.micro.gate === 29 || s.micro.gate === 29 && t.micro.gate === 28,
    execute: (s, t) => {
      const commit = s.amplitude;
      const desire = t.amplitude;
      const depth = commit * desire;
      const emergent = {
        type: "RECOGNITION",
        coherence: depth,
        resonance: 7.83 * (1 + depth),
        tension: Math.abs(commit - desire),
        amplification: depth,
        trace: `Recognition: Gates ${s.micro.gate + 1} commits ${commit.toFixed(3)}, Gate ${t.micro.gate + 1} desires ${desire.toFixed(3)} \u2192 depth ${depth.toFixed(3)}`
      };
      return {
        source: { ...s, amplitude: commit },
        target: { ...t, amplitude: desire },
        emergent
      };
    }
  },
  {
    type: "PRESERVATION",
    name: "Preservation",
    description: "Nurturing and values for care provision",
    gateA: 26,
    gateB: 49,
    compatible: (s, t) => s.micro.gate === 26 && t.micro.gate === 49 || s.micro.gate === 49 && t.micro.gate === 26,
    execute: (s, t) => {
      const nurture = s.amplitude;
      const values = t.coherence;
      const care = nurture * values;
      const emergent = {
        type: "PRESERVATION",
        coherence: care,
        resonance: 7.83 * (1 + care),
        tension: 0,
        amplification: care,
        trace: `Preservation: Gates ${s.micro.gate + 1} nurtures ${nurture.toFixed(3)}, Gate ${t.micro.gate + 1} values ${values.toFixed(3)} \u2192 care ${care.toFixed(3)}`
      };
      return {
        source: { ...s, amplitude: care },
        target: { ...t, amplitude: care },
        emergent
      };
    }
  },
  {
    type: "INITIATION",
    name: "Initiation",
    description: "Shock and love for spiritual awakening",
    gateA: 24,
    gateB: 50,
    compatible: (s, t) => s.micro.gate === 24 && t.micro.gate === 50 || s.micro.gate === 50 && t.micro.gate === 24,
    execute: (s, t) => {
      const shock = Math.random() < 0.3 ? 1 : 0;
      const love = s.coherence * t.coherence;
      const awakening = shock > 0 ? love : love * 0.5;
      const emergent = {
        type: "INITIATION",
        coherence: awakening,
        resonance: 7.83 * (1 + shock),
        tension: 1 - shock,
        amplification: awakening,
        trace: `Initiation: Gates ${s.micro.gate + 1} and ${t.micro.gate + 1} ${shock > 0 ? "shock" : "gently"} awaken to ${awakening.toFixed(3)}`
      };
      return {
        source: { ...s, amplitude: awakening },
        target: { ...t, amplitude: awakening },
        emergent
      };
    }
  },
  {
    type: "POWER",
    name: "Power",
    description: "Sacral force and intuitive knowing",
    gateA: 33,
    gateB: 56,
    compatible: (s, t) => s.micro.gate === 33 && t.micro.gate === 56 || s.micro.gate === 56 && t.micro.gate === 33,
    execute: (s, t) => {
      const force = s.amplitude * s.amplitude;
      const intuition = t.phase;
      const power = force * Math.cos(intuition);
      const emergent = {
        type: "POWER",
        coherence: Math.abs(power),
        resonance: 7.83 * (1 + Math.abs(power)),
        tension: power < 0 ? Math.abs(power) : 0,
        amplification: Math.abs(power),
        trace: `Power: Gates ${s.micro.gate + 1} and ${t.micro.gate + 1} generate ${power.toFixed(3)}`
      };
      return {
        source: { ...s, amplitude: force },
        target: { ...t, phase: intuition },
        emergent
      };
    }
  },
  {
    type: "TRANSIENCE",
    name: "Transience",
    description: "Change and crisis in experiential cycles",
    gateA: 34,
    gateB: 35,
    compatible: (s, t) => s.micro.gate === 34 && t.micro.gate === 35 || s.micro.gate === 35 && t.micro.gate === 34,
    execute: (s, t) => {
      const change = Math.abs(s.amplitude - t.amplitude);
      const crisis = 1 - Math.min(s.coherence, t.coherence);
      const cycle = change * crisis;
      const emergent = {
        type: "TRANSIENCE",
        coherence: 1 - crisis,
        resonance: 7.83 * (1 + change),
        tension: crisis,
        amplification: change,
        trace: `Transience: Gates ${s.micro.gate + 1} and ${t.micro.gate + 1} change ${change.toFixed(3)} through crisis ${crisis.toFixed(3)}`
      };
      return {
        source: { ...s, amplitude: s.amplitude * (1 - cycle) },
        target: { ...t, amplitude: t.amplitude * (1 + cycle) },
        emergent
      };
    }
  },
  {
    type: "COMMUNITY",
    name: "Community",
    description: "Friendship and aloneness in heart bonding",
    gateA: 36,
    gateB: 39,
    compatible: (s, t) => s.micro.gate === 36 && t.micro.gate === 39 || s.micro.gate === 39 && t.micro.gate === 36,
    execute: (s, t) => {
      const friend = Math.min(s.amplitude, t.amplitude);
      const alone = Math.abs(s.amplitude - t.amplitude);
      const bond = friend * (1 - alone);
      const emergent = {
        type: "COMMUNITY",
        coherence: bond,
        resonance: 7.83 * (1 + bond),
        tension: alone,
        amplification: friend,
        trace: `Community: Gates ${s.micro.gate + 1} and ${t.micro.gate + 1} bond ${bond.toFixed(3)} (friend ${friend.toFixed(3)}, alone ${alone.toFixed(3)})`
      };
      return {
        source: { ...s, amplitude: bond },
        target: { ...t, amplitude: bond },
        emergent
      };
    }
  },
  {
    type: "MATURATION",
    name: "Maturation",
    description: "Growth and beginnings in developmental cycles",
    gateA: 41,
    gateB: 52,
    compatible: (s, t) => s.micro.gate === 41 && t.micro.gate === 52 || s.micro.gate === 52 && t.micro.gate === 41,
    execute: (s, t) => {
      const growth = s.coherence;
      const begin = 1 - t.coherence;
      const mature = growth * begin;
      const emergent = {
        type: "MATURATION",
        coherence: mature,
        resonance: 7.83 * (1 + mature),
        tension: 1 - mature,
        amplification: growth,
        trace: `Maturation: Gates ${s.micro.gate + 1} grows ${growth.toFixed(3)}, Gate ${t.micro.gate + 1} begins ${begin.toFixed(3)} \u2192 mature ${mature.toFixed(3)}`
      };
      return {
        source: { ...s, coherence: mature },
        target: { ...t, coherence: mature },
        emergent
      };
    }
  },
  {
    type: "TRANSFORMATION",
    name: "Transformation",
    description: "Ambition and drive through root pressure",
    gateA: 31,
    gateB: 53,
    compatible: (s, t) => s.micro.gate === 31 && t.micro.gate === 53 || s.micro.gate === 53 && t.micro.gate === 31,
    execute: (s, t) => {
      const ambition = s.amplitude;
      const drive = t.amplitude;
      const pressure = ambition * drive;
      const transform = Math.sqrt(pressure);
      const emergent = {
        type: "TRANSFORMATION",
        coherence: transform,
        resonance: 7.83 * (1 + pressure),
        tension: 1 - transform,
        amplification: pressure,
        trace: `Transformation: Gates ${s.micro.gate + 1} and ${t.micro.gate + 1} transform ${transform.toFixed(3)} from pressure ${pressure.toFixed(3)}`
      };
      return {
        source: { ...s, amplitude: transform },
        target: { ...t, amplitude: transform },
        emergent
      };
    }
  },
  {
    type: "EMOTING",
    name: "Emoting",
    description: "Provocation and spirit in emotional wave",
    gateA: 38,
    gateB: 54,
    compatible: (s, t) => s.micro.gate === 38 && t.micro.gate === 54 || s.micro.gate === 54 && t.micro.gate === 38,
    execute: (s, t) => {
      const provoke = s.phase;
      const spirit = t.phase;
      const wave = Math.sin(provoke) * Math.cos(spirit);
      const emergent = {
        type: "EMOTING",
        coherence: Math.abs(wave),
        resonance: 7.83 * (1 + Math.abs(wave)),
        tension: wave < 0 ? Math.abs(wave) : 0,
        amplification: Math.abs(wave),
        trace: `Emoting: Gates ${s.micro.gate + 1} and ${t.micro.gate + 1} wave ${wave.toFixed(3)}`
      };
      return {
        source: { ...s, phase: provoke + 0.1 },
        target: { ...t, phase: spirit + 0.1 },
        emergent
      };
    }
  },
  {
    type: "AWARENESS",
    name: "Awareness",
    description: "Rationalization and mystery in mental awareness",
    gateA: 23,
    gateB: 60,
    compatible: (s, t) => s.micro.gate === 23 && t.micro.gate === 60 || s.micro.gate === 60 && t.micro.gate === 23,
    execute: (s, t) => {
      const rational = s.coherence;
      const mystery = 1 - t.coherence;
      const aware = rational * mystery;
      const emergent = {
        type: "AWARENESS",
        coherence: aware,
        resonance: 7.83 * (1 + aware),
        tension: mystery,
        amplification: rational,
        trace: `Awareness: Gates ${s.micro.gate + 1} rational ${rational.toFixed(3)} \xD7 Gate ${t.micro.gate + 1} mystery ${mystery.toFixed(3)} = ${aware.toFixed(3)}`
      };
      return {
        source: { ...s, coherence: aware },
        target: { ...t, coherence: aware },
        emergent
      };
    }
  },
  {
    type: "STRUCTURING",
    name: "Structuring",
    description: "Insight and assimilation in mental architecture",
    gateA: 22,
    gateB: 42,
    compatible: (s, t) => s.micro.gate === 22 && t.micro.gate === 42 || s.micro.gate === 42 && t.micro.gate === 22,
    execute: (s, t) => {
      const insight = s.amplitude;
      const assimilate = t.amplitude;
      const structure = Math.min(insight, assimilate);
      const emergent = {
        type: "STRUCTURING",
        coherence: structure,
        resonance: 7.83 * (1 + structure),
        tension: Math.abs(insight - assimilate),
        amplification: structure,
        trace: `Structuring: Gates ${s.micro.gate + 1} insight ${insight.toFixed(3)} + Gate ${t.micro.gate + 1} assimilate ${assimilate.toFixed(3)} = ${structure.toFixed(3)}`
      };
      return {
        source: { ...s, amplitude: structure },
        target: { ...t, amplitude: structure },
        emergent
      };
    }
  },
  {
    type: "SURRENDER",
    name: "Surrender",
    description: "Ego and alertness in transmission",
    gateA: 25,
    gateB: 43,
    compatible: (s, t) => s.micro.gate === 25 && t.micro.gate === 43 || s.micro.gate === 43 && t.micro.gate === 25,
    execute: (s, t) => {
      const ego = s.amplitude;
      const alert = t.coherence;
      const surrender = ego * (1 - alert);
      const transmit = ego * alert;
      const emergent = {
        type: "SURRENDER",
        coherence: transmit,
        resonance: 7.83 * (1 + transmit),
        tension: surrender,
        amplification: transmit,
        trace: `Surrender: Gates ${s.micro.gate + 1} ego ${ego.toFixed(3)} \xD7 Gate ${t.micro.gate + 1} alert ${alert.toFixed(3)} = transmit ${transmit.toFixed(3)}`
      };
      return {
        source: { ...s, amplitude: transmit },
        target: { ...t, amplitude: transmit },
        emergent
      };
    }
  },
  {
    type: "DISCOVERY",
    name: "Discovery",
    description: "Determination and perseverance in body discovery",
    gateA: 45,
    gateB: 28,
    compatible: (s, t) => s.micro.gate === 45 && t.micro.gate === 28 || s.micro.gate === 28 && t.micro.gate === 45,
    execute: (s, t) => {
      const determine = s.amplitude;
      const persevere = t.coherence;
      const discover = determine * persevere;
      const emergent = {
        type: "DISCOVERY",
        coherence: discover,
        resonance: 7.83 * (1 + discover),
        tension: 1 - persevere,
        amplification: discover,
        trace: `Discovery: Gates ${s.micro.gate + 1} determines ${determine.toFixed(3)} \xD7 Gate ${t.micro.gate + 1} perseveres ${persevere.toFixed(3)} = ${discover.toFixed(3)}`
      };
      return {
        source: { ...s, amplitude: discover },
        target: { ...t, amplitude: discover },
        emergent
      };
    }
  },
  {
    type: "ABSTRACTION",
    name: "Abstraction",
    description: "Realization and confusion in mental abstraction",
    gateA: 46,
    gateB: 63,
    compatible: (s, t) => s.micro.gate === 46 && t.micro.gate === 63 || s.micro.gate === 63 && t.micro.gate === 46,
    execute: (s, t) => {
      const realize = s.coherence;
      const confuse = 1 - t.coherence;
      const abstract = realize * confuse;
      const emergent = {
        type: "ABSTRACTION",
        coherence: abstract,
        resonance: 7.83 * (1 + abstract),
        tension: confuse,
        amplification: realize,
        trace: `Abstraction: Gates ${s.micro.gate + 1} realizes ${realize.toFixed(3)} through Gate ${t.micro.gate + 1} confusion ${confuse.toFixed(3)}`
      };
      return {
        source: { ...s, coherence: abstract },
        target: { ...t, coherence: abstract },
        emergent
      };
    }
  },
  {
    type: "DETACHMENT",
    name: "Detachment",
    description: "Contraction and desire as fantasy fuel",
    gateA: 40,
    gateB: 29,
    compatible: (s, t) => s.micro.gate === 40 && t.micro.gate === 29 || s.micro.gate === 29 && t.micro.gate === 40,
    execute: (s, t) => {
      const contract = 1 - s.amplitude;
      const desire = t.amplitude;
      const fantasy = contract * desire;
      const emergent = {
        type: "DETACHMENT",
        coherence: fantasy,
        resonance: 7.83 * (1 + fantasy),
        tension: contract,
        amplification: desire,
        trace: `Detachment: Gates ${s.micro.gate + 1} contracts ${contract.toFixed(3)} \xD7 Gate ${t.micro.gate + 1} desires ${desire.toFixed(3)} = fantasy ${fantasy.toFixed(3)}`
      };
      return {
        source: { ...s, amplitude: contract },
        target: { ...t, amplitude: desire },
        emergent
      };
    }
  },
  {
    type: "COMPLETION",
    name: "Completion",
    description: "Truth and mystery in mental completion",
    gateA: 60,
    gateB: 23,
    compatible: (s, t) => s.micro.gate === 60 && t.micro.gate === 23 || s.micro.gate === 23 && t.micro.gate === 60,
    execute: (s, t) => {
      const truth = s.coherence;
      const mystery = 1 - t.coherence;
      const complete = truth * (1 - mystery * 0.5);
      const emergent = {
        type: "COMPLETION",
        coherence: complete,
        resonance: 7.83 * (1 + complete),
        tension: mystery,
        amplification: truth,
        trace: `Completion: Gates ${s.micro.gate + 1} truth ${truth.toFixed(3)} + Gate ${t.micro.gate + 1} mystery ${mystery.toFixed(3)} = ${complete.toFixed(3)}`
      };
      return {
        source: { ...s, coherence: complete },
        target: { ...t, coherence: complete },
        emergent
      };
    }
  },
  {
    type: "DETAILS",
    name: "Details",
    description: "Impeccability and opinion in logical details",
    gateA: 61,
    gateB: 16,
    compatible: (s, t) => s.micro.gate === 61 && t.micro.gate === 16 || s.micro.gate === 16 && t.micro.gate === 61,
    execute: (s, t) => {
      const impeccable = s.coherence;
      const opinion = t.phase;
      const detail = impeccable * Math.cos(opinion);
      const emergent = {
        type: "DETAILS",
        coherence: Math.abs(detail),
        resonance: 7.83 * (1 + Math.abs(detail)),
        tension: detail < 0 ? Math.abs(detail) : 0,
        amplification: Math.abs(detail),
        trace: `Details: Gates ${s.micro.gate + 1} impeccable ${impeccable.toFixed(3)} \xD7 Gate ${t.micro.gate + 1} opinion ${opinion.toFixed(3)} = ${detail.toFixed(3)}`
      };
      return {
        source: { ...s, coherence: Math.abs(detail) },
        target: { ...t, phase: opinion },
        emergent
      };
    }
  },
  {
    type: "TRUTH",
    name: "Truth",
    description: "Doubt and answers in mental truth",
    gateA: 62,
    gateB: 3,
    compatible: (s, t) => s.micro.gate === 62 && t.micro.gate === 3 || s.micro.gate === 3 && t.micro.gate === 62,
    execute: (s, t) => {
      const doubt = 1 - s.coherence;
      const answer = t.coherence;
      const truth = answer * (1 - doubt * 0.5);
      const emergent = {
        type: "TRUTH",
        coherence: truth,
        resonance: 7.83 * (1 + truth),
        tension: doubt,
        amplification: answer,
        trace: `Truth: Gates ${s.micro.gate + 1} doubts ${doubt.toFixed(3)}, Gate ${t.micro.gate + 1} answers ${answer.toFixed(3)} = ${truth.toFixed(3)}`
      };
      return {
        source: { ...s, coherence: truth },
        target: { ...t, coherence: truth },
        emergent
      };
    }
  },
  {
    type: "IMAGINATION",
    name: "Imagination",
    description: "Confusion and realization in mental imagination",
    gateA: 63,
    gateB: 46,
    compatible: (s, t) => s.micro.gate === 63 && t.micro.gate === 46 || s.micro.gate === 46 && t.micro.gate === 63,
    execute: (s, t) => {
      const confuse = 1 - s.coherence;
      const realize = t.coherence;
      const imagine = confuse * realize;
      const emergent = {
        type: "IMAGINATION",
        coherence: imagine,
        resonance: 7.83 * (1 + imagine),
        tension: confuse,
        amplification: realize,
        trace: `Imagination: Gates ${s.micro.gate + 1} confuses ${confuse.toFixed(3)} \xD7 Gate ${t.micro.gate + 1} realizes ${realize.toFixed(3)} = ${imagine.toFixed(3)}`
      };
      return {
        source: { ...s, coherence: imagine },
        target: { ...t, coherence: imagine },
        emergent
      };
    }
  }
];
class EmergentEdgeResolver {
  edgeFunctions;
  constructor() {
    this.edgeFunctions = /* @__PURE__ */ new Map();
    for (const fn of EDGE_FUNCTIONS) {
      const key = `${fn.gateA}-${fn.gateB}`;
      this.edgeFunctions.set(key, fn);
    }
  }
  /**
   * Resolve an edge between two states.
   * 
   * Flow:
   *   1. Check compatibility
   *   2. Find matching edge function(s)
   *   3. Select best match based on compatibility score
   *   4. Execute the edge function
   *   5. Return transformed states + emergent properties
   */
  resolve(source, target) {
    const compatibility = checkCompatibility(source, target);
    if (!compatibility.compatible) {
      return {
        compatible: false,
        score: compatibility.score,
        edgeType: "NONE",
        edgeName: "No Edge",
        source,
        target,
        emergent: {
          type: "NONE",
          coherence: 0,
          resonance: 0,
          tension: 0,
          amplification: 0,
          trace: `No compatible edge between Gate ${source.micro.gate + 1} and Gate ${target.micro.gate + 1}`
        }
      };
    }
    const matches = EDGE_FUNCTIONS.filter((fn) => fn.compatible(source, target));
    if (matches.length === 0) {
      return {
        compatible: true,
        score: compatibility.score,
        edgeType: "NONE",
        edgeName: "Generic Connection",
        source,
        target,
        emergent: {
          type: "NONE",
          coherence: compatibility.score,
          resonance: 7.83 * compatibility.score,
          tension: 1 - compatibility.score,
          amplification: compatibility.score,
          trace: `Generic connection between Gate ${source.micro.gate + 1} and Gate ${target.micro.gate + 1} (score: ${compatibility.score.toFixed(3)})`
        }
      };
    }
    const bestMatch = matches[0];
    const result = bestMatch.execute(source, target);
    return {
      compatible: true,
      score: compatibility.score,
      edgeType: bestMatch.type,
      edgeName: bestMatch.name,
      source: result.source,
      target: result.target,
      emergent: result.emergent
    };
  }
  /**
   * Resolve all possible edges from a source state to all targets in a set.
   */
  resolveAll(source, targets) {
    const edges = targets.map((t) => this.resolve(source, t));
    const validEdges = edges.filter((e) => e.compatible);
    const bestEdge = validEdges.length > 0 ? validEdges.reduce((best, current) => current.score > best.score ? current : best) : null;
    return { source, edges, bestEdge };
  }
  /**
   * Get all 36 edge type definitions.
   */
  getEdgeTypes() {
    return [...EDGE_FUNCTIONS];
  }
}
var emergent_edge_resolver_default = EmergentEdgeResolver;
export {
  EDGE_FUNCTIONS,
  EmergentEdgeResolver,
  emergent_edge_resolver_default as default
};
