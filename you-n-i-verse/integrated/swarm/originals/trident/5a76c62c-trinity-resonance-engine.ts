/**
 * Trinity Resonance Engine
 * 
 * Calculates Mind/Body/Heart compatibility between users for optimal matching.
 * Based on the Trinity Assessment system that measures consciousness across
 * three primary energy centers.
 */

export const ROLES = ["Generator", "Amplifier", "Anchor", "Catalyst", "Harmonizer"] as const;
export type Role = typeof ROLES[number];

export const PLACEMENTS = [
  "Core Receiver (Dream Focus)",
  "Distributor (Daily Support Exchange)",
  "Peer Mirror (Mutual Reflection)",
  "Guide (Lived Experience Mentor)",
  "Circle (Group Resonance)",
] as const;
export type Placement = typeof PLACEMENTS[number];

export interface TrinityChart {
  mind: number;  // 0-100: Strategic thinking, clarity, vision
  body: number;  // 0-100: Physical vitality, discipline, action
  heart: number; // 0-100: Compassion, connection, empathy
}

export interface TrinityAssessment {
  clarity: number;      // 1-10: Strategic thinking and vision
  discipline: number;   // 1-10: Focus and execution
  compassion: number;   // 1-10: Empathy and care
  adaptability: number; // 1-10: Flexibility and creativity
  vitality: number;     // 1-10: Physical energy and action
  connection: number;   // 1-10: Interpersonal skills
}

export interface UserResonanceProfile {
  userId: string;
  trinityChart: TrinityChart;
  dominantRole: Role;
  readiness: number;     // 0-100: Overall readiness index
  placement: Placement;
  skills?: string[];
  tasks?: string[];
  dream?: string;
}

const clamp = (n: number, min = 0, max = 100) => 
  Math.max(min, Math.min(max, Math.round(n)));

/**
 * Calculate Mind/Body/Heart energies from Trinity Assessment
 * 
 * This algorithm distributes assessment scores across the three centers:
 * - Clarity → Mind (primary)
 * - Discipline → Mind + Body (split)
 * - Compassion → Heart (primary)
 * - Adaptability → Mind + Heart (split)
 * - Vitality → Body (primary)
 * - Connection → Heart + Mind (weighted)
 */
export function chartFromAssessment(assessment: TrinityAssessment): TrinityChart {
  // Mind components
  const mindFromClarity = assessment.clarity * 10;       // 10-100 points
  const mindFromDiscipline = assessment.discipline * 5;   // 5-50 points
  const mindFromAdaptability = assessment.adaptability * 5; // 5-50 points
  const mindFromConnection = assessment.connection * 3;   // 3-30 points
  
  // Body components
  const bodyFromDiscipline = assessment.discipline * 5;   // 5-50 points
  const bodyFromVitality = assessment.vitality * 10;      // 10-100 points
  
  // Heart components
  const heartFromCompassion = assessment.compassion * 10; // 10-100 points
  const heartFromAdaptability = assessment.adaptability * 5; // 5-50 points
  const heartFromConnection = assessment.connection * 5;  // 5-50 points
  
  const mind = mindFromClarity + mindFromDiscipline + mindFromAdaptability + mindFromConnection;
  const body = bodyFromDiscipline + bodyFromVitality;
  const heart = heartFromCompassion + heartFromAdaptability + heartFromConnection;
  
  return {
    mind: clamp(mind),
    body: clamp(body),
    heart: clamp(heart)
  };
}

/**
 * Determine dominant role from Trinity Chart
 * 
 * Roles represent consciousness archetypes:
 * - Generator: High Mind + Body (creates & executes)
 * - Amplifier: High Heart + Mind (inspires & communicates)
 * - Anchor: High Body + Heart (grounds & supports)
 * - Catalyst: Balanced high energies (transforms & balances)
 * - Harmonizer: Moderate balanced energies (integrates & adapts)
 */
export function roleFromChart(chart: TrinityChart): Role {
  const { mind, body, heart } = chart;
  
  // Catalyst: Balanced high energies (all three high)
  if (mind >= 70 && body >= 70 && heart >= 70) {
    const variance = Math.max(mind, body, heart) - Math.min(mind, body, heart);
    if (variance <= 15) return "Catalyst";
  }
  
  // Harmonizer: Moderate balanced energies
  const variance = Math.max(mind, body, heart) - Math.min(mind, body, heart);
  if (variance <= 20 && Math.max(mind, body, heart) < 70) return "Harmonizer";
  
  // High energy role patterns
  if (Math.max(mind, body, heart) >= 70) {
    // Generator: High Mind + Body combination
    if (mind >= 70 && body >= 70 && mind >= heart && body >= heart) return "Generator";
    
    // Amplifier: High Heart + Mind combination  
    if (heart >= 70 && mind >= 70 && heart >= body && mind >= body) return "Amplifier";
    
    // Anchor: High Body + Heart combination
    if (body >= 70 && heart >= 70 && body >= mind && heart >= mind) return "Anchor";
    
    // Single high energy fallbacks
    if (mind >= body && mind >= heart) return "Generator";
    if (heart >= mind && heart >= body) return "Amplifier";
    if (body >= mind && body >= heart) return "Anchor";
  }
  
  // Default fallback for edge cases
  return "Harmonizer";
}

/**
 * Calculate readiness index from Trinity Chart
 * 
 * Combines average energy with balance penalty.
 * Higher = more ready for intensive growth work.
 */
export function readinessIndex(chart: TrinityChart): number {
  const avg = (chart.mind + chart.body + chart.heart) / 3;
  const balancePenalty = (
    Math.abs(chart.mind - avg) + 
    Math.abs(chart.body - avg) + 
    Math.abs(chart.heart - avg)
  ) / 3;
  return clamp(avg - balancePenalty);
}

/**
 * Suggest optimal placement in resonance network
 */
export function placementSuggestion(profile: UserResonanceProfile): Placement {
  if (profile.readiness >= 75) return "Core Receiver (Dream Focus)";
  if (profile.dominantRole === "Amplifier") return "Distributor (Daily Support Exchange)";
  if (profile.dominantRole === "Anchor") return "Circle (Group Resonance)";
  if (profile.dominantRole === "Catalyst") return "Peer Mirror (Mutual Reflection)";
  return "Guide (Lived Experience Mentor)";
}

/**
 * Role complementarity map
 * Defines which roles work best together
 */
const COMPLEMENTS: Record<Role, Role[]> = {
  Generator: ["Amplifier", "Harmonizer"],
  Amplifier: ["Generator", "Anchor"],
  Anchor: ["Catalyst", "Amplifier"],
  Catalyst: ["Anchor", "Harmonizer"],
  Harmonizer: ["Generator", "Catalyst"]
};

/**
 * Calculate cosine similarity between two Trinity Charts
 * Returns value between -1 and 1 (higher = more similar)
 */
function cosineSimilarity(a: TrinityChart, b: TrinityChart): number {
  const v1 = [a.mind, a.body, a.heart];
  const v2 = [b.mind, b.body, b.heart];
  const dot = v1.reduce((s, x, i) => s + x * v2[i], 0);
  const m1 = Math.sqrt(v1.reduce((s, x) => s + x * x, 0));
  const m2 = Math.sqrt(v2.reduce((s, x) => s + x * x, 0));
  return dot / (m1 * m2);
}

/**
 * Calculate energy compatibility between two profiles
 * Returns 0-1 score (higher = better match)
 */
export function compatibilityEnergy(a: UserResonanceProfile, b: UserResonanceProfile): number {
  const base = cosineSimilarity(a.trinityChart, b.trinityChart);
  const bonus = COMPLEMENTS[a.dominantRole]?.includes(b.dominantRole) ? 0.08 : 0;
  return (base + 1) / 2 + bonus;
}

/**
 * Calculate Jaccard similarity between two sets
 * Used for skill/task overlap
 */
function jaccardSimilarity(a: string[] = [], b: string[] = []): number {
  if (!a.length && !b.length) return 0;
  const A = new Set(a);
  const B = new Set(b);
  const intersection = Array.from(A).filter(x => B.has(x)).length;
  const union = new Set([...a, ...b]).size;
  return intersection / union;
}

/**
 * Calculate overall compatibility between two profiles
 * 
 * @param weights - Adjust importance of different factors
 * @returns Compatibility score 0-100
 */
export function overallCompatibility(
  a: UserResonanceProfile, 
  b: UserResonanceProfile,
  weights = { energy: 0.85, skills: 0.1, tasks: 0.05 }
): number {
  const energyScore = compatibilityEnergy(a, b);
  const skillsScore = jaccardSimilarity(a.skills, b.skills);
  const tasksScore = jaccardSimilarity(a.tasks, b.tasks);
  
  const score = weights.energy * energyScore + weights.skills * skillsScore + weights.tasks * tasksScore;
  return clamp(score * 100);
}

/**
 * Find top N matches for a user from a pool of candidates
 * 
 * @param user - User to find matches for
 * @param candidates - Pool of potential matches
 * @param limit - Maximum number of matches to return
 * @returns Array of matches sorted by compatibility (highest first)
 */
export function findTopMatches(
  user: UserResonanceProfile,
  candidates: UserResonanceProfile[],
  limit: number = 10
): Array<{ profile: UserResonanceProfile; compatibility: number }> {
  return candidates
    .filter(c => c.userId !== user.userId) // Don't match with self
    .map(candidate => ({
      profile: candidate,
      compatibility: overallCompatibility(user, candidate)
    }))
    .sort((a, b) => b.compatibility - a.compatibility)
    .slice(0, limit);
}

/**
 * Form optimal pods (teams) for a specific goal/project
 * 
 * Pods are groups of 3-6 people matched for maximum resonance
 * around a shared objective.
 */
export function formOptimalPod(
  goal: {
    requiredSkills: string[];
    optimalRoles?: Role[];
    minSize?: number;
    maxSize?: number;
  },
  candidates: UserResonanceProfile[]
): UserResonanceProfile[] {
  const minSize = goal.minSize || 3;
  const maxSize = goal.maxSize || 6;
  
  // Filter candidates by required skills
  const qualified = candidates.filter(c => 
    goal.requiredSkills.some(skill => c.skills?.includes(skill))
  );
  
  // If optimal roles specified, prioritize those
  let selected: UserResonanceProfile[] = [];
  if (goal.optimalRoles && goal.optimalRoles.length > 0) {
    for (const role of goal.optimalRoles) {
      const roleMatch = qualified.find(c => c.dominantRole === role && !selected.includes(c));
      if (roleMatch) selected.push(roleMatch);
    }
  }
  
  // Fill remaining spots with highest readiness candidates
  const remaining = qualified
    .filter(c => !selected.includes(c))
    .sort((a, b) => b.readiness - a.readiness);
  
  while (selected.length < maxSize && remaining.length > 0) {
    selected.push(remaining.shift()!);
  }
  
  return selected.length >= minSize ? selected : [];
}
