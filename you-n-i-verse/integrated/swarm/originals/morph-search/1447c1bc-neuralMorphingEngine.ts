/**
 * Neural Morphing Engine
 * 
 * Implements system changes at different interaction levels:
 * - FAST: Real-time responses to immediate feedback (minutes)
 * - SLOW: Long-term pattern evolution (hours/days)
 * - STABLE: Core knowledge that rarely changes (canonical, proven)
 */

import * as db from "./db";
import { invokeLLM } from "./_core/llm";

export type ChangeSpeed = "fast" | "slow" | "stable";

export interface MorphingState {
  userId: number;
  timestamp: Date;
  speed: ChangeSpeed;
  changeType: string;
  oldValue: string;
  newValue: string;
  confidence: number; // 0-1
  impact: number; // 0-100, how much this affects behavior
}

export interface NeuralNetwork {
  userId: number;
  fastWeights: Record<string, number>; // Real-time adjustments
  slowWeights: Record<string, number>; // Long-term patterns
  stableWeights: Record<string, number>; // Canonical knowledge
  lastUpdated: Date;
}

// In-memory neural networks for active users
const userNeuralNetworks = new Map<number, NeuralNetwork>();

/**
 * Initialize neural network for a user
 */
export function initializeNeuralNetwork(userId: number): NeuralNetwork {
  const network: NeuralNetwork = {
    userId,
    fastWeights: {},
    slowWeights: {},
    stableWeights: {},
    lastUpdated: new Date(),
  };

  userNeuralNetworks.set(userId, network);
  console.log(`[Neural Network] Initialized for user ${userId}`);

  return network;
}

/**
 * Get neural network for a user
 */
export function getNeuralNetwork(userId: number): NeuralNetwork {
  let network = userNeuralNetworks.get(userId);

  if (!network) {
    network = initializeNeuralNetwork(userId);
  }

  return network;
}

/**
 * Apply FAST morphing: Real-time response to immediate feedback
 * Used for: Response style adjustments, tone changes, immediate corrections
 */
export async function applyFastMorphing(
  userId: number,
  changeType: string,
  newValue: string,
  confidence: number = 0.8
): Promise<MorphingState> {
  try {
    const network = getNeuralNetwork(userId);
    const oldValue = network.fastWeights[changeType]?.toString() || "default";

    // Update fast weights immediately
    network.fastWeights[changeType] = confidence * 100;
    network.lastUpdated = new Date();

    const morphingState: MorphingState = {
      userId,
      timestamp: new Date(),
      speed: "fast",
      changeType,
      oldValue,
      newValue,
      confidence,
      impact: Math.round(confidence * 100),
    };

    console.log(`[Fast Morphing] User ${userId}: ${changeType} → ${newValue} (${confidence * 100}% confidence)`);

    return morphingState;
  } catch (error) {
    console.error("Error applying fast morphing:", error);
    throw error;
  }
}

/**
 * Apply SLOW morphing: Long-term pattern evolution
 * Used for: Behavior patterns, communication style preferences, learning trends
 */
export async function applySlowMorphing(
  userId: number,
  changeType: string,
  newValue: string,
  successRate: number = 0.5
): Promise<MorphingState> {
  try {
    const network = getNeuralNetwork(userId);
    const oldValue = network.slowWeights[changeType]?.toString() || "default";

    // Update slow weights gradually
    const currentWeight = network.slowWeights[changeType] || 50;
    const newWeight = currentWeight * 0.7 + successRate * 100 * 0.3; // Exponential moving average
    network.slowWeights[changeType] = newWeight;
    network.lastUpdated = new Date();

    const morphingState: MorphingState = {
      userId,
      timestamp: new Date(),
      speed: "slow",
      changeType,
      oldValue,
      newValue,
      confidence: successRate,
      impact: Math.round(newWeight),
    };

    console.log(`[Slow Morphing] User ${userId}: ${changeType} → ${newValue} (${(successRate * 100).toFixed(0)}% success rate)`);

    return morphingState;
  } catch (error) {
    console.error("Error applying slow morphing:", error);
    throw error;
  }
}

/**
 * Apply STABLE morphing: Core knowledge updates
 * Used for: Proven facts, canonical knowledge, fundamental understanding
 * These changes are rare and require high confidence
 */
export async function applyStableMorphing(
  userId: number,
  changeType: string,
  newValue: string,
  confidence: number = 0.95
): Promise<MorphingState> {
  try {
    if (confidence < 0.9) {
      throw new Error("Stable morphing requires 90%+ confidence");
    }

    const network = getNeuralNetwork(userId);
    const oldValue = network.stableWeights[changeType]?.toString() || "default";

    // Update stable weights (rarely changes)
    network.stableWeights[changeType] = confidence * 100;
    network.lastUpdated = new Date();

    const morphingState: MorphingState = {
      userId,
      timestamp: new Date(),
      speed: "stable",
      changeType,
      oldValue,
      newValue,
      confidence,
      impact: 100, // Stable changes have maximum impact
    };

    console.log(`[Stable Morphing] User ${userId}: ${changeType} → ${newValue} (${(confidence * 100).toFixed(0)}% confidence)`);

    return morphingState;
  } catch (error) {
    console.error("Error applying stable morphing:", error);
    throw error;
  }
}

/**
 * Calculate morphing recommendation based on feedback
 */
export async function calculateMorphingRecommendation(
  userId: number,
  feedback: {
    rating: number; // 1-5
    comment?: string;
    context?: string;
  }
): Promise<{
  speed: ChangeSpeed;
  changeType: string;
  recommendation: string;
  confidence: number;
}> {
  try {
    // Determine speed based on feedback strength
    let speed: ChangeSpeed;
    let confidence: number;

    if (feedback.rating >= 4) {
      // Strong positive: reinforce with slow morphing
      speed = "slow";
      confidence = 0.7;
    } else if (feedback.rating <= 2) {
      // Strong negative: quick adjustment with fast morphing
      speed = "fast";
      confidence = 0.6;
    } else {
      // Neutral: observe more before changing
      speed = "slow";
      confidence = 0.4;
    }

    // Use LLM to generate specific recommendation
    const prompt = `Based on this feedback, what specific change should the system make?

Rating: ${feedback.rating}/5
Comment: ${feedback.comment || "No comment"}
Context: ${feedback.context || "General interaction"}

Respond with a JSON object:
{
  "changeType": "specific change category",
  "recommendation": "specific change to make"
}`;

    const response = await invokeLLM({
      messages: [{ role: "user", content: prompt }],
    });

    const responseText = response.choices[0]?.message?.content || "";
    const responseStr = typeof responseText === "string" ? responseText : "";
    const jsonMatch = responseStr.match(/\{[\s\S]*\}/);

    if (!jsonMatch) {
      return {
        speed,
        changeType: "general-adjustment",
        recommendation: "Adjust response style based on feedback",
        confidence,
      };
    }

    const recommendation = JSON.parse(jsonMatch[0]);

    return {
      speed,
      changeType: recommendation.changeType || "general-adjustment",
      recommendation: recommendation.recommendation || "Adjust response style",
      confidence,
    };
  } catch (error) {
    console.error("Error calculating morphing recommendation:", error);
    return {
      speed: "slow",
      changeType: "error-recovery",
      recommendation: "Continue with current approach",
      confidence: 0.3,
    };
  }
}

/**
 * Get morphing history for a user
 */
export async function getMorphingHistory(userId: number, limit: number = 20): Promise<MorphingState[]> {
  try {
    // This would query a morphing_history table
    // For now, return empty array
    return [];
  } catch (error) {
    console.error("Error getting morphing history:", error);
    return [];
  }
}

/**
 * Get neural network state summary
 */
export function getNeuralNetworkSummary(userId: number): {
  fastChanges: number;
  slowChanges: number;
  stableChanges: number;
  dominantSpeed: ChangeSpeed;
  lastUpdated: Date;
} {
  const network = getNeuralNetwork(userId);

  const fastCount = Object.keys(network.fastWeights).length;
  const slowCount = Object.keys(network.slowWeights).length;
  const stableCount = Object.keys(network.stableWeights).length;

  // Determine which speed has most changes
  let dominantSpeed: ChangeSpeed = "slow";
  if (fastCount > slowCount && fastCount > stableCount) {
    dominantSpeed = "fast";
  } else if (stableCount > slowCount && stableCount > fastCount) {
    dominantSpeed = "stable";
  }

  return {
    fastChanges: fastCount,
    slowChanges: slowCount,
    stableChanges: stableCount,
    dominantSpeed,
    lastUpdated: network.lastUpdated,
  };
}

/**
 * Apply neural network weights to a prompt
 */
export async function applyNeuralWeights(userId: number, basePrompt: string): Promise<string> {
  try {
    const network = getNeuralNetwork(userId);

    // Build weight context
    const fastContext = Object.entries(network.fastWeights)
      .map(([k, v]) => `- ${k}: ${(v / 100).toFixed(2)} (immediate)`)
      .join("\n");

    const slowContext = Object.entries(network.slowWeights)
      .map(([k, v]) => `- ${k}: ${(v / 100).toFixed(2)} (learned)`)
      .join("\n");

    const stableContext = Object.entries(network.stableWeights)
      .map(([k, v]) => `- ${k}: ${(v / 100).toFixed(2)} (proven)`)
      .join("\n");

    const weightContext = `
NEURAL NETWORK STATE:

FAST (Real-time adjustments):
${fastContext || "None"}

SLOW (Long-term patterns):
${slowContext || "None"}

STABLE (Canonical knowledge):
${stableContext || "None"}
`;

    const enhancedPrompt = `${basePrompt}

${weightContext}

Apply these neural weights to your response. Prioritize stable knowledge, then incorporate slow patterns, then apply fast adjustments.`;

    return enhancedPrompt;
  } catch (error) {
    console.error("Error applying neural weights:", error);
    return basePrompt;
  }
}

/**
 * Decay fast weights over time (they should fade if not reinforced)
 */
export function decayFastWeights(userId: number, decayRate: number = 0.95): void {
  try {
    const network = getNeuralNetwork(userId);

    // Apply exponential decay to fast weights
    Object.keys(network.fastWeights).forEach((key) => {
      network.fastWeights[key] *= decayRate;

      // Remove if weight drops below threshold
      if (network.fastWeights[key] < 5) {
        delete network.fastWeights[key];
      }
    });

    console.log(`[Decay] Fast weights decayed for user ${userId}`);
  } catch (error) {
    console.error("Error decaying fast weights:", error);
  }
}

/**
 * Promote slow weights to stable if they reach high confidence
 */
export function promoteSlowToStable(userId: number, threshold: number = 85): void {
  try {
    const network = getNeuralNetwork(userId);

    // Check each slow weight
    Object.entries(network.slowWeights).forEach(([key, value]) => {
      if (value >= threshold) {
        // Promote to stable
        network.stableWeights[key] = value;
        delete network.slowWeights[key];

        console.log(`[Promotion] ${key} promoted from slow to stable for user ${userId}`);
      }
    });
  } catch (error) {
    console.error("Error promoting slow weights to stable:", error);
  }
}
