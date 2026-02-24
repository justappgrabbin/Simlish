using System.Text.Json;

namespace ModularSimWorld.Components;

/// <summary>
/// Unlimited Outcomes Engine - Infinite variations from 64 harmonic base frequencies
/// Each codon can manifest in unlimited innocent situations through combinatorial expansion
/// Awaiting user's sentence structure and calculation framework
/// </summary>
public class UnlimitedOutcomesEngine
{
    private readonly HarmonicFrequencyEngine _harmonicEngine;
    private readonly Dictionary<string, OutcomeMultiplier> _multipliers;
    private readonly Dictionary<string, SentenceTemplate> _sentenceTemplates;
    private readonly Random _random;

    public UnlimitedOutcomesEngine(HarmonicFrequencyEngine harmonicEngine)
    {
        _harmonicEngine = harmonicEngine;
        _random = new Random();
        _multipliers = InitializeOutcomeMultipliers();
        _sentenceTemplates = InitializeSentenceTemplates(); // Placeholder for user's structure
        Console.WriteLine("[Unlimited] Outcome expansion engine initialized - awaiting user calculations");
    }

    /// <summary>
    /// Generates unlimited outcome variations from a single codon frequency
    /// Uses combinatorial expansion: Base × Context × Timing × Relationships × Environment
    /// </summary>
    public UnlimitedOutcome GenerateOutcome(TamagotchiAgent agent, int codonId)
    {
        var baseFrequency = _harmonicEngine.GetAvailableFrequencies(agent)
            .FirstOrDefault(f => f.CodonId == codonId);
        
        if (baseFrequency == null)
        {
            // Generate accessible frequency for agent's evolution level
            var availableFrequencies = _harmonicEngine.GetAvailableFrequencies(agent);
            baseFrequency = availableFrequencies[_random.Next(availableFrequencies.Count)];
        }

        // Apply unlimited expansion multipliers
        var contextVariation = SelectContextVariation(agent);
        var timingVariation = SelectTimingVariation();
        var relationshipVariation = SelectRelationshipVariation(agent);
        var environmentVariation = SelectEnvironmentVariation();
        var emotionalTone = SelectEmotionalTone(agent);

        var outcome = new UnlimitedOutcome
        {
            BaseCodon = baseFrequency.CodonId,
            BaseArchetype = baseFrequency.GameArchetype,
            FrequencyPattern = baseFrequency.CorePattern,
            
            // Unlimited variations
            ContextLayer = contextVariation,
            TimingLayer = timingVariation,
            RelationshipLayer = relationshipVariation,
            EnvironmentLayer = environmentVariation,
            EmotionalTone = emotionalTone,
            
            // Calculated outcome signature
            OutcomeSignature = CalculateOutcomeSignature(baseFrequency, contextVariation, 
                timingVariation, relationshipVariation, environmentVariation, emotionalTone),
            
            // Generated innocent scenario
            InnocentScenario = GenerateInnocentScenario(baseFrequency, contextVariation,
                timingVariation, relationshipVariation, environmentVariation, emotionalTone),
            
            AgentId = agent.Id,
            GeneratedAt = DateTime.Now
        };

        Console.WriteLine($"[Unlimited] Generated outcome {outcome.OutcomeSignature} for agent {agent.Name}");
        
        return outcome;
    }

    #region Outcome Multiplier System

    private Dictionary<string, OutcomeMultiplier> InitializeOutcomeMultipliers()
    {
        return new Dictionary<string, OutcomeMultiplier>
        {
            // Context variations (WHO is involved)
            ["context"] = new OutcomeMultiplier
            {
                Name = "Context",
                Variations = new List<string>
                {
                    "Solo introspective", "With best friend", "With family group", 
                    "With strangers", "With mentor figure", "With romantic interest",
                    "With work colleagues", "With neighbors", "With pets/animals",
                    "With children", "With elderly person", "With authority figure"
                }
            },
            
            // Timing variations (WHEN it occurs)
            ["timing"] = new OutcomeMultiplier
            {
                Name = "Timing", 
                Variations = new List<string>
                {
                    "Dawn breakthrough", "Midday routine", "Evening reflection",
                    "Late night insight", "Weekend leisure", "Holiday celebration",
                    "Seasonal transition", "Birthday significance", "Anniversary moment",
                    "Crisis timing", "Perfect synchronicity", "Unexpected surprise"
                }
            },
            
            // Relationship variations (WHAT connections are activated)
            ["relationship"] = new OutcomeMultiplier
            {
                Name = "Relationship",
                Variations = new List<string>
                {
                    "Harmony building", "Boundary setting", "Conflict resolution",
                    "Trust deepening", "Vulnerability sharing", "Joy celebration",
                    "Support offering", "Wisdom receiving", "Growth witnessing",
                    "Healing facilitation", "Creative collaboration", "Soul recognition"
                }
            },
            
            // Environment variations (WHERE it happens)
            ["environment"] = new OutcomeMultiplier
            {
                Name = "Environment",
                Variations = new List<string>
                {
                    "Cozy home space", "Natural outdoor setting", "Bustling public area",
                    "Quiet sacred space", "Creative workshop", "Community gathering",
                    "Travel adventure", "Familiar neighborhood", "New territory",
                    "Childhood memory place", "Future vision space", "Liminal threshold"
                }
            },
            
            // Emotional tone (HOW it feels)
            ["emotional"] = new OutcomeMultiplier
            {
                Name = "Emotional",
                Variations = new List<string>
                {
                    "Playful lightness", "Gentle compassion", "Excited anticipation",
                    "Peaceful contentment", "Curious exploration", "Grateful appreciation",
                    "Confident empowerment", "Tender vulnerability", "Joyful celebration",
                    "Wise understanding", "Creative inspiration", "Loving connection"
                }
            }
        };
    }

    private Dictionary<string, SentenceTemplate> InitializeSentenceTemplates()
    {
        // Placeholder for user's sentence structure and calculations
        // This will be replaced when they provide their framework
        return new Dictionary<string, SentenceTemplate>
        {
            ["basic"] = new SentenceTemplate
            {
                Pattern = "{Agent} experiences {BaseArchetype} through {Context} during {Timing} with {EmotionalTone}",
                CalculationWeight = 1.0f
            },
            ["advanced"] = new SentenceTemplate
            {
                Pattern = "In {Environment}, {Agent} navigates {BaseArchetype} by {RelationshipLayer} while {TimingLayer} brings {EmotionalTone}",
                CalculationWeight = 1.5f
            }
        };
    }

    #endregion

    #region Selection Methods

    private string SelectContextVariation(TamagotchiAgent agent)
    {
        var contexts = _multipliers["context"].Variations;
        
        // Weight selection based on agent's social needs and evolution
        if (agent.Resonance > 70) // High resonance agents prefer meaningful connections
        {
            var meaningfulContexts = contexts.Where(c => 
                c.Contains("mentor") || c.Contains("romantic") || c.Contains("family")).ToList();
            return meaningfulContexts.Any() ? meaningfulContexts[_random.Next(meaningfulContexts.Count)] : contexts[_random.Next(contexts.Count)];
        }
        
        return contexts[_random.Next(contexts.Count)];
    }

    private string SelectTimingVariation()
    {
        var timings = _multipliers["timing"].Variations;
        
        // Could weight based on real time, agent circadian rhythms, etc.
        var currentHour = DateTime.Now.Hour;
        if (currentHour < 6) return "Dawn breakthrough";
        if (currentHour < 12) return "Midday routine";  
        if (currentHour < 18) return "Evening reflection";
        return "Late night insight";
    }

    private string SelectRelationshipVariation(TamagotchiAgent agent)
    {
        var relationships = _multipliers["relationship"].Variations;
        
        // Weight based on agent's current needs and personality
        if (agent.Energy < 50) // Low energy agents need support
        {
            var supportiveRelations = relationships.Where(r => 
                r.Contains("support") || r.Contains("healing") || r.Contains("receiving")).ToList();
            return supportiveRelations.Any() ? supportiveRelations[_random.Next(supportiveRelations.Count)] : relationships[_random.Next(relationships.Count)];
        }
        
        return relationships[_random.Next(relationships.Count)];
    }

    private string SelectEnvironmentVariation()
    {
        var environments = _multipliers["environment"].Variations;
        return environments[_random.Next(environments.Count)];
    }

    private string SelectEmotionalTone(TamagotchiAgent agent)
    {
        var emotions = _multipliers["emotional"].Variations;
        
        // Weight based on agent happiness and current mood
        if (agent.Happiness > 80) // Happy agents get more celebratory tones
        {
            var joyfulEmotions = emotions.Where(e => 
                e.Contains("joy") || e.Contains("celebration") || e.Contains("excited")).ToList();
            return joyfulEmotions.Any() ? joyfulEmotions[_random.Next(joyfulEmotions.Count)] : emotions[_random.Next(emotions.Count)];
        }
        
        return emotions[_random.Next(emotions.Count)];
    }

    #endregion

    #region Outcome Generation

    private string CalculateOutcomeSignature(FrequencyAnchor baseFreq, string context, string timing, 
        string relationship, string environment, string emotion)
    {
        // Create unique signature for this specific outcome combination
        // This will be enhanced with user's calculation framework
        var hash = $"{baseFreq.CodonId}-{context.GetHashCode()}-{timing.GetHashCode()}-{relationship.GetHashCode()}-{environment.GetHashCode()}-{emotion.GetHashCode()}";
        return $"O-{Math.Abs(hash.GetHashCode()) % 69120}"; // Map to lattice position
    }

    private string GenerateInnocentScenario(FrequencyAnchor baseFreq, string context, string timing,
        string relationship, string environment, string emotion)
    {
        // Use base archetype pattern and multiply through all variations
        var basePattern = baseFreq.SimsManifest;
        
        // Apply innocent scenario generation - awaiting user's sentence structure
        var scenario = $"During {timing.ToLower()}, while in {environment.ToLower()}, " +
                      $"you find yourself {basePattern} {context.ToLower()}. " +
                      $"The experience involves {relationship.ToLower()} with {emotion.ToLower()}.";
        
        return MakeInnocent(scenario);
    }

    private string MakeInnocent(string scenario)
    {
        // Apply innocence filters - keep everything light and playful
        return scenario
            .Replace("conflict", "friendly disagreement")
            .Replace("crisis", "interesting challenge")
            .Replace("problem", "puzzle") 
            .Replace("difficult", "engaging")
            .Replace("failure", "learning opportunity")
            .Replace("mistake", "discovery");
    }

    #endregion

    /// <summary>
    /// Calculate total possible outcomes (will be enhanced with user's calculations)
    /// </summary>
    public long CalculateTotalPossibleOutcomes()
    {
        // Base calculation - will be refined with user's framework
        var contextVariations = _multipliers["context"].Variations.Count;
        var timingVariations = _multipliers["timing"].Variations.Count;  
        var relationshipVariations = _multipliers["relationship"].Variations.Count;
        var environmentVariations = _multipliers["environment"].Variations.Count;
        var emotionalVariations = _multipliers["emotional"].Variations.Count;
        
        var baseOutcomes = 64L; // 64 codons
        var totalCombinations = baseOutcomes * contextVariations * timingVariations * 
                               relationshipVariations * environmentVariations * emotionalVariations;
        
        Console.WriteLine($"[Unlimited] Total possible outcomes: {totalCombinations:N0}");
        return totalCombinations;
    }
}

#region Data Models

public class UnlimitedOutcome
{
    public int BaseCodon { get; set; }
    public string BaseArchetype { get; set; } = string.Empty;
    public string FrequencyPattern { get; set; } = string.Empty;
    
    // Multiplier layers for unlimited variation
    public string ContextLayer { get; set; } = string.Empty;        // WHO
    public string TimingLayer { get; set; } = string.Empty;         // WHEN  
    public string RelationshipLayer { get; set; } = string.Empty;   // WHAT connections
    public string EnvironmentLayer { get; set; } = string.Empty;    // WHERE
    public string EmotionalTone { get; set; } = string.Empty;       // HOW it feels
    
    // Generated unique signature and scenario
    public string OutcomeSignature { get; set; } = string.Empty;
    public string InnocentScenario { get; set; } = string.Empty;
    
    public string AgentId { get; set; } = string.Empty;
    public DateTime GeneratedAt { get; set; }
}

public class OutcomeMultiplier
{
    public string Name { get; set; } = string.Empty;
    public List<string> Variations { get; set; } = new();
}

public class SentenceTemplate
{
    public string Pattern { get; set; } = string.Empty;
    public float CalculationWeight { get; set; } = 1.0f;
    // Will be expanded with user's calculation framework
}

#endregion