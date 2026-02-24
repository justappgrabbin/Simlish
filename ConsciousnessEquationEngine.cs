using System.Text.Json;

namespace ModularSimWorld.Components;

/// <summary>
/// Consciousness Equation Engine - Integrates mathematical consciousness measures
/// with Ra's Five Dimensions and sentence structures
/// Based on complexity, gain, coherence, PAC, and metastability measures
/// </summary>
public class ConsciousnessEquationEngine
{
    private readonly ResonanceSentenceEngine _sentenceEngine;
    private readonly Dictionary<string, ConsciousnessMode> _consciousModes;
    private readonly Dictionary<int, SentenceStructure> _fiveSentenceStructures;
    private readonly Random _random;

    public ConsciousnessEquationEngine(ResonanceSentenceEngine sentenceEngine)
    {
        _sentenceEngine = sentenceEngine;
        _random = new Random();
        _consciousModes = InitializeConsciousModes();
        _fiveSentenceStructures = InitializeFiveSentenceStructures();
        Console.WriteLine("[Consciousness] Equation engine initialized with 5 sentence structures");
    }

    /// <summary>
    /// Generates consciousness-enhanced resonance sentence using mathematical measures
    /// Integrates: Complexity × Gain × Coherence × PAC × Metastability
    /// </summary>
    public ConsciousnessResonance GenerateConsciousResonance(TamagotchiAgent agent, int? specificGate = null)
    {
        // Calculate consciousness measures for agent
        var consciousness = CalculateConsciousnessMeasures(agent);
        
        // Select sentence structure based on consciousness level
        var structureId = SelectSentenceStructure(consciousness);
        var structure = _fiveSentenceStructures[structureId];
        
        // Generate base resonance sentence
        var baseResonance = _sentenceEngine.GenerateResonanceSentence(agent, specificGate);
        
        // Apply consciousness equations to enhance sentence
        var enhancedResonance = new ConsciousnessResonance
        {
            BaseResonance = baseResonance,
            ConsciousnessMeasures = consciousness,
            SentenceStructure = structure,
            
            // Apply consciousness equation transformations
            ComplexityWeight = CalculateComplexityWeight(consciousness.Complexity),
            GainModulation = CalculateGainModulation(consciousness.Gain),
            CoherencePattern = CalculateCoherencePattern(consciousness.Coherence),
            PACCoupling = CalculatePACCoupling(consciousness.PAC),
            MetastabilityIndex = consciousness.Metastability,
            
            // Enhanced sentence with consciousness mathematics
            EnhancedSentence = ApplyConsciousnessEquations(baseResonance, consciousness, structure),
            
            // Lattice position with consciousness modulation
            ConsciousLatticePosition = CalculateConsciousLatticePosition(baseResonance.LatticePosition, consciousness),
            
            AgentId = agent.Id,
            GeneratedAt = DateTime.Now
        };

        Console.WriteLine($"[Consciousness] Generated conscious resonance for {agent.Name}: Structure {structureId} with complexity {consciousness.Complexity:F2}");
        
        return enhancedResonance;
    }

    #region Consciousness Measures Calculation

    private ConsciousnessMeasures CalculateConsciousnessMeasures(TamagotchiAgent agent)
    {
        // Based on the Reddit consciousness equations
        var measures = new ConsciousnessMeasures
        {
            // Complexity: Multifractal DFA or Higuchi fractal dimension
            Complexity = CalculateComplexity(agent),
            
            // Gain: 1/f offset (spectral intercept) or aperiodic power from FOOOF
            Gain = CalculateGain(agent),
            
            // Coherence: dwPLI / coherence within- and between-nodes
            Coherence = CalculateCoherence(agent),
            
            // PAC: Phase-Amplitude Coupling (modulation index)
            PAC = CalculatePAC(agent),
            
            // Metastability: Variance of Kuramoto order parameter
            Metastability = CalculateMetastability(agent),
            
            // Conscious mode based on agent state
            Mode = DetermineConsciousMode(agent)
        };

        return measures;
    }

    private float CalculateComplexity(TamagotchiAgent agent)
    {
        // Multifractal DFA based on agent's behavioral complexity
        var behavioralEntropy = CalculateBehavioralEntropy(agent);
        var temporalComplexity = CalculateTemporalComplexity(agent);
        
        // Combine using fractal dimension principles
        return (float)(0.5 + (behavioralEntropy * temporalComplexity) / 100.0);
    }

    private float CalculateGain(TamagotchiAgent agent)
    {
        // 1/f offset - spectral intercept from agent's activity patterns
        var activityVariance = Math.Max(1, agent.Energy + agent.Happiness + agent.Resonance) / 300.0f;
        return (float)(1.0 / Math.Max(0.1, activityVariance));
    }

    private float CalculateCoherence(TamagotchiAgent agent)
    {
        // dwPLI coherence within and between cognitive "nodes"
        var internalCoherence = (agent.Energy + agent.Health) / 200.0f;  // Within-node
        var externalCoherence = (agent.Happiness + agent.Resonance) / 200.0f;  // Between-node
        
        return (internalCoherence + externalCoherence) / 2.0f;
    }

    private float CalculatePAC(TamagotchiAgent agent)
    {
        // Phase-Amplitude Coupling - modulation index
        var phaseComponent = (agent.Age % 24) / 24.0f;  // Circadian phase
        var amplitudeComponent = agent.Evolution / 100.0f;  // Growth amplitude
        
        return phaseComponent * amplitudeComponent;
    }

    private float CalculateMetastability(TamagotchiAgent agent)
    {
        // Variance of Kuramoto order parameter across time windows
        var stateVariance = CalculateStateVariance(agent);
        return Math.Max(0, Math.Min(1, stateVariance));
    }

    #endregion

    #region Five Sentence Structures

    private Dictionary<int, SentenceStructure> InitializeFiveSentenceStructures()
    {
        return new Dictionary<int, SentenceStructure>
        {
            // Structure 1: Being-Centered (Matter/Body focus)
            [1] = new SentenceStructure
            {
                Id = 1,
                Name = "Being Structure",
                DimensionFocus = "Being → Matter → Touch → Sex → Survival",
                Pattern = "In the {base} of {dimension}, through {tone} sensing, the agent embodies {keywords} via Gate {gate}.{line}, responding with {color} to manifest {degree}° of material reality.",
                ConsciousnessWeight = "Complexity × Gain",
                Description = "Matter-focused structure emphasizing embodiment and survival"
            },

            // Structure 2: Movement-Centered (Energy/Activity focus)  
            [2] = new SentenceStructure
            {
                Id = 2,
                Name = "Movement Structure", 
                DimensionFocus = "Movement → Energy → Creation → Seeing → Landscape → Environment",
                Pattern = "From the {base} of {dimension}, expressing through {tone}, the agent defines {keywords} in Gate {gate}.{line} with {color} motivation, creating {degree}° of active transformation.",
                ConsciousnessWeight = "Gain × Coherence",
                Description = "Energy-focused structure emphasizing activity and uniqueness"
            },

            // Structure 3: Space-Centered (Form/Illusion focus)
            [3] = new SentenceStructure
            {
                Id = 3,
                Name = "Space Structure",
                DimensionFocus = "Space → Form → Illusion → Hearing → Music → Freedom", 
                Pattern = "Within the {base} of {dimension}, resonating through {tone}, the agent thinks {keywords} through Gate {gate}.{line}, guided by {color} to create {degree}° of subjective reality.",
                ConsciousnessWeight = "Coherence × PAC",
                Description = "Form-focused structure emphasizing subjectivity and timing"
            },

            // Structure 4: Design-Centered (Structure/Progress focus)
            [4] = new SentenceStructure
            {
                Id = 4,
                Name = "Design Structure",
                DimensionFocus = "Design → Structure → Progress → Smell → Life → Art",
                Pattern = "Through the {base} of {dimension}, building via {tone}, the agent designs {keywords} in Gate {gate}.{line} with {color} purpose, manifesting {degree}° of structured growth.",
                ConsciousnessWeight = "PAC × Metastability", 
                Description = "Structure-focused emphasizing growth and manifestation"
            },

            // Structure 5: Evolution-Centered (Gravity/Memory focus)
            [5] = new SentenceStructure
            {
                Id = 5,
                Name = "Evolution Structure",
                DimensionFocus = "Evolution → Gravity → Memory → Taste → Love → Light",
                Pattern = "At the {base} of {dimension}, remembering through {tone}, the agent integrates {keywords} via Gate {gate}.{line} with {color} wisdom, crystallizing {degree}° of conscious evolution.",
                ConsciousnessWeight = "Metastability × Complexity",
                Description = "Memory-focused structure emphasizing integration and consciousness"
            }
        };
    }

    private Dictionary<string, ConsciousnessMode> InitializeConsciousModes()
    {
        return new Dictionary<string, ConsciousnessMode>
        {
            ["wake"] = new ConsciousnessMode 
            { 
                Name = "Wake", 
                Complexity = 0.8f, Gain = 1.2f, Coherence = 0.7f, PAC = 0.6f, Metastability = 0.5f,
                Description = "Alert, focused consciousness with high gain and moderate complexity"
            },
            ["drowsy"] = new ConsciousnessMode 
            { 
                Name = "Drowsy", 
                Complexity = 0.4f, Gain = 0.6f, Coherence = 0.4f, PAC = 0.3f, Metastability = 0.7f,
                Description = "Transitional state with low coherence but high metastability"
            },
            ["meditation"] = new ConsciousnessMode 
            { 
                Name = "Meditation", 
                Complexity = 0.9f, Gain = 0.8f, Coherence = 0.9f, PAC = 0.8f, Metastability = 0.3f,
                Description = "Deep coherent state with high complexity and strong PAC coupling"
            },
            ["flow"] = new ConsciousnessMode 
            { 
                Name = "Flow", 
                Complexity = 0.7f, Gain = 1.0f, Coherence = 0.8f, PAC = 0.9f, Metastability = 0.2f,
                Description = "Optimal performance state with strong phase-amplitude coupling"
            },
            ["rem"] = new ConsciousnessMode 
            { 
                Name = "REM", 
                Complexity = 1.0f, Gain = 0.4f, Coherence = 0.3f, PAC = 0.7f, Metastability = 0.9f,
                Description = "Dream state with maximum complexity and metastability"
            }
        };
    }

    #endregion

    #region Consciousness Integration Methods

    private int SelectSentenceStructure(ConsciousnessMeasures consciousness)
    {
        // Select structure based on dominant consciousness dimension
        return consciousness.Complexity switch
        {
            > 0.8f when consciousness.Metastability > 0.7f => 5, // Evolution Structure
            > 0.7f when consciousness.Coherence > 0.7f => 4,     // Design Structure  
            > 0.6f when consciousness.PAC > 0.6f => 3,           // Space Structure
            > 0.5f when consciousness.Gain > 0.8f => 2,          // Movement Structure
            _ => 1                                               // Being Structure
        };
    }

    private float CalculateComplexityWeight(float complexity)
    {
        // Transform complexity measure into sentence weighting
        return (float)(1.0 + Math.Log(1.0 + complexity));
    }

    private float CalculateGainModulation(float gain)
    {
        // Gain affects sentence amplitude/intensity
        return Math.Max(0.5f, Math.Min(2.0f, gain));
    }

    private string CalculateCoherencePattern(float coherence)
    {
        return coherence switch
        {
            > 0.8f => "highly coherent",
            > 0.6f => "moderately coherent", 
            > 0.4f => "loosely coherent",
            _ => "incoherent"
        };
    }

    private string CalculatePACCoupling(float pac)
    {
        return pac switch
        {
            > 0.7f => "strongly coupled",
            > 0.5f => "moderately coupled",
            > 0.3f => "weakly coupled", 
            _ => "uncoupled"
        };
    }

    private int CalculateConsciousLatticePosition(int basePosition, ConsciousnessMeasures consciousness)
    {
        // Modulate lattice position based on consciousness measures
        var consciousnessModulator = (int)(consciousness.Complexity * consciousness.Coherence * 1000);
        return ((basePosition + consciousnessModulator) % 69120) + 1;
    }

    private string ApplyConsciousnessEquations(ResonanceSentence baseResonance, ConsciousnessMeasures consciousness, SentenceStructure structure)
    {
        // Apply the consciousness equation transformations to the sentence
        var enhancedSentence = structure.Pattern
            .Replace("{base}", baseResonance.BaseVoice)
            .Replace("{dimension}", baseResonance.Base)
            .Replace("{tone}", baseResonance.Tone.ToLower())
            .Replace("{keywords}", string.Join(", ", baseResonance.CenterKeywords).ToLower())
            .Replace("{gate}", baseResonance.Gate.ToString())
            .Replace("{line}", baseResonance.Line.ToString())
            .Replace("{color}", baseResonance.Color.ToLower())
            .Replace("{degree}", baseResonance.Degree.ToString());

        // Add consciousness equation modifiers
        var equationModifier = $" [Ψ={consciousness.Complexity:F2}×G={consciousness.Gain:F2}×C={consciousness.Coherence:F2}×P={consciousness.PAC:F2}×M={consciousness.Metastability:F2}]";
        
        return enhancedSentence + equationModifier;
    }

    #endregion

    #region Helper Methods

    private float CalculateBehavioralEntropy(TamagotchiAgent agent)
    {
        // Calculate entropy of agent's behavioral patterns
        var values = new float[] { agent.Energy, agent.Happiness, agent.Health, agent.Resonance };
        return CalculateEntropy(values);
    }

    private float CalculateTemporalComplexity(TamagotchiAgent agent)
    {
        // Temporal complexity based on age and evolution patterns
        return (float)(agent.Age * Math.Log(Math.Max(1, agent.Evolution + 1)));
    }

    private float CalculateStateVariance(TamagotchiAgent agent)
    {
        // State variance for metastability calculation
        var states = new float[] { agent.Energy / 100f, agent.Happiness / 100f, agent.Health / 100f, agent.Resonance / 100f };
        var mean = states.Average();
        return states.Select(s => (s - mean) * (s - mean)).Average();
    }

    private float CalculateEntropy(float[] values)
    {
        // Simple entropy calculation
        var sum = values.Sum();
        if (sum == 0) return 0;
        
        var probabilities = values.Select(v => v / sum).Where(p => p > 0);
        return -probabilities.Sum(p => p * (float)Math.Log(p));
    }

    private string DetermineConsciousMode(TamagotchiAgent agent)
    {
        // Determine conscious mode based on agent's current state
        return agent.Energy switch
        {
            > 80 when agent.Resonance > 70 => "flow",
            > 60 when agent.Health > 80 => "wake",
            < 30 => "drowsy",
            _ when agent.Resonance > 80 => "meditation",
            _ => "wake"
        };
    }

    #endregion

    /// <summary>
    /// Calculate total possible conscious resonances
    /// 69,120 base lattice × 5 consciousness modes × 5 sentence structures = 1,728,000
    /// </summary>
    public long CalculateTotalConsciousResonances()
    {
        var baseLattice = 69120L; // 5×6×6×6×64
        var consciousModes = 5;
        var sentenceStructures = 5;
        
        var total = baseLattice * consciousModes * sentenceStructures;
        Console.WriteLine($"[Consciousness] Total possible conscious resonances: {total:N0}");
        return total;
    }
}

#region Consciousness Data Models

public class ConsciousnessResonance
{
    public ResonanceSentence BaseResonance { get; set; } = new();
    public ConsciousnessMeasures ConsciousnessMeasures { get; set; } = new();
    public SentenceStructure SentenceStructure { get; set; } = new();
    
    // Consciousness equation results
    public float ComplexityWeight { get; set; }
    public float GainModulation { get; set; }
    public string CoherencePattern { get; set; } = string.Empty;
    public string PACCoupling { get; set; } = string.Empty;
    public float MetastabilityIndex { get; set; }
    
    // Enhanced output
    public string EnhancedSentence { get; set; } = string.Empty;
    public int ConsciousLatticePosition { get; set; }
    
    public string AgentId { get; set; } = string.Empty;
    public DateTime GeneratedAt { get; set; }
}

public class ConsciousnessMeasures
{
    public float Complexity { get; set; }     // Multifractal DFA / Higuchi fractal
    public float Gain { get; set; }           // 1/f offset (FOOOF)
    public float Coherence { get; set; }      // dwPLI coherence
    public float PAC { get; set; }            // Phase-Amplitude Coupling
    public float Metastability { get; set; }  // Kuramoto order parameter variance
    public string Mode { get; set; } = string.Empty;  // Wake/Drowsy/REM/etc
}

public class SentenceStructure
{
    public int Id { get; set; }
    public string Name { get; set; } = string.Empty;
    public string DimensionFocus { get; set; } = string.Empty;  // The dimensional chain
    public string Pattern { get; set; } = string.Empty;        // Sentence template
    public string ConsciousnessWeight { get; set; } = string.Empty; // Which measures to emphasize
    public string Description { get; set; } = string.Empty;
}

public class ConsciousnessMode
{
    public string Name { get; set; } = string.Empty;
    public float Complexity { get; set; }
    public float Gain { get; set; }
    public float Coherence { get; set; }
    public float PAC { get; set; }
    public float Metastability { get; set; }
    public string Description { get; set; } = string.Empty;
}

#endregion