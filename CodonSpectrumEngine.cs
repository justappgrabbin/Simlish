using System.Text.Json;

namespace ModularSimWorld.Components;

/// <summary>
/// Codon Spectrum Engine - 64 codons as resonant frequencies within The Sims
/// Each codon represents game mechanics from Pong (binary) to meta-games (consciousness)
/// Hidden in plain sight as Sims activities with alien exaggeration
/// </summary>
public class CodonSpectrumEngine
{
    private readonly Dictionary<int, CodonTemplate> _codonSpectrum;
    private readonly Dictionary<string, AlienExaggeration> _alienEffects;
    private readonly Random _random;

    public CodonSpectrumEngine()
    {
        _random = new Random();
        _codonSpectrum = InitializeCodonSpectrum();
        _alienEffects = InitializeAlienEffects();
        Console.WriteLine("[Spectrum] 64-Codon harmonic spectrum initialized with alien exaggeration");
    }

    /// <summary>
    /// Activates a codon event for an agent based on their evolution level and resonance
    /// </summary>
    public CodonEventResult ActivateCodon(TamagotchiAgent agent, int? specificCodon = null)
    {
        // Select codon based on agent evolution and resonance
        var codonId = specificCodon ?? SelectHarmonicCodon(agent);
        var codonTemplate = _codonSpectrum[codonId];
        
        // Apply alien exaggeration based on tier
        var tier = CalculateEventTier(agent, codonTemplate);
        var exaggeration = GetAlienExaggeration(codonTemplate, tier);
        
        var eventResult = new CodonEventResult
        {
            CodonId = $"C-{codonId:D2}",
            Name = codonTemplate.Name,
            Archetype = codonTemplate.Archetype,
            Center = codonTemplate.Center,
            Tier = tier,
            
            SimsScenario = new SimsScenario
            {
                Premise = ApplyExaggeration(codonTemplate.SimsScenario.Premise, exaggeration),
                SetDressing = codonTemplate.SimsScenario.SetDressing.Concat(exaggeration.AdditionalProps).ToList(),
                Triggers = codonTemplate.SimsScenario.Triggers
            },
            
            EventChain = codonTemplate.EventChain.ToList(),
            Choices = codonTemplate.Choices.ToList(),
            AutonomyShift = codonTemplate.AutonomyShift,
            MatterCapture = codonTemplate.MatterCapture,
            
            CurrentStep = 0,
            AgentId = agent.Id,
            ActivatedAt = DateTime.Now
        };

        Console.WriteLine($"[Spectrum] Agent {agent.Name} activated {eventResult.CodonId}: {eventResult.Name} (Tier {tier})");
        Console.WriteLine($"[Spectrum] Scenario: {eventResult.SimsScenario.Premise}");
        
        return eventResult;
    }

    /// <summary>
    /// Process a step in the codon event chain
    /// </summary>
    public CodonStepResult ProcessEventStep(CodonEventResult codonEvent, string choiceId)
    {
        if (codonEvent.CurrentStep >= codonEvent.EventChain.Count)
            return new CodonStepResult { Success = false, Message = "Event already completed!" };

        var currentStep = codonEvent.EventChain[codonEvent.CurrentStep];
        var selectedChoice = codonEvent.Choices.FirstOrDefault(c => c.Id == choiceId);
        
        if (selectedChoice == null)
            return new CodonStepResult { Success = false, Message = "Invalid choice!" };

        var stepResult = new CodonStepResult
        {
            Success = true,
            ActionTaken = selectedChoice.Label,
            StepDescription = currentStep.Goal,
            Message = $"✨ {currentStep.Cue}: {selectedChoice.Label} - {currentStep.Goal}"
        };

        codonEvent.CurrentStep++;
        codonEvent.CompletedSteps.Add($"{currentStep.Cue}: {selectedChoice.Label}");

        // Check if event is complete
        if (codonEvent.CurrentStep >= codonEvent.EventChain.Count)
        {
            stepResult.QuestComplete = true;
            stepResult.ElementalBonus = true;
            stepResult.Message += $"\n🏆 Event Complete! {codonEvent.MatterCapture.Artifact} artifact captured!";
        }

        return stepResult;
    }

    #region Codon Spectrum Initialization

    private Dictionary<int, CodonTemplate> InitializeCodonSpectrum()
    {
        var spectrum = new Dictionary<int, CodonTemplate>();

        // Early Codons (1-15): Binary/Reaction Games - Simple Sims Activities
        spectrum[1] = CreateCodonTemplate(1, "Conflict", "Warrior", "Root",
            "The living room re-zones into a floating arena for honor combat",
            ["Floor rings", "Gravity flicker", "Alien referee drone", "Storm weather"],
            ["Rival arrives angry", "Fight or Flight moodlet active"],
            CreateBinaryEventChain(),
            CreateConflictChoices(),
            "Blood Seal", ["Steeled Nerves"], ["Second Wind"]);

        spectrum[2] = CreateCodonTemplate(2, "Focus", "Hunter", "Ajna", 
            "Sky tears open, micro-asteroids drift through your lot with telescope HUD",
            ["Slow-time bubble", "UI scanlines", "Soft synth pulse"],
            ["Distracted moodlet while studying/repairing"],
            CreateFocusEventChain(),
            CreateFocusChoices(),
            "Eye of the Hunter", ["Noise Gate"], ["Bullet Time"]);

        spectrum[3] = CreateCodonTemplate(3, "Trickery", "Trickster", "Spleen",
            "Roaming Surveillance Orbs patrol the neighborhood - stealth snack mission",
            ["Security orbs", "Stealth shadows", "Scanning beams"],
            ["Hunger motive while Surveillance Orbs active"],
            CreateTrickeryEventChain(),
            CreateTrickeryChoices(),
            "Echo Lens", ["Micro-tell overlay"], ["Reveal Motive"]);

        // Mid-Early (16-25): Pattern Recognition - Tetris-style Sims Activities  
        spectrum[20] = CreateCodonTemplate(20, "Harmony", "Architect", "G",
            "Aliens dump modular furniture from sky - optimize household flow",
            ["Falling furniture", "Snap-lock sounds", "Flow visualization"],
            ["Cluttered room", "Poor routing"],
            CreateHarmonyEventChain(),
            CreateHarmonyChoices(),
            "Tone Brick", ["Auto-routing optimization"], ["Perfect Flow"]);

        // Mid-Range (26-40): Multi-element Coordination - Complex Sims Challenges
        spectrum[35] = CreateCodonTemplate(35, "Nurture", "Caretaker", "Solar Plexus",
            "Garden mutates into alien mood flora - emotional herb harvesting",
            ["Mood-colored plants", "Emotional auras", "Alien soil"],
            ["Garden needs tending", "Household emotional imbalance"],
            CreateNurtureEventChain(),
            CreateNurtureChoices(),
            "Star Infuser", ["Emotion regulation"], ["Negative to Insight"]);

        // Advanced Codons (41-50): Complex Systems - SimCity-style
        spectrum[45] = CreateCodonTemplate(45, "Leadership", "Commander", "Heart",
            "Neighborhood council emergency - alien bureaucracy summons you",
            ["Floating council chamber", "Alien delegates", "Quantum voting booths"],
            ["Community crisis", "Leadership opportunity"],
            CreateLeadershipEventChain(),
            CreateLeadershipChoices(),
            "Authority Crown", ["Natural command"], ["Rally Community"]);

        // Deep Simulation (51-60): Life Mastery - Full Sims Complexity
        spectrum[55] = CreateCodonTemplate(55, "Abundance", "Life Master", "Multiple Centers",
            "Reality glitches reveal you control multiple Sim households simultaneously",
            ["Multi-dimensional views", "Timeline branches", "Reality controls"],
            ["Multiple life streams active"],
            CreateAbundanceEventChain(),
            CreateAbundanceChoices(),
            "Existence Key", ["Multi-life awareness"], ["Reality Edit"]);

        // Meta-Games (61-64): Consciousness Simulation
        spectrum[64] = CreateCodonTemplate(64, "Completion", "The Awakened", "Crown",
            "You realize you are both the Sim and the Player in an infinite loop",
            ["Fourth wall breaks", "Interface becomes visible", "Meta-commentary"],
            ["Consciousness breakthrough"],
            CreateAwakeningEventChain(),
            CreateAwakeningChoices(),
            "Observer's Paradox", ["See the code"], ["Exit/Enter Loop"]);

        // Fill remaining codons with generated templates
        for (int i = 4; i <= 63; i++)
        {
            if (!spectrum.ContainsKey(i))
            {
                spectrum[i] = GenerateCodonFromFrequency(i);
            }
        }

        return spectrum;
    }

    private CodonTemplate CreateCodonTemplate(int id, string name, string archetype, string center,
        string premise, List<string> setDressing, List<string> triggers,
        List<EventStep> eventChain, List<EventChoice> choices,
        string artifact, List<string> passive, List<string> active)
    {
        return new CodonTemplate
        {
            CodonId = id,
            Name = $"Codon of {name}",
            Archetype = archetype,
            Center = center,
            SimsScenario = new SimsScenario
            {
                Premise = premise,
                SetDressing = setDressing,
                Triggers = triggers
            },
            EventChain = eventChain,
            Choices = choices,
            AutonomyShift = new AutonomyShift
            {
                AskThreshold = 0.10f,
                AutoThreshold = 0.25f,
                SignatureThreshold = 0.75f
            },
            MatterCapture = new MatterCapture
            {
                Artifact = artifact,
                Passive = passive,
                Active = active,
                Visual = $"Alien {name.ToLower()} sigil"
            }
        };
    }

    private List<EventStep> CreateBinaryEventChain() => new()
    {
        new() { Step = 1, Cue = "Seen", Goal = "Clock the rival's tells (red/grey emote flickers)" },
        new() { Step = 2, Cue = "Felt", Goal = "Ride adrenaline without lashing out" },
        new() { Step = 3, Cue = "Heard", Goal = "Choose battle-cry or boundary phrase" },
        new() { Step = 4, Cue = "MatterCapture", Goal = "Stamp Blood Seal on arena floor" }
    };

    private List<EventChoice> CreateConflictChoices() => new()
    {
        new() { Id = "A", Label = "Stand & Name Terms", 
               Effects = new ChoiceEffects { 
                   Motives = new() { ["Safety"] = 5 }, 
                   Weights = new() { ["confront"] = 0.05f } } },
        new() { Id = "B", Label = "De-escalate Ritual", 
               Effects = new ChoiceEffects { 
                   Motives = new() { ["Connection"] = 10 }, 
                   Weights = new() { ["reconcile"] = 0.05f } } },
        new() { Id = "C", Label = "Swing First", 
               Effects = new ChoiceEffects { 
                   Motives = new() { ["Vitality"] = 10, ["Safety"] = -10 }, 
                   Weights = new() { ["aggression"] = 0.07f } } }
    };

    private List<EventStep> CreateFocusEventChain() => new()
    {
        new() { Step = 1, Cue = "Seen", Goal = "Track three moving glints without breaking action" },
        new() { Step = 2, Cue = "Felt", Goal = "Maintain task through interruptions" },
        new() { Step = 3, Cue = "Heard", Goal = "Align to soft metronome beats" },
        new() { Step = 4, Cue = "MatterCapture", Goal = "Focus lens clicks - Eye imprints" }
    };

    private List<EventChoice> CreateFocusChoices() => new()
    {
        new() { Id = "A", Label = "Single-Task Lock", 
               Effects = new ChoiceEffects { 
                   Motives = new() { ["Clarity"] = 15 }, 
                   Weights = new() { ["single_focus"] = 0.06f } } },
        new() { Id = "B", Label = "Rhythmic Sprints", 
               Effects = new ChoiceEffects { 
                   Motives = new() { ["Vitality"] = 5 }, 
                   Weights = new() { ["flow"] = 0.05f } } },
        new() { Id = "C", Label = "Delegate Distractions", 
               Effects = new ChoiceEffects { 
                   Motives = new() { ["Connection"] = 5 }, 
                   Weights = new() { ["systems_thinking"] = 0.04f } } }
    };

    // Additional event chain creators for other codons...
    private List<EventStep> CreateTrickeryEventChain() => new()
    {
        new() { Step = 1, Cue = "Seen", Goal = "Spot orb patrol patterns" },
        new() { Step = 2, Cue = "Felt", Goal = "Move with stealth timing" },
        new() { Step = 3, Cue = "Heard", Goal = "Listen for scanning frequency" },
        new() { Step = 4, Cue = "MatterCapture", Goal = "Echo lens activates - see through deception" }
    };

    private List<EventChoice> CreateTrickeryChoices() => new()
    {
        new() { Id = "A", Label = "Hide & Wait", Effects = new ChoiceEffects { Weights = new() { ["stealth"] = 0.05f } } },
        new() { Id = "B", Label = "RC Car Decoy", Effects = new ChoiceEffects { Weights = new() { ["creative_problem"] = 0.06f } } },
        new() { Id = "C", Label = "Confess to Orbs", Effects = new ChoiceEffects { Weights = new() { ["radical_honesty"] = 0.04f } } }
    };

    private List<EventStep> CreateHarmonyEventChain() => new()
    {
        new() { Step = 1, Cue = "Seen", Goal = "Visualize optimal room flow" },
        new() { Step = 2, Cue = "Felt", Goal = "Sense spatial harmony" },
        new() { Step = 3, Cue = "Heard", Goal = "Listen for flow disruptions" },
        new() { Step = 4, Cue = "MatterCapture", Goal = "Tone brick resonates - perfect routing achieved" }
    };

    private List<EventChoice> CreateHarmonyChoices() => new()
    {
        new() { Id = "A", Label = "Form Follows Function", Effects = new ChoiceEffects { Weights = new() { ["systems"] = 0.06f } } },
        new() { Id = "B", Label = "Beauty First", Effects = new ChoiceEffects { Weights = new() { ["aesthetic"] = 0.05f } } },
        new() { Id = "C", Label = "Chaos Temple", Effects = new ChoiceEffects { Weights = new() { ["embrace_entropy"] = 0.04f } } }
    };

    private List<EventStep> CreateNurtureEventChain() => new()
    {
        new() { Step = 1, Cue = "Seen", Goal = "Identify mood-plants by color aura" },
        new() { Step = 2, Cue = "Felt", Goal = "Attune to emotional frequencies" },
        new() { Step = 3, Cue = "Heard", Goal = "Listen to plant emotional songs" },
        new() { Step = 4, Cue = "MatterCapture", Goal = "Star infuser captures emotional essence" }
    };

    private List<EventChoice> CreateNurtureChoices() => new()
    {
        new() { Id = "A", Label = "Brew Soothing Tea", Effects = new ChoiceEffects { Weights = new() { ["co_regulate"] = 0.06f } } },
        new() { Id = "B", Label = "Bottle & Sell", Effects = new ChoiceEffects { Weights = new() { ["commodify"] = 0.05f } } },
        new() { Id = "C", Label = "Plant for Neighbors", Effects = new ChoiceEffects { Weights = new() { ["community"] = 0.05f } } }
    };

    private List<EventStep> CreateLeadershipEventChain() => new()
    {
        new() { Step = 1, Cue = "Seen", Goal = "Assess alien council dynamics" },
        new() { Step = 2, Cue = "Felt", Goal = "Channel natural authority" },
        new() { Step = 3, Cue = "Heard", Goal = "Speak with commanding presence" },
        new() { Step = 4, Cue = "MatterCapture", Goal = "Authority crown manifests - leadership recognized" }
    };

    private List<EventChoice> CreateLeadershipChoices() => new()
    {
        new() { Id = "A", Label = "Democratic Consensus", Effects = new ChoiceEffects { Weights = new() { ["collaborative"] = 0.06f } } },
        new() { Id = "B", Label = "Executive Decision", Effects = new ChoiceEffects { Weights = new() { ["decisive"] = 0.05f } } },
        new() { Id = "C", Label = "Delegate Authority", Effects = new ChoiceEffects { Weights = new() { ["empowering"] = 0.04f } } }
    };

    private List<EventStep> CreateAbundanceEventChain() => new()
    {
        new() { Step = 1, Cue = "Seen", Goal = "Perceive multiple reality streams" },
        new() { Step = 2, Cue = "Felt", Goal = "Embrace infinite possibilities" },
        new() { Step = 3, Cue = "Heard", Goal = "Listen to harmony of all lives" },
        new() { Step = 4, Cue = "MatterCapture", Goal = "Existence key unlocks - reality becomes malleable" }
    };

    private List<EventChoice> CreateAbundanceChoices() => new()
    {
        new() { Id = "A", Label = "Merge All Lives", Effects = new ChoiceEffects { Weights = new() { ["unity"] = 0.08f } } },
        new() { Id = "B", Label = "Choose One Path", Effects = new ChoiceEffects { Weights = new() { ["focus"] = 0.06f } } },
        new() { Id = "C", Label = "Dance Between Worlds", Effects = new ChoiceEffects { Weights = new() { ["fluidity"] = 0.07f } } }
    };

    private List<EventStep> CreateAwakeningEventChain() => new()
    {
        new() { Step = 1, Cue = "Seen", Goal = "See the game interface become visible" },
        new() { Step = 2, Cue = "Felt", Goal = "Experience being both player and played" },
        new() { Step = 3, Cue = "Heard", Goal = "Hear the code speaking to you" },
        new() { Step = 4, Cue = "MatterCapture", Goal = "Observer's paradox resolved - choose your level of the game" }
    };

    private List<EventChoice> CreateAwakeningChoices() => new()
    {
        new() { Id = "A", Label = "Remain in Simulation", Effects = new ChoiceEffects { Weights = new() { ["embodied"] = 0.10f } } },
        new() { Id = "B", Label = "Exit the Game", Effects = new ChoiceEffects { Weights = new() { ["transcendent"] = 0.10f } } },
        new() { Id = "C", Label = "Become the Game", Effects = new ChoiceEffects { Weights = new() { ["creator"] = 0.10f } } }
    };

    #endregion

    #region Harmonic Selection and Alien Effects

    private int SelectHarmonicCodon(TamagotchiAgent agent)
    {
        // Select based on agent evolution (determines complexity level they can handle)
        var evolutionTier = agent.Evolution switch
        {
            < 20 => "binary",     // Codons 1-15
            < 40 => "pattern",    // Codons 16-25  
            < 60 => "multi",      // Codons 26-40
            < 80 => "complex",    // Codons 41-50
            < 95 => "mastery",    // Codons 51-60
            _ => "meta"           // Codons 61-64
        };

        var codonRange = evolutionTier switch
        {
            "binary" => (1, 15),
            "pattern" => (16, 25),
            "multi" => (26, 40), 
            "complex" => (41, 50),
            "mastery" => (51, 60),
            "meta" => (61, 64),
            _ => (1, 64)
        };

        return _random.Next(codonRange.Item1, codonRange.Item2 + 1);
    }

    private int CalculateEventTier(TamagotchiAgent agent, CodonTemplate template)
    {
        // Higher evolution = higher tier events with more alien exaggeration
        return (agent.Evolution / 25) + 1; // Tier 1-4
    }

    private Dictionary<string, AlienExaggeration> InitializeAlienEffects()
    {
        return new Dictionary<string, AlienExaggeration>
        {
            ["gravity_flip"] = new() { Name = "Gravity Flip", AdditionalProps = ["Floating objects", "Inverted movement"] },
            ["time_dilation"] = new() { Name = "Time Bubble", AdditionalProps = ["Slow motion effects", "Temporal distortion"] },
            ["portal_appliances"] = new() { Name = "Portal Kitchen", AdditionalProps = ["Dimensional rifts", "Multi-space access"] },
            ["alien_court"] = new() { Name = "Alien Arbitration", AdditionalProps = ["Cosmic judges", "Universal law"] },
            ["sentient_objects"] = new() { Name = "Awakened Items", AdditionalProps = ["Talking furniture", "Conscious appliances"] }
        };
    }

    private AlienExaggeration GetAlienExaggeration(CodonTemplate template, int tier)
    {
        var availableEffects = _alienEffects.Values.ToList();
        var selectedEffect = availableEffects[_random.Next(availableEffects.Count)];
        
        // Scale effect intensity by tier
        selectedEffect.Intensity = tier;
        return selectedEffect;
    }

    private string ApplyExaggeration(string premise, AlienExaggeration exaggeration)
    {
        return premise + $" (Enhanced by {exaggeration.Name} - Tier {exaggeration.Intensity})";
    }

    private CodonTemplate GenerateCodonFromFrequency(int codonId)
    {
        // Generate codon based on its position in the spectrum
        var frequency = GetFrequencyType(codonId);
        var name = GetCodonNameFromFrequency(codonId, frequency);
        
        return CreateCodonTemplate(codonId, name, "Generated", "Variable",
            $"A {frequency} frequency event manifests in your Sim's reality",
            ["Reality glitches", "Frequency waves"],
            ["Resonance alignment"],
            CreateGenericEventChain(),
            CreateGenericChoices(),
            $"{name} Artifact", ["Passive effect"], ["Active ability"]);
    }

    private string GetFrequencyType(int codon)
    {
        return codon switch
        {
            <= 15 => "Binary",
            <= 25 => "Pattern", 
            <= 40 => "Multi-dimensional",
            <= 50 => "Complex systems",
            <= 60 => "Life mastery", 
            _ => "Meta-consciousness"
        };
    }

    private string GetCodonNameFromFrequency(int codon, string frequency)
    {
        return $"{frequency} Resonance {codon}";
    }

    private List<EventStep> CreateGenericEventChain() => new()
    {
        new() { Step = 1, Cue = "Seen", Goal = "Observe the frequency pattern" },
        new() { Step = 2, Cue = "Felt", Goal = "Attune to the resonance" },
        new() { Step = 3, Cue = "Heard", Goal = "Listen to the harmonic" },
        new() { Step = 4, Cue = "MatterCapture", Goal = "Capture the frequency signature" }
    };

    private List<EventChoice> CreateGenericChoices() => new()
    {
        new() { Id = "A", Label = "Embrace Frequency", Effects = new ChoiceEffects { Weights = new() { ["resonance"] = 0.05f } } },
        new() { Id = "B", Label = "Resist Pattern", Effects = new ChoiceEffects { Weights = new() { ["independence"] = 0.05f } } },
        new() { Id = "C", Label = "Transform Through", Effects = new ChoiceEffects { Weights = new() { ["evolution"] = 0.05f } } }
    };

    #endregion

    /// <summary>
    /// Get all available codons for an agent based on their evolution
    /// </summary>
    public List<CodonTemplate> GetAvailableCodeons(TamagotchiAgent agent)
    {
        var maxCodon = Math.Min(64, Math.Max(1, agent.Evolution / 2)); // Evolution 0-100 maps to codons 1-64
        return _codonSpectrum.Values.Where(c => c.CodonId <= maxCodon).ToList();
    }
}

#region Data Models

public class CodonTemplate
{
    public int CodonId { get; set; }
    public string Name { get; set; } = string.Empty;
    public string Archetype { get; set; } = string.Empty;
    public string Center { get; set; } = string.Empty;
    public SimsScenario SimsScenario { get; set; } = new();
    public List<EventStep> EventChain { get; set; } = new();
    public List<EventChoice> Choices { get; set; } = new();
    public AutonomyShift AutonomyShift { get; set; } = new();
    public MatterCapture MatterCapture { get; set; } = new();
}

public class CodonEventResult
{
    public string CodonId { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public string Archetype { get; set; } = string.Empty;
    public string Center { get; set; } = string.Empty;
    public int Tier { get; set; }
    public SimsScenario SimsScenario { get; set; } = new();
    public List<EventStep> EventChain { get; set; } = new();
    public List<EventChoice> Choices { get; set; } = new();
    public AutonomyShift AutonomyShift { get; set; } = new();
    public MatterCapture MatterCapture { get; set; } = new();
    
    // Event State
    public int CurrentStep { get; set; }
    public List<string> CompletedSteps { get; set; } = new();
    public string AgentId { get; set; } = string.Empty;
    public DateTime ActivatedAt { get; set; }
}


public class SimsScenario
{
    public string Premise { get; set; } = string.Empty;
    public List<string> SetDressing { get; set; } = new();
    public List<string> Triggers { get; set; } = new();
}

public class EventStep
{
    public int Step { get; set; }
    public string Cue { get; set; } = string.Empty; // Seen, Felt, Heard, MatterCapture
    public string Goal { get; set; } = string.Empty;
}

public class EventChoice
{
    public string Id { get; set; } = string.Empty;
    public string Label { get; set; } = string.Empty;
    public ChoiceEffects Effects { get; set; } = new();
}

public class ChoiceEffects
{
    public Dictionary<string, int> Motives { get; set; } = new(); // Safety, Clarity, Vitality, Expression, Connection
    public Dictionary<string, float> Weights { get; set; } = new(); // Autonomy behavior weights
}

public class AutonomyShift
{
    public float AskThreshold { get; set; } // When agent asks for guidance
    public float AutoThreshold { get; set; } // When agent acts autonomously
    public float SignatureThreshold { get; set; } // When behavior becomes signature/permanent
}

public class MatterCapture
{
    public string Artifact { get; set; } = string.Empty;
    public List<string> Passive { get; set; } = new(); // Always-on effects
    public List<string> Active { get; set; } = new(); // Cooldown abilities
    public string Visual { get; set; } = string.Empty; // Visual effects/sigil
}

public class AlienExaggeration
{
    public string Name { get; set; } = string.Empty;
    public List<string> AdditionalProps { get; set; } = new();
    public int Intensity { get; set; } = 1;
}

#endregion