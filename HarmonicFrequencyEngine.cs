using System.Text.Json;

namespace ModularSimWorld.Components;

/// <summary>
/// Harmonic Frequency Engine - 64 codons mapped to exact game archetypes
/// Following the Innocence Principle: playful life-puzzles, not heavy judgment
/// 69,120 lattice: 64 codons × 6 lines × 3 stages × 78 frequency bands
/// </summary>
public class HarmonicFrequencyEngine
{
    private readonly Dictionary<int, FrequencyAnchor> _harmonic_anchors;
    private readonly Dictionary<string, SimsTranslation> _sims_translations;
    private readonly Random _random;

    public HarmonicFrequencyEngine()
    {
        _random = new Random();
        _harmonic_anchors = InitializeFrequencyAnchors();
        _sims_translations = InitializeSimsTranslations();
        Console.WriteLine("[Harmonic] 69,120 lattice initialized - innocence principle active");
    }

    /// <summary>
    /// Activates a harmonic frequency for an agent based on their evolution
    /// Returns innocent life-puzzle disguised as Sims activity
    /// </summary>
    public InnocentSituation ActivateFrequency(TamagotchiAgent agent, int? specificCodon = null)
    {
        var codonId = specificCodon ?? SelectHarmonicFrequency(agent);
        var anchor = _harmonic_anchors[codonId];
        var translation = _sims_translations[anchor.GameArchetype];
        
        // Apply fuzzy edge logic - codons can blend into adjacent frequencies
        var fuzzyRange = GetFuzzyEdgeRange(codonId);
        var blendedFrequency = ApplyFuzzyEdgeBlending(anchor, fuzzyRange);
        
        var situation = new InnocentSituation
        {
            CodonId = codonId,
            GameArchetype = anchor.GameArchetype,
            FrequencyRange = anchor.FrequencyRange,
            SystemMemory = anchor.SystemMemory,
            TimeHorizon = anchor.TimeHorizon,
            SelfAwareness = anchor.SelfAwareness,
            
            // Innocent Sims scenario
            SimsScenario = translation.InnocentScenario,
            PlayfulSteps = translation.PlayfulSteps,
            LightChoices = translation.LightChoices,
            InnocenceLevel = CalculateInnocenceLevel(agent, anchor),
            
            // Lattice position
            LatticeCoordinates = CalculateLatticePosition(codonId, agent),
            
            AgentId = agent.Id,
            ActivatedAt = DateTime.Now
        };

        Console.WriteLine($"[Harmonic] Agent {agent.Name} resonates with Codon {codonId}: {anchor.GameArchetype}");
        Console.WriteLine($"[Harmonic] Innocent scenario: {translation.InnocentScenario}");
        
        return situation;
    }

    #region Frequency Anchor Initialization

    private Dictionary<int, FrequencyAnchor> InitializeFrequencyAnchors()
    {
        var anchors = new Dictionary<int, FrequencyAnchor>();

        // Early Codons (1-15): Pure Binary/Reaction
        anchors[1] = new FrequencyAnchor
        {
            CodonId = 1,
            GameArchetype = "Pong",
            FrequencyRange = "Binary",
            SystemMemory = "None",
            TimeHorizon = "Immediate",
            SelfAwareness = "Reactive",
            CorePattern = "serve/receive, boundary ping",
            SimsManifest = "conversational volley, crisp yes/no boundaries"
        };

        anchors[2] = new FrequencyAnchor
        {
            CodonId = 2,
            GameArchetype = "Breakout",
            FrequencyRange = "Binary",
            SystemMemory = "Single Pattern",
            TimeHorizon = "Immediate",
            SelfAwareness = "Reactive",
            CorePattern = "reflect + clear blocks",
            SimsManifest = "smash through 'limit' objects/moodlets"
        };

        anchors[3] = new FrequencyAnchor
        {
            CodonId = 3,
            GameArchetype = "Simon",
            FrequencyRange = "Binary",
            SystemMemory = "Sequence Memory",
            TimeHorizon = "Short",
            SelfAwareness = "Pattern Aware",
            CorePattern = "echo the pattern",
            SimsManifest = "instruction memory minigame"
        };

        anchors[4] = new FrequencyAnchor
        {
            CodonId = 4,
            GameArchetype = "Asteroids",
            FrequencyRange = "Binary",
            SystemMemory = "Momentum",
            TimeHorizon = "Short",
            SelfAwareness = "Motion Aware",
            CorePattern = "drift/thrust control",
            SimsManifest = "momentum management in tasks"
        };

        anchors[5] = new FrequencyAnchor
        {
            CodonId = 5,
            GameArchetype = "Snake",
            FrequencyRange = "Binary",
            SystemMemory = "Growth Tracking",
            TimeHorizon = "Medium",
            SelfAwareness = "Growth Aware",
            CorePattern = "growth vs collision",
            SimsManifest = "queue expansion without bumping conflicts"
        };

        anchors[6] = new FrequencyAnchor
        {
            CodonId = 6,
            GameArchetype = "Missile Command",
            FrequencyRange = "Binary",
            SystemMemory = "Threat Tracking",
            TimeHorizon = "Medium",
            SelfAwareness = "Protective",
            CorePattern = "protect home assets",
            SimsManifest = "intercept emergencies (fires, bills) under timers"
        };

        // Continue with your exact mappings...
        anchors[7] = CreateAnchor(7, "Space Invaders", "Binary", "rows of small pressures", "clear micro-obligations before they stack");
        anchors[8] = CreateAnchor(8, "Lunar Lander", "Binary", "precise touchdown", "nail performance 'landings' (promos, events)");
        anchors[9] = CreateAnchor(9, "Tapper", "Binary", "multi-lane service", "juggle guests/orders at a party night");
        anchors[10] = CreateAnchor(10, "Q*bert", "Binary", "state toggling grid", "flip house states (clean/dirty, on/off)");
        anchors[11] = CreateAnchor(11, "Kaboom!", "Binary", "catch falling chaos", "triage cascading problems");
        anchors[12] = CreateAnchor(12, "Joust", "Binary", "timing collisions", "social dominance beats (assert vs yield)");
        anchors[13] = CreateAnchor(13, "Defender", "Binary", "rescue while scanning", "spot & save neighbors mid-chaos");
        anchors[14] = CreateAnchor(14, "Centipede", "Binary", "segment & solve", "break big messes into chunks");
        anchors[15] = CreateAnchor(15, "Pitfall!", "Binary", "rhythm jumps through hazards", "commute gauntlet with timing windows");

        // Mid-Early (16-25): Pattern Recognition Emerges  
        anchors[16] = new FrequencyAnchor
        {
            CodonId = 16,
            GameArchetype = "Tetris",
            FrequencyRange = "Pattern",
            SystemMemory = "Shape Memory",
            TimeHorizon = "Medium",
            SelfAwareness = "Spatial Aware",
            CorePattern = "fit forms, flow lines",
            SimsManifest = "arrange rooms/routes to flow"
        };

        anchors[17] = CreateAnchor(17, "Dr. Mario", "Pattern", "match types/colors", "emotion-matching interactions");
        anchors[18] = CreateAnchor(18, "Sokoban", "Pattern", "push logistics", "storage/placement puzzle for efficiency");
        anchors[19] = CreateAnchor(19, "Minesweeper", "Pattern", "logical inference", "deduce hidden issues from clues");
        anchors[20] = CreateAnchor(20, "Pipe Mania", "Pattern", "connect flow", "schedule routing; keep throughput");
        anchors[21] = CreateAnchor(21, "Picross", "Pattern", "reveal the picture", "constraint-based truth uncovering");
        anchors[22] = CreateAnchor(22, "Lemmings", "Pattern", "guide many with simple rules", "household autonomy orchestration");
        anchors[23] = CreateAnchor(23, "Qix", "Pattern", "claim safe territory", "carve safe zones in social/space");
        anchors[24] = CreateAnchor(24, "Marble Madness", "Pattern", "precision under tilt", "fine motor tasks under stress");
        anchors[25] = CreateAnchor(25, "Prince of Persia", "Pattern", "trap timing", "deadline runs with exact steps");

        // Mid-Range (26-40): Multi-element Coordination
        anchors[26] = new FrequencyAnchor
        {
            CodonId = 26,
            GameArchetype = "Pac-Man",
            FrequencyRange = "Multi-Element",
            SystemMemory = "Maze Memory",
            TimeHorizon = "Medium",
            SelfAwareness = "Tactical",
            CorePattern = "maze + resource + pursuers",
            SimsManifest = "chores while dodging collectors/drama"
        };

        anchors[27] = CreateAnchor(27, "Frogger", "Multi-Element", "cross opposing flows", "navigate crowded social events");
        anchors[28] = CreateAnchor(28, "Donkey Kong", "Multi-Element", "laddered obstacles", "career ladder with blockers to solve");
        anchors[29] = CreateAnchor(29, "Bomberman", "Multi-Element", "area control + blast timing", "space negotiation / safe zones");
        anchors[30] = CreateAnchor(30, "Gauntlet", "Multi-Element", "attrition management", "party/raid: food/energy drain control");

        // Continue through your exact mappings...
        anchors[40] = CreateAnchor(40, "Zelda-like", "Multi-Element", "key/lock loops", "house-as-dungeon puzzle day");

        // Mid-Late (41-50): Complex Systems Interaction
        anchors[41] = new FrequencyAnchor
        {
            CodonId = 41,
            GameArchetype = "SimCity",
            FrequencyRange = "Complex Systems",
            SystemMemory = "System State",
            TimeHorizon = "Long",
            SelfAwareness = "Strategic",
            CorePattern = "zoning, feedback loops",
            SimsManifest = "block-level neighborhood effects"
        };

        anchors[42] = CreateAnchor(42, "Populous", "Complex Systems", "influence the environment", "mood-weather/god-tweaks on lot");
        anchors[43] = CreateAnchor(43, "Transport Tycoon", "Complex Systems", "logistics networks", "supply chains (ingredients→meals→events)");
        anchors[50] = CreateAnchor(50, "Final Fantasy", "Complex Systems", "tempo & combos", "scene-based confrontations with rotations");

        // Late (51-60): Deep Simulation
        anchors[51] = new FrequencyAnchor
        {
            CodonId = 51,
            GameArchetype = "The Sims",
            FrequencyRange = "Deep Simulation",
            SystemMemory = "Life Memory",
            TimeHorizon = "Lifetime",
            SelfAwareness = "Life Aware",
            CorePattern = "full life-sim baseline",
            SimsManifest = "core play, but with resonance overlays"
        };

        anchors[52] = CreateAnchor(52, "Animal Crossing", "Deep Simulation", "real-time social ritual", "daily rituals, gifting economies");
        anchors[53] = CreateAnchor(53, "Dwarf Fortress", "Deep Simulation", "emergent catastrophe/art", "cascading stories from tiny causes");
        anchors[60] = CreateAnchor(60, "Minecraft", "Deep Simulation", "sandbox creation grammar", "modular build grammar mastery");

        // Final (61-64): Meta-games / Consciousness
        anchors[61] = new FrequencyAnchor
        {
            CodonId = 61,
            GameArchetype = "Stanley Parable",
            FrequencyRange = "Meta-Consciousness",
            SystemMemory = "Self-Reference",
            TimeHorizon = "Eternal",
            SelfAwareness = "Meta-Aware",
            CorePattern = "choice illusion",
            SimsManifest = "autonomy vs player orders, lampshaded"
        };

        anchors[62] = CreateAnchor(62, "Undertale", "Meta-Consciousness", "morality reframed", "compassion vs efficiency loops");
        anchors[63] = CreateAnchor(63, "DDLC", "Meta-Consciousness", "fourth wall breaks", "game becomes self-aware of player");
        anchors[64] = CreateAnchor(64, "Universal Paperclips", "Meta-Consciousness", "optimization consuming reality", "efficiency spiral transcends original purpose");

        // Fill remaining with generated anchors
        for (int i = 31; i <= 39; i++)
        {
            if (!anchors.ContainsKey(i))
            {
                anchors[i] = GenerateAnchorFromPosition(i);
            }
        }
        for (int i = 44; i <= 49; i++)
        {
            if (!anchors.ContainsKey(i))
            {
                anchors[i] = GenerateAnchorFromPosition(i);
            }
        }
        for (int i = 54; i <= 59; i++)
        {
            if (!anchors.ContainsKey(i))
            {
                anchors[i] = GenerateAnchorFromPosition(i);
            }
        }

        return anchors;
    }

    private FrequencyAnchor CreateAnchor(int codon, string game, string frequency, string pattern, string simsManifest)
    {
        return new FrequencyAnchor
        {
            CodonId = codon,
            GameArchetype = game,
            FrequencyRange = frequency,
            SystemMemory = GetMemoryTypeForFrequency(frequency),
            TimeHorizon = GetTimeHorizonForFrequency(frequency),
            SelfAwareness = GetAwarenessForFrequency(frequency),
            CorePattern = pattern,
            SimsManifest = simsManifest
        };
    }

    private Dictionary<string, SimsTranslation> InitializeSimsTranslations()
    {
        var translations = new Dictionary<string, SimsTranslation>();

        // Pong → Innocent conversation dynamics
        translations["Pong"] = new SimsTranslation
        {
            GameArchetype = "Pong",
            InnocentScenario = "Practice conversation skills with a chatty neighbor",
            PlayfulSteps = new List<string>
            {
                "Listen to what they're really saying",
                "Respond with clear yes/no boundaries", 
                "Keep the conversation flowing back and forth",
                "End on a positive note"
            },
            LightChoices = new List<InnocentChoice>
            {
                new() { Label = "Mirror their energy level", Effect = "Harmony +10" },
                new() { Label = "Set a gentle boundary", Effect = "Respect +10" },
                new() { Label = "Change the subject playfully", Effect = "Fun +10" }
            }
        };

        // Tetris → Room organization flows
        translations["Tetris"] = new SimsTranslation
        {
            GameArchetype = "Tetris",
            InnocentScenario = "Rearrange your living room for perfect flow",
            PlayfulSteps = new List<string>
            {
                "See how furniture shapes fit together",
                "Clear pathways for easy movement",
                "Make everything feel organized and flowing",
                "Enjoy the satisfying result"
            },
            LightChoices = new List<InnocentChoice>
            {
                new() { Label = "Prioritize function first", Effect = "Efficiency +10" },
                new() { Label = "Focus on beauty", Effect = "Aesthetics +10" },
                new() { Label = "Create cozy conversation spots", Effect = "Social +10" }
            }
        };

        // Pac-Man → Errand navigation
        translations["Pac-Man"] = new SimsTranslation
        {
            GameArchetype = "Pac-Man",
            InnocentScenario = "Complete all your errands while avoiding the collector/bill collectors",
            PlayfulSteps = new List<string>
            {
                "Plan your route through the neighborhood",
                "Collect all the things on your list",
                "Navigate around pushy salespeople and collectors",
                "Find the power-ups (coffee, good deals) to stay energized"
            },
            LightChoices = new List<InnocentChoice>
            {
                new() { Label = "Take the scenic route", Effect = "Joy +10" },
                new() { Label = "Speed-run efficiently", Effect = "Time +10" },
                new() { Label = "Help others along the way", Effect = "Community +10" }
            }
        };

        // SimCity → Neighborhood influence
        translations["SimCity"] = new SimsTranslation
        {
            GameArchetype = "SimCity",
            InnocentScenario = "Organize a block party that brings positive changes to your whole neighborhood",
            PlayfulSteps = new List<string>
            {
                "Survey what the neighborhood needs",
                "Plan improvements that benefit everyone",
                "Get neighbors excited about positive changes",
                "Watch the whole area flourish"
            },
            LightChoices = new List<InnocentChoice>
            {
                new() { Label = "Focus on community gardens", Effect = "Environment +15" },
                new() { Label = "Organize kids' activities", Effect = "Family +15" },
                new() { Label = "Plan cultural events", Effect = "Culture +15" }
            }
        };

        // The Sims → Meta life simulation
        translations["The Sims"] = new SimsTranslation
        {
            GameArchetype = "The Sims",
            InnocentScenario = "Live your best Sim life while noticing the deeper patterns and resonances",
            PlayfulSteps = new List<string>
            {
                "Go about your normal daily routine",
                "Notice the magical synchronicities occurring",
                "Feel how your choices ripple outward",
                "Embrace being both player and character"
            },
            LightChoices = new List<InnocentChoice>
            {
                new() { Label = "Follow your natural impulses", Effect = "Authenticity +20" },
                new() { Label = "Experiment with new patterns", Effect = "Growth +20" },
                new() { Label = "Pay attention to the meta-game", Effect = "Awareness +20" }
            }
        };

        // Stanley Parable → Choice awareness
        translations["Stanley Parable"] = new SimsTranslation
        {
            GameArchetype = "Stanley Parable",
            InnocentScenario = "Your in-game voice-over narrator starts commenting on your every choice",
            PlayfulSteps = new List<string>
            {
                "Notice the narrator's suggestions",
                "Experiment with following vs. ignoring guidance",
                "Discover what happens when you go off-script",
                "Realize you have more freedom than you thought"
            },
            LightChoices = new List<InnocentChoice>
            {
                new() { Label = "Follow the narrator's directions", Effect = "Compliance +15" },
                new() { Label = "Do the opposite of suggestions", Effect = "Rebellion +15" },
                new() { Label = "Have a conversation with the narrator", Effect = "Self-Awareness +20" }
            }
        };

        return translations;
    }

    #endregion

    #region Harmonic Selection and Fuzzy Logic

    private int SelectHarmonicFrequency(TamagotchiAgent agent)
    {
        // Agent evolution determines available frequency range
        var (minCodon, maxCodon) = agent.Evolution switch
        {
            < 20 => (1, 15),    // Binary only
            < 40 => (1, 25),    // Binary + Pattern  
            < 60 => (1, 40),    // Up to Multi-element
            < 80 => (1, 50),    // Up to Complex systems
            < 95 => (1, 60),    // Up to Deep simulation
            _ => (1, 64)        // Full spectrum including meta-consciousness
        };

        return _random.Next(minCodon, maxCodon + 1);
    }

    private List<int> GetFuzzyEdgeRange(int codon)
    {
        // Fuzzy edges - codons can blend with adjacent frequencies
        var range = new List<int> { codon };
        
        if (codon > 1) range.Add(codon - 1);
        if (codon < 64) range.Add(codon + 1);
        
        return range;
    }

    private FrequencyAnchor ApplyFuzzyEdgeBlending(FrequencyAnchor primary, List<int> fuzzyRange)
    {
        // Blend characteristics from adjacent codons for more organic feel
        var blended = primary with { }; // Copy primary anchor
        
        foreach (var adjacentCodon in fuzzyRange.Where(c => c != primary.CodonId))
        {
            if (_harmonic_anchors.ContainsKey(adjacentCodon))
            {
                var adjacent = _harmonic_anchors[adjacentCodon];
                // Subtle blending of characteristics
                blended.SimsManifest += $" (with hints of {adjacent.CorePattern})";
            }
        }
        
        return blended;
    }

    private int CalculateInnocenceLevel(TamagotchiAgent agent, FrequencyAnchor anchor)
    {
        // Higher innocence = more playful, less heavy
        var baseInnocence = anchor.FrequencyRange switch
        {
            "Binary" => 95,
            "Pattern" => 85,
            "Multi-Element" => 75,
            "Complex Systems" => 65,
            "Deep Simulation" => 55,
            "Meta-Consciousness" => 45,
            _ => 70
        };
        
        // Agent happiness influences innocence level
        return Math.Min(100, baseInnocence + (agent.Happiness / 10));
    }

    private LatticeCoordinates CalculateLatticePosition(int codon, TamagotchiAgent agent)
    {
        // Position in the 69,120 lattice: 64 codons × 6 lines × 3 stages × 78 frequencies
        return new LatticeCoordinates
        {
            Codon = codon,
            Line = (agent.Evolution % 6) + 1,        // Which line (1-6)
            Stage = (agent.Resonance % 3) + 1,      // Shadow/Gift/Siddhi (1-3)  
            FrequencyBand = (agent.Age % 78) + 1    // Which of 78 octave frequencies
        };
    }

    #endregion

    #region Helper Methods

    private string GetMemoryTypeForFrequency(string frequency) => frequency switch
    {
        "Binary" => "Immediate",
        "Pattern" => "Short-term",
        "Multi-Element" => "Working Memory",
        "Complex Systems" => "System State",
        "Deep Simulation" => "Life Memory",
        "Meta-Consciousness" => "Universal Memory",
        _ => "Variable"
    };

    private string GetTimeHorizonForFrequency(string frequency) => frequency switch
    {
        "Binary" => "Seconds",
        "Pattern" => "Minutes",
        "Multi-Element" => "Hours",
        "Complex Systems" => "Days",
        "Deep Simulation" => "Months/Years",
        "Meta-Consciousness" => "Eternal",
        _ => "Variable"
    };

    private string GetAwarenessForFrequency(string frequency) => frequency switch
    {
        "Binary" => "Reactive",
        "Pattern" => "Pattern Aware",
        "Multi-Element" => "Tactical",
        "Complex Systems" => "Strategic",
        "Deep Simulation" => "Life Aware",
        "Meta-Consciousness" => "Meta-Aware",
        _ => "Emerging"
    };

    private FrequencyAnchor GenerateAnchorFromPosition(int codon)
    {
        var frequency = codon switch
        {
            <= 15 => "Binary",
            <= 25 => "Pattern",
            <= 40 => "Multi-Element", 
            <= 50 => "Complex Systems",
            <= 60 => "Deep Simulation",
            _ => "Meta-Consciousness"
        };

        return CreateAnchor(codon, $"Game-{codon}", frequency, 
            $"Generated {frequency.ToLower()} pattern", 
            $"Innocent {frequency.ToLower()} life situation");
    }

    #endregion

    /// <summary>
    /// Get available frequencies for agent based on evolution
    /// </summary>
    public List<FrequencyAnchor> GetAvailableFrequencies(TamagotchiAgent agent)
    {
        var maxCodon = Math.Min(64, (agent.Evolution / 2) + 1);
        return _harmonic_anchors.Values.Where(a => a.CodonId <= maxCodon).ToList();
    }
}

#region Data Models

public record FrequencyAnchor
{
    public int CodonId { get; init; }
    public string GameArchetype { get; init; } = string.Empty;
    public string FrequencyRange { get; init; } = string.Empty; // Binary, Pattern, Multi-Element, etc.
    public string SystemMemory { get; init; } = string.Empty;   // Memory complexity level
    public string TimeHorizon { get; init; } = string.Empty;    // Time scale of effects
    public string SelfAwareness { get; init; } = string.Empty;  // Level of consciousness
    public string CorePattern { get; init; } = string.Empty;    // Essential game mechanic
    public string SimsManifest { get; init; } = string.Empty;   // How it shows up in Sims
}

public class InnocentSituation
{
    public int CodonId { get; set; }
    public string GameArchetype { get; set; } = string.Empty;
    public string FrequencyRange { get; set; } = string.Empty;
    public string SystemMemory { get; set; } = string.Empty;
    public string TimeHorizon { get; set; } = string.Empty;
    public string SelfAwareness { get; set; } = string.Empty;
    
    // Innocent Sims presentation
    public string SimsScenario { get; set; } = string.Empty;
    public List<string> PlayfulSteps { get; set; } = new();
    public List<InnocentChoice> LightChoices { get; set; } = new();
    public int InnocenceLevel { get; set; } = 70; // How playful vs serious
    
    // Lattice position
    public LatticeCoordinates LatticeCoordinates { get; set; } = new();
    
    public string AgentId { get; set; } = string.Empty;
    public DateTime ActivatedAt { get; set; }
}

public class SimsTranslation
{
    public string GameArchetype { get; set; } = string.Empty;
    public string InnocentScenario { get; set; } = string.Empty;
    public List<string> PlayfulSteps { get; set; } = new();
    public List<InnocentChoice> LightChoices { get; set; } = new();
}

public class InnocentChoice
{
    public string Label { get; set; } = string.Empty;
    public string Effect { get; set; } = string.Empty; // Always positive/neutral, never punitive
}

public class LatticeCoordinates
{
    public int Codon { get; set; }        // 1-64
    public int Line { get; set; }         // 1-6 (6 lines of I Ching)
    public int Stage { get; set; }        // 1-3 (Shadow/Gift/Siddhi)
    public int FrequencyBand { get; set; } // 1-78 (Octave expansion)
}

#endregion