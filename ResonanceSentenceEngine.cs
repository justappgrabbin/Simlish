using System.Text.Json;

namespace ModularSimWorld.Components;

/// <summary>
/// Resonance Sentence Engine - Ra Uru Hu's complete calculation framework
/// Builds sentences using Five Centers keyword banks and full astrological lens
/// Base→Tone→Center Keywords→Gate→Line→Color→Degree→Axis→Zodiac→House
/// </summary>
public class ResonanceSentenceEngine
{
    private readonly Dictionary<string, CenterDimension> _fiveCenters;
    private readonly Dictionary<int, ToneMapping> _toneSystem;
    private readonly Dictionary<int, ColorMapping> _colorSystem;
    private readonly Dictionary<string, string[]> _zodiacKeywords;
    private readonly Dictionary<int, string> _houseContexts;
    private readonly Dictionary<char, SymbolicOperator> _operators;
    private readonly Random _random;

    public ResonanceSentenceEngine()
    {
        _random = new Random();
        _fiveCenters = InitializeFiveCenters();
        _toneSystem = InitializeToneSystem();
        _colorSystem = InitializeColorSystem();
        _zodiacKeywords = InitializeZodiacKeywords();
        _houseContexts = InitializeHouseContexts();
        _operators = InitializeSymbolicOperators();
        Console.WriteLine("[Resonance] Ra's sentence structure system initialized with Five Centers");
    }

    /// <summary>
    /// Generates resonance sentence from 69,120 total combinations
    /// 5 bases × 6 tones × 6 colors × 6 lines × 64 gates = 69,120 possibilities
    /// Mix and match and repeat!
    /// </summary>
    public ResonanceSentence GenerateResonanceSentence(TamagotchiAgent agent, int? specificGate = null)
    {
        // Mix and match all components - 69,120 total combinations!
        var baseDimension = SelectBaseDimension(agent);
        var tone = SelectTone(agent);
        var line = SelectLine(agent);
        var color = SelectColor(agent);
        var gateNumber = specificGate ?? SelectGate(agent);
        var axis = SelectAxis(agent);
        var zodiac = SelectZodiac(agent);
        var house = SelectHouse(agent);
        
        // Calculate unique lattice position from combination
        var latticePosition = CalculateLatticePosition(baseDimension.Id, tone.Id, color.Id, line, gateNumber);
        var degree = CalculateDegree(gateNumber, line, color, tone, baseDimension);
        var minute = CalculateMinute(agent);
        var second = CalculateSecond(agent);
        
        // Build the sentence
        var sentence = new ResonanceSentence
        {
            // Core components
            Base = baseDimension.Name,
            BaseVoice = baseDimension.Voice,
            Tone = tone.Nature,
            CenterKeywords = baseDimension.Keywords.Take(3).ToList(),
            Gate = gateNumber,
            Line = line,
            Color = color.Motivation,
            
            // Calculated components
            Degree = degree,
            Minute = minute,
            Second = second,
            Axis = axis,
            Zodiac = zodiac,
            House = house,
            
            // Astrological lens notation
            AstrologicalLens = $"{gateNumber}.{line}.{color.Id}.{tone.Id}.{baseDimension.Id} {degree} {minute}'{second}.{axis}\" {zodiac} {house}",
            
            // Generated sentence with lattice position
            LatticePosition = latticePosition,
            CompleteSentence = BuildCompleteSentence(baseDimension, tone, gateNumber, line, color, degree, minute, second, axis, zodiac, house),
            
            // Metadata
            AgentId = agent.Id,
            GeneratedAt = DateTime.Now
        };

        Console.WriteLine($"[Resonance] Generated sentence for agent {agent.Name}: {sentence.BaseVoice} through {sentence.Tone}");
        
        return sentence;
    }

    #region Five Centers Initialization

    private Dictionary<string, CenterDimension> InitializeFiveCenters()
    {
        return new Dictionary<string, CenterDimension>
        {
            ["Movement"] = new CenterDimension
            {
                Id = 1,
                Name = "Movement",
                Voice = "I Define",
                Domain = "Activity, uniqueness, orientation",
                Keywords = new[] { "Movement", "Energy", "Creation", "Seeing", "Landscape", "Environment" },
                Macrocosmic = "Movement is Energy → Energy is Creation → Creation is Seeing → Seeing is Landscape → Landscape is Environment",
                Microcosmic = "Individuality → Activity → Uniqueness → Reaction → Limitation → Perspective → Relation"
            },
            
            ["Evolution"] = new CenterDimension
            {
                Id = 2,
                Name = "Evolution", 
                Voice = "I Remember",
                Domain = "Character, integration, transgenerational consciousness",
                Keywords = new[] { "Evolution", "Gravity", "Memory", "Taste", "Love", "Light" },
                Macrocosmic = "Evolution is Gravity → Gravity is Memory → Memory is Taste → Taste is Love → Love is Light",
                Microcosmic = "The Mind → Character → Role → Separation → Nature → Integration → Spirit"
            },
            
            ["Being"] = new CenterDimension
            {
                Id = 3,
                Name = "Being",
                Voice = "I Am", 
                Domain = "Biology, chemistry, embodiment, genetics",
                Keywords = new[] { "Being", "Matter", "Touch", "Sex", "Survival" },
                Macrocosmic = "Being is Matter → Matter is Touch → Touch is Sex → Sex is Survival",
                Microcosmic = "The Body → Biology → Genetics → Chemistry → Objectivity → Geometry → Trajectory"
            },
            
            ["Design"] = new CenterDimension
            {
                Id = 4,
                Name = "Design",
                Voice = "I Design",
                Domain = "Growth, continuity, decay, manifestation", 
                Keywords = new[] { "Design", "Structure", "Progress", "Smell", "Life", "Art" },
                Macrocosmic = "Design is Structure → Structure is Progress → Progress is Smell → Smelling is Life → Life is Art",
                Microcosmic = "The Ego → Homo Sapiens (Self) → Growth → Decay → Continuity → Manifestation"
            },
            
            ["Space"] = new CenterDimension
            {
                Id = 5,
                Name = "Space",
                Voice = "I Think",
                Domain = "Fantasy, rhythm, subjectivity, timing",
                Keywords = new[] { "Space", "Form", "Illusion", "Hearing", "Music", "Freedom" },
                Macrocosmic = "Space is Form → Form is Illusion → Illusion is Hearing → Hearing is Music → Music is Freedom", 
                Microcosmic = "Personality → Type → Presence → Fantasy → Subjectivity → Rhythm → Timing"
            }
        };
    }

    private Dictionary<int, ToneMapping> InitializeToneSystem()
    {
        return new Dictionary<int, ToneMapping>
        {
            [1] = new ToneMapping { Id = 1, Nature = "Security", Department = "Smell", Center = "Splenic" },
            [2] = new ToneMapping { Id = 2, Nature = "Uncertainty", Department = "Taste", Center = "Splenic" },
            [3] = new ToneMapping { Id = 3, Nature = "Action", Department = "Outer Vision", Center = "Ajna" },
            [4] = new ToneMapping { Id = 4, Nature = "Meditation", Department = "Inner Vision", Center = "Ajna" },
            [5] = new ToneMapping { Id = 5, Nature = "Judgement", Department = "Feeling", Center = "Solar" },
            [6] = new ToneMapping { Id = 6, Nature = "Acceptance", Department = "Touch", Center = "Plexus" }
        };
    }

    private Dictionary<int, ColorMapping> InitializeColorSystem()
    {
        return new Dictionary<int, ColorMapping>
        {
            [1] = new ColorMapping { Id = 1, Motivation = "Fear", Mode = "Communalist vs. Separatist", Center = "Splenic" },
            [2] = new ColorMapping { Id = 2, Motivation = "Hope", Mode = "Theist vs. Anti-theist", Center = "Splenic" },
            [3] = new ColorMapping { Id = 3, Motivation = "Desire", Mode = "Leader vs. Follower", Center = "Ajna" },
            [4] = new ColorMapping { Id = 4, Motivation = "Need", Mode = "Master vs. Novice", Center = "Solar" },
            [5] = new ColorMapping { Id = 5, Motivation = "Guilt", Mode = "Conditioner vs. Conditioned", Center = "Solar" },
            [6] = new ColorMapping { Id = 6, Motivation = "Innocence", Mode = "Observer vs. Observed", Center = "Plexus" }
        };
    }

    private Dictionary<string, string[]> InitializeZodiacKeywords()
    {
        return new Dictionary<string, string[]>
        {
            ["Aries"] = new[] { "Initiative", "Beginning", "Fire", "Cardinal" },
            ["Taurus"] = new[] { "Stability", "Form", "Earth", "Fixed" },
            ["Gemini"] = new[] { "Communication", "Duality", "Air", "Mutable" },
            ["Cancer"] = new[] { "Nurturing", "Security", "Water", "Cardinal" },
            ["Leo"] = new[] { "Expression", "Creativity", "Fire", "Fixed" },
            ["Virgo"] = new[] { "Service", "Analysis", "Earth", "Mutable" },
            ["Libra"] = new[] { "Balance", "Harmony", "Air", "Cardinal" },
            ["Scorpio"] = new[] { "Transformation", "Depth", "Water", "Fixed" },
            ["Sagittarius"] = new[] { "Expansion", "Philosophy", "Fire", "Mutable" },
            ["Capricorn"] = new[] { "Structure", "Authority", "Earth", "Cardinal" },
            ["Aquarius"] = new[] { "Innovation", "Disruption", "Air", "Fixed" },
            ["Pisces"] = new[] { "Dissolution", "Compassion", "Water", "Mutable" }
        };
    }

    private Dictionary<int, string> InitializeHouseContexts()
    {
        return new Dictionary<int, string>
        {
            [1] = "Identity and Self-Expression",
            [2] = "Resources and Values",
            [3] = "Communication and Learning",
            [4] = "Home and Foundation", 
            [5] = "Creativity and Children",
            [6] = "Health and Service",
            [7] = "Relationships and Others",
            [8] = "Transformation and Shared Resources",
            [9] = "Philosophy and Higher Learning",
            [10] = "Career and Public Image",
            [11] = "Community and Ideals",
            [12] = "Spirituality and Transcendence"
        };
    }

    private Dictionary<char, SymbolicOperator> InitializeSymbolicOperators()
    {
        return new Dictionary<char, SymbolicOperator>
        {
            ['•'] = new SymbolicOperator { Symbol = '•', Name = "Singularity", Function = "Pre-collapse seed, infinite potential" },
            ['.'] = new SymbolicOperator { Symbol = '.', Name = "Transitioner", Function = "Step inward, descend to next chamber" },
            ['°'] = new SymbolicOperator { Symbol = '°', Name = "Collapse", Function = "Anchor into coordinate (phase/degree)" },
            [':'] = new SymbolicOperator { Symbol = ':', Name = "Portal", Function = "Threshold, parallel chamber" },
            [';'] = new SymbolicOperator { Symbol = ';', Name = "Fork", Function = "Branch, divergent streams" },
            [','] = new SymbolicOperator { Symbol = ',', Name = "Breath", Function = "Pause, collect fragments" },
            ['–'] = new SymbolicOperator { Symbol = '–', Name = "Current", Function = "Span, wave flow" },
            ['′'] = new SymbolicOperator { Symbol = '′', Name = "Pulse", Function = "Arcminute, heartbeat tick" },
            ['″'] = new SymbolicOperator { Symbol = '″', Name = "Flicker", Function = "Arcsecond, micro-shimmer" },
            ['/'] = new SymbolicOperator { Symbol = '/', Name = "Blade", Function = "Cut, divide, choice" },
            ['\\'] = new SymbolicOperator { Symbol = '\\', Name = "Escape", Function = "Sideways exit" },
            ['*'] = new SymbolicOperator { Symbol = '*', Name = "Starburst", Function = "Expansion, multiplication" },
            ['='] = new SymbolicOperator { Symbol = '=', Name = "Mirror", Function = "Collapse of two into one" },
            ['→'] = new SymbolicOperator { Symbol = '→', Name = "Vector", Function = "Direction, energy flow" }
        };
    }

    #endregion

    #region Selection Methods

    private CenterDimension SelectBaseDimension(TamagotchiAgent agent)
    {
        // Select based on agent's dominant elemental nature
        var baseName = agent.DNA.PrimaryElement.ToLower() switch
        {
            "fire" => "Movement",   // I Define
            "air" => "Space",       // I Think  
            "earth" => "Design",    // I Design
            "water" => "Evolution", // I Remember
            _ => "Being"            // I Am (default)
        };
        
        return _fiveCenters[baseName];
    }

    private ToneMapping SelectTone(TamagotchiAgent agent)
    {
        // Mix and match: any of 6 tones can combine with any base
        var toneId = _random.Next(1, 7);  // 1-6 tones
        return _toneSystem[toneId];
    }

    private int SelectLine(TamagotchiAgent agent)
    {
        // Mix and match: any of 6 lines
        return _random.Next(1, 7);  // 1-6 lines
    }

    private ColorMapping SelectColor(TamagotchiAgent agent)
    {
        // Mix and match: any of 6 colors
        var colorId = _random.Next(1, 7);  // 1-6 colors
        return _colorSystem[colorId];
    }
    
    private int SelectGate(TamagotchiAgent agent)
    {
        // Mix and match: any of 64 gates
        return _random.Next(1, 65);  // 1-64 gates
    }

    private string SelectAxis(TamagotchiAgent agent)
    {
        // Vertical/Horizontal axis based on agent's orientation
        return agent.Resonance > 50 ? "Vertical" : "Horizontal";
    }

    private string SelectZodiac(TamagotchiAgent agent)
    {
        // Based on agent's astrological character if available
        if (agent.AstroCharacter?.BodyTraits?.SunSign != null)
        {
            return agent.AstroCharacter.BodyTraits.SunSign.ToString();
        }
        
        // Default to calculation based on agent characteristics
        var zodiacNames = _zodiacKeywords.Keys.ToArray();
        return zodiacNames[agent.Id.GetHashCode() % zodiacNames.Length];
    }

    private int SelectHouse(TamagotchiAgent agent)
    {
        // House 1-12 based on agent's current life focus area
        return ((agent.Evolution / 10) % 12) + 1;
    }

    #endregion

    #region Calculation Methods

    private int CalculateDegree(int gate, int line, ColorMapping color, ToneMapping tone, CenterDimension baseDimension)
    {
        // Compound expression: Gate + Line + Color + Tone + Base
        return ((gate + line + color.Id + tone.Id + baseDimension.Id) % 360);
    }

    private int CalculateMinute(TamagotchiAgent agent)
    {
        // Arcminute refinement (0-59)
        return (agent.Age + agent.Evolution) % 60;
    }

    private int CalculateSecond(TamagotchiAgent agent)
    {
        // Arcsecond micro-variation (0-59)
        return (agent.Resonance + agent.Happiness) % 60;
    }
    
    private int CalculateLatticePosition(int baseId, int toneId, int colorId, int line, int gate)
    {
        // Calculate unique position in 69,120 lattice
        // Formula: (base-1)*6*6*6*64 + (tone-1)*6*6*64 + (color-1)*6*64 + (line-1)*64 + (gate-1)
        var position = (baseId - 1) * 13824 +  // 6*6*6*64 = 13,824
                      (toneId - 1) * 2304 +   // 6*6*64 = 2,304  
                      (colorId - 1) * 384 +   // 6*64 = 384
                      (line - 1) * 64 +       // 64
                      (gate - 1);             // Individual gate
        
        return position + 1; // 1-indexed
    }

    private string BuildCompleteSentence(CenterDimension baseDimension, ToneMapping tone, int gate, int line, 
        ColorMapping color, int degree, int minute, int second, string axis, string zodiac, int house)
    {
        var centerKeywords = string.Join(", ", baseDimension.Keywords.Take(3));
        var zodiacFlavor = string.Join(" and ", _zodiacKeywords[zodiac].Take(2));
        var houseContext = _houseContexts[house];
        
        return $"At the Base of {baseDimension.Name}, expressed through the Tone of {tone.Nature}, " +
               $"with keywords {centerKeywords}, moving through Gate {gate} in its {GetLineRole(line)} role, " +
               $"responding with {color.Motivation} motivation, crystallized in Degree {degree}, " +
               $"refined through {minute}'{second} fractal nuance, aligned to the {axis} Axis, " +
               $"flavored by {zodiac}'s {zodiacFlavor.ToLower()}, " +
               $"and contextualized in the {house} House of {houseContext.ToLower()}.";
    }

    private string GetLineRole(int line) => line switch
    {
        1 => "foundation",
        2 => "natural", 
        3 => "experimental",
        4 => "opportunistic",
        5 => "projective", 
        6 => "transcendent",
        _ => "dynamic"
    };

    #endregion

    /// <summary>
    /// Apply symbolic operators to create resonance variations
    /// </summary>
    public string ApplySymbolicOperators(ResonanceSentence sentence, string operatorSequence)
    {
        var modified = sentence.CompleteSentence;
        
        foreach (var op in operatorSequence)
        {
            if (_operators.ContainsKey(op))
            {
                var operation = _operators[op];
                // Apply transformation based on operator function
                modified = operation.Function switch
                {
                    var f when f.Contains("expansion") => modified + " *expanding outward*",
                    var f when f.Contains("pause") => modified.Replace(",", ", *pause*,"),
                    var f when f.Contains("portal") => "*portal opens* " + modified,
                    var f when f.Contains("choice") => modified + " /at the crossroads/",
                    _ => modified
                };
            }
        }
        
        return modified;
    }
}

#region Data Models

public class ResonanceSentence
{
    public string Base { get; set; } = string.Empty;
    public string BaseVoice { get; set; } = string.Empty;  // I Define, I Remember, etc.
    public string Tone { get; set; } = string.Empty;
    public List<string> CenterKeywords { get; set; } = new();
    public int Gate { get; set; }
    public int Line { get; set; }
    public string Color { get; set; } = string.Empty;
    
    public int Degree { get; set; }
    public int Minute { get; set; }
    public int Second { get; set; }
    public string Axis { get; set; } = string.Empty;
    public string Zodiac { get; set; } = string.Empty;
    public int House { get; set; }
    
    public string AstrologicalLens { get; set; } = string.Empty; // gate.line.color.tone.base degree minute'second.axis" zodiac house
    public string CompleteSentence { get; set; } = string.Empty;
    
    public string AgentId { get; set; } = string.Empty;
    public string LatticePosition { get; set; } = string.Empty;
    public DateTime GeneratedAt { get; set; }
}

public class CenterDimension
{
    public int Id { get; set; }
    public string Name { get; set; } = string.Empty;
    public string Voice { get; set; } = string.Empty;
    public string Domain { get; set; } = string.Empty;
    public string[] Keywords { get; set; } = Array.Empty<string>();
    public string Macrocosmic { get; set; } = string.Empty;
    public string Microcosmic { get; set; } = string.Empty;
}

public class ToneMapping
{
    public int Id { get; set; }
    public string Nature { get; set; } = string.Empty;
    public string Department { get; set; } = string.Empty;  // Sense/awareness
    public string Center { get; set; } = string.Empty;      // Body awareness center
}

public class ColorMapping
{
    public int Id { get; set; }
    public string Motivation { get; set; } = string.Empty;
    public string Mode { get; set; } = string.Empty;        // Binary modes
    public string Center { get; set; } = string.Empty;      // Body awareness center
}

public class SymbolicOperator
{
    public char Symbol { get; set; }
    public string Name { get; set; } = string.Empty;
    public string Function { get; set; } = string.Empty;
}

#endregion