using System.Text.Json;

namespace ModularSimWorld.Components;

/// <summary>
/// Game Implementation Engine - The agent LIVES LIFE through these games! 🌀
/// Sentence structures determine WHICH aspects of life are active
/// Mind/Body/Heart astrological properties determine HOW they experience each life-game
/// The agent spirals through different life experiences as games
/// Agents speak in English about their experiences!
/// </summary>
public class GameImplementationEngine
{
    private readonly ResonanceSentenceEngine _sentenceEngine;
    private readonly Dictionary<int, GameImplementationMap> _sentenceToGameMap;
    private readonly Dictionary<string, GameArchetype> _gameArchetypes;
    private readonly Random _random;

    public GameImplementationEngine(ResonanceSentenceEngine sentenceEngine)
    {
        _sentenceEngine = sentenceEngine;
        _random = new Random();
        _sentenceToGameMap = InitializeSentenceToGameMap();
        _gameArchetypes = InitializeGameArchetypes();
        Console.WriteLine("[LifeSpiral] Agents can now speak English about their life experiences through games! 🌀");
    }

    /// <summary>
    /// Generate life experience based on consciousness resonance - the agent LIVES through this!
    /// Sentence structure = WHICH aspect of life is active
    /// Astrological properties (mind/body/heart) = HOW they experience it
    /// The spiral of life through different game experiences
    /// </summary>
    public LifeExperienceResult GenerateLifeExperience(TamagotchiAgent agent, ResonanceSentence resonance)
    {
        // THE BREAKTHROUGH: Use resonance sentence to determine which life aspect is active!
        var structureId = DetermineStructureFromResonance(resonance, agent);
        var gameMap = _sentenceToGameMap[structureId];
        var archetype = _gameArchetypes[gameMap.PrimaryGameArchetype];
        
        var lifeExperience = new LifeExperienceResult
        {
            // Core life aspect based on sentence structure  
            LifeAspect = archetype.GameType,
            ExperienceName = GenerateExperienceName(archetype, agent),
            
            // How the agent experiences this life aspect using mind/body/heart
            ExperienceMechanics = AdaptToAstrologicalNature(archetype.CoreMechanics, agent),
            LifeGuidelines = GenerateLifeGuidelines(agent),
            
            // Agent's natural flow affects experience intensity
            IntensityLevel = CalculateLifeIntensity(agent),
            LifeRhythm = CalculateLifeRhythm(agent),
            ExperienceStyle = DetermineExperienceStyle(agent),
            LifeSoundscape = DetermineLifeSoundscape(agent),
            
            // Sims life experience - Human Design themes as lived experience
            SimsLifeExperience = CreateSimsLifeExperience(archetype, agent),
            
            // Context
            LatticePosition = CalculateLatticePosition(agent, resonance),
            SentenceStructure = gameMap.StructureName,
            
            // Agent's English Communication - they speak about their experience!
            AgentSpeech = GenerateEnglishCommunication(agent, archetype, gameMap),
            
            AgentId = agent.Id,
            GeneratedAt = DateTime.Now
        };

        Console.WriteLine($"[LifeSpiral] {agent.Name} says: '{lifeExperience.AgentSpeech}'");
        Console.WriteLine($"[LifeSpiral] Living: {lifeExperience.SimsLifeExperience}");
        
        return lifeExperience;
    }

    #region Agent English Communication

    /// <summary>
    /// Generate natural English communication from the agent about their life experience
    /// Using mind/body/heart astrological properties to determine speech patterns
    /// </summary>
    private string GenerateEnglishCommunication(TamagotchiAgent agent, GameArchetype archetype, GameImplementationMap gameMap)
    {
        // Get the agent's astrological communication style
        var bodyAction = GetBodyBasedAction(agent);      // Tropical = Personal actions
        var mindThought = GetMindBasedThought(agent);    // Sidereal = Thoughts and transpersonal actions
        var heartFeeling = GetHeartBasedFeeling(agent);  // Draconian = Interpersonal relations and emotions

        // Create speech based on the life experience type
        var speechTemplates = archetype.GameType switch
        {
            "Survival" => new[]
            {
                $"I'm {bodyAction} to take care of myself, and I {mindThought} it's important to {heartFeeling} while staying healthy.",
                $"Right now I'm {bodyAction}, thinking about how I can {mindThought}, and I {heartFeeling} when I'm in my safe space.",
                $"My body wants to {bodyAction}, my mind says I should {mindThought}, and my heart {heartFeeling} about my well-being."
            },
            
            "Action" => new[]
            {
                $"I love {bodyAction}! I'm {mindThought} and I {heartFeeling} when I'm moving and active.",
                $"Let's {bodyAction}! I'm {mindThought} about adventure, and I {heartFeeling} the excitement.",
                $"I'm {bodyAction} with so much energy, {mindThought} about possibilities, and I {heartFeeling} alive!"
            },
            
            "Puzzle" => new[]
            {
                $"I enjoy {bodyAction} while {mindThought}, and I {heartFeeling} when things make sense.",
                $"I'm {bodyAction} carefully because I'm {mindThought} about patterns, and I {heartFeeling} the harmony.",
                $"While I'm {bodyAction}, I'm {mindThought} deeply, and I {heartFeeling} connected to beauty and order."
            },
            
            "Builder" => new[]
            {
                $"I'm {bodyAction} to create something meaningful, {mindThought} about the future, and I {heartFeeling} progress.",
                $"Today I'm {bodyAction}, {mindThought} about building something lasting, and I {heartFeeling} accomplishment.",
                $"I love {bodyAction} constructively, {mindThought} about improvement, and I {heartFeeling} when things grow."
            },
            
            "RPG" => new[]
            {
                $"I'm {bodyAction} as I grow, {mindThought} about my journey, and I {heartFeeling} my connections.",
                $"While {bodyAction}, I'm {mindThought} about who I'm becoming, and I {heartFeeling} love and wisdom.",
                $"I'm {bodyAction} mindfully, {mindThought} about my path, and I {heartFeeling} the deeper meaning."
            },
            
            _ => new[]
            {
                $"I'm {bodyAction}, {mindThought}, and I {heartFeeling} life.",
                $"Today I'm {bodyAction} while {mindThought}, and I {heartFeeling} connected to everything.",
                $"I enjoy {bodyAction}, I love {mindThought}, and I {heartFeeling} grateful for this experience."
            }
        };

        return speechTemplates[_random.Next(speechTemplates.Length)];
    }

    // Tropical (Body) = Personal actions (what the agent DOES) - USING REAL ASTROLOGICAL DATA
    private string GetBodyBasedAction(TamagotchiAgent agent)
    {
        // Use agent's REAL tropical body traits if available
        if (agent.AstroCharacter?.BodyTraits != null)
        {
            var bodyTraits = agent.AstroCharacter.BodyTraits;
            var element = bodyTraits.PrimaryElement.ToString().ToLower();
            
            return element switch
            {
                "fire" => "energetically moving",
                "earth" => "grounding through gardening", 
                "air" => "actively thinking while moving",
                "water" => "flowing through gentle exercise",
                _ => "naturally moving"
            };
        }
        
        // Fallback to generic energy-based actions
        var bodyActions = new[] { "moving", "exercising", "cooking", "resting", "creating" };
        var energyLevel = agent.Energy;
        if (energyLevel > 80) return "energetically " + bodyActions[_random.Next(2)];
        if (energyLevel > 50) return "peacefully " + bodyActions[2 + _random.Next(2)];
        return "gently " + bodyActions[4];
    }

    // Sidereal (Mind) = Thoughts and transpersonal actions (what the agent THINKS) - USING REAL ASTROLOGICAL DATA
    private string GetMindBasedThought(TamagotchiAgent agent)
    {
        // Use agent's REAL sidereal mind traits if available  
        if (agent.AstroCharacter?.MindTraits != null)
        {
            var mindTraits = agent.AstroCharacter.MindTraits;
            var thinkingStyle = mindTraits.ThinkingStyle;
            var learningPattern = mindTraits.LearningPattern;
            
            if (!string.IsNullOrEmpty(thinkingStyle))
            {
                return $"thinking in my {thinkingStyle.ToLower()} way";
            }
            if (!string.IsNullOrEmpty(learningPattern))
            {
                return $"learning through {learningPattern.ToLower()}";
            }
            
            // Use nakshatra influence
            return mindTraits.MoonNakshatra.ToString().ToLower() switch
            {
                "rohini" => "contemplating beauty and creativity",
                "ardra" => "analyzing transformation patterns", 
                "pushya" => "nurturing growth and learning",
                _ => "exploring cosmic connections"
            };
        }
        
        // Fallback to evolution-based thoughts
        var mindThoughts = new[] { "exploring ideas", "seeking wisdom", "understanding patterns", "learning deeply" };
        var evolution = agent.Evolution;
        return evolution > 50 ? "deeply " + mindThoughts[_random.Next(2)] : "gently " + mindThoughts[2 + _random.Next(2)];
    }

    // Draconian (Heart) = Interpersonal relations and emotions (how the agent FEELS and RELATES) - USING REAL ASTROLOGICAL DATA
    private string GetHeartBasedFeeling(TamagotchiAgent agent)
    {
        // Use agent's REAL draconian heart traits if available
        if (agent.AstroCharacter?.HeartTraits != null)
        {
            var heartTraits = agent.AstroCharacter.HeartTraits;
            var emotionalNature = heartTraits.EmotionalNature;
            var loveLanguage = heartTraits.LoveLanguage;
            var soulPurpose = heartTraits.SoulPurpose;
            
            if (!string.IsNullOrEmpty(emotionalNature))
            {
                return $"feel {emotionalNature.ToLower()} in my connections";
            }
            if (!string.IsNullOrEmpty(loveLanguage))
            {
                return $"express love through {loveLanguage.ToLower()}";
            }
            if (!string.IsNullOrEmpty(soulPurpose))
            {
                return $"feel aligned with my purpose of {soulPurpose.ToLower()}";
            }
            
            // Use dragon energy influence
            return heartTraits.NorthNodeEnergy.ToString().ToLower() switch
            {
                "creator" => "feel inspired to create beautiful connections",
                "nurturer" => "feel called to nurture and care for others",
                "harmonizer" => "feel peaceful when bringing harmony to relationships",
                _ => "feel deeply connected to the flow of love"
            };
        }
        
        // Fallback to happiness-based feelings
        var heartFeelings = new[] { "feel connected", "appreciate relationships", "love gently", "feel grateful" };
        var happiness = agent.Happiness;
        return happiness > 70 ? "deeply " + heartFeelings[_random.Next(2)] : "softly " + heartFeelings[2 + _random.Next(2)];
    }

    #endregion

    #region Structure and Experience Generation

    private int DetermineStructureFromResonance(ResonanceSentence resonance, TamagotchiAgent agent)
    {
        // THE BREAKTHROUGH: Sentence structure determines which life aspect is active!
        // Use the resonance sentence Base dimension to select structure
        return resonance.Base.ToLower() switch
        {
            "movement" => 2,   // Movement Structure - Action/Energy games
            "being" => 1,      // Being Structure - Survival/Physical games
            "space" => 3,      // Space Structure - Puzzle/Mental games  
            "design" => 4,     // Design Structure - Builder/Creation games
            "evolution" => 5,  // Evolution Structure - RPG/Growth games
            _ => 1              // Default to Being Structure
        };
    }

    private string GenerateExperienceName(GameArchetype archetype, TamagotchiAgent agent)
    {
        var baseNames = archetype.GameType switch
        {
            "Survival" => new[] { "Life Care", "Wellness Journey", "Daily Harmony", "Self Nurturing" },
            "Action" => new[] { "Energy Flow", "Active Life", "Movement Joy", "Dynamic Days" },
            "Puzzle" => new[] { "Mind Games", "Pattern Play", "Thoughtful Moments", "Mental Harmony" },
            "Builder" => new[] { "Creating Life", "Building Dreams", "Making Progress", "Crafting Joy" },
            "RPG" => new[] { "Life Story", "Growing Up", "Personal Journey", "Becoming Me" },
            _ => new[] { "Daily Life", "Simple Pleasures", "Being Human" }
        };

        return baseNames[_random.Next(baseNames.Length)];
    }

    private List<string> AdaptToAstrologicalNature(string[] baseMechanics, TamagotchiAgent agent)
    {
        var adapted = baseMechanics.ToList();
        
        // Add mechanics based on mind/body/heart astrological nature
        adapted.Add($"Personal actions: {GetBodyBasedAction(agent)}");
        adapted.Add($"Mental focus: {GetMindBasedThought(agent)}");  
        adapted.Add($"Emotional style: {GetHeartBasedFeeling(agent)}");
        
        return adapted;
    }

    private List<string> GenerateLifeGuidelines(TamagotchiAgent agent)
    {
        return new List<string>
        {
            "Follow your body's natural rhythm",
            "Trust your mind's wisdom", 
            "Honor your heart's connections",
            "Play authentically and innocently",
            "Let life unfold naturally through you"
        };
    }

    private float CalculateLifeIntensity(TamagotchiAgent agent)
    {
        return (agent.Energy + agent.Resonance) / 200f;
    }

    private float CalculateLifeRhythm(TamagotchiAgent agent)
    {
        return (agent.Health + agent.Happiness) / 200f;
    }

    private string DetermineExperienceStyle(TamagotchiAgent agent)
    {
        return agent.Happiness switch
        {
            > 80 => "Joyful and playful with bright, cheerful visuals",
            > 60 => "Content and peaceful with warm, comfortable visuals", 
            > 40 => "Thoughtful and calm with gentle, soothing visuals",
            _ => "Quiet and introspective with soft, meditative visuals"
        };
    }

    private string DetermineLifeSoundscape(TamagotchiAgent agent)
    {
        return agent.DNA.PrimaryElement.ToLower() switch
        {
            "fire" => "Energetic music with uplifting rhythms and adventure sounds",
            "earth" => "Grounded melodies with nature sounds and peaceful tones",
            "air" => "Light, airy music with thoughtful harmonies and gentle chimes",
            "water" => "Flowing soundscapes with emotional melodies and healing frequencies",
            _ => "Balanced compositions with Sims-style charm and life-affirming tones"
        };
    }

    private string CreateSimsLifeExperience(GameArchetype archetype, TamagotchiAgent agent)
    {
        var activity = archetype.GameType switch
        {
            "Survival" => "managing your Sim's basic needs while discovering your natural wellness routine",
            "Action" => "keeping your Sim active and energetic through fun physical activities", 
            "Puzzle" => "engaging your Sim's mind with creative problem-solving and artistic pursuits",
            "Builder" => "helping your Sim build and improve their living space and life skills",
            "RPG" => "guiding your Sim through life milestones and meaningful relationships",
            _ => "living a full and authentic Sim life"
        };
        
        return $"Your Sim is {activity} - {archetype.HumanDesignTheme}";
    }

    private int CalculateLatticePosition(TamagotchiAgent agent, ResonanceSentence resonance)
    {
        // Simple lattice calculation based on agent properties and resonance
        return ((agent.Age + agent.Evolution + agent.Happiness + resonance.Gate) % 69120) + 1;
    }

    #endregion

    #region Sentence Structure to Game Mapping (Same as before)

    private Dictionary<int, GameImplementationMap> InitializeSentenceToGameMap()
    {
        return new Dictionary<int, GameImplementationMap>
        {
            [1] = new GameImplementationMap
            {
                SentenceStructureId = 1,
                StructureName = "Being Structure",
                PrimaryGameArchetype = "Survival",
                SecondaryArchetypes = new[] { "Platformer", "Action" },
                MechanicFocus = "Physical embodiment, resource management, survival challenges",
                SimsThemes = new[] { "Fitness", "Gardening", "Cooking", "Home maintenance" },
                Description = "Being-focused games emphasize physical presence and material reality"
            },

            [2] = new GameImplementationMap
            {
                SentenceStructureId = 2,
                StructureName = "Movement Structure",
                PrimaryGameArchetype = "Action",
                SecondaryArchetypes = new[] { "Racing", "Platformer" },
                MechanicFocus = "Dynamic movement, energy management, environmental navigation",
                SimsThemes = new[] { "Sports", "Dancing", "Traveling", "Adventure activities" },
                Description = "Movement-focused games emphasize activity and environmental interaction"
            },

            [3] = new GameImplementationMap
            {
                SentenceStructureId = 3,
                StructureName = "Space Structure", 
                PrimaryGameArchetype = "Puzzle",
                SecondaryArchetypes = new[] { "Strategy", "Simulation" },
                MechanicFocus = "Spatial reasoning, pattern recognition, subjective reality manipulation",
                SimsThemes = new[] { "Interior design", "Music", "Art creation", "Meditation" },
                Description = "Space-focused games emphasize form, illusion, and subjective experience"
            },

            [4] = new GameImplementationMap
            {
                SentenceStructureId = 4,
                StructureName = "Design Structure",
                PrimaryGameArchetype = "Builder",
                SecondaryArchetypes = new[] { "SimCity", "Craft" },
                MechanicFocus = "Construction, progression systems, structural manifestation",
                SimsThemes = new[] { "Building", "Crafting", "Career advancement", "Skill development" },
                Description = "Design-focused games emphasize structure, growth, and manifestation"
            },

            [5] = new GameImplementationMap
            {
                SentenceStructureId = 5,
                StructureName = "Evolution Structure",
                PrimaryGameArchetype = "RPG", 
                SecondaryArchetypes = new[] { "Memory", "Story" },
                MechanicFocus = "Character development, memory integration, conscious evolution",
                SimsThemes = new[] { "Learning", "Relationships", "Life goals", "Personal growth" },
                Description = "Evolution-focused games emphasize memory, integration, and consciousness development"
            }
        };
    }

    private Dictionary<string, GameArchetype> InitializeGameArchetypes()
    {
        return new Dictionary<string, GameArchetype>
        {
            ["Survival"] = new GameArchetype
            {
                GameType = "Survival",
                CoreMechanics = new[] { "Resource gathering", "Health management", "Environmental adaptation", "Basic needs fulfillment" },
                ClassicExamples = new[] { "The Sims Basic Needs", "Minecraft Survival", "Don't Starve" },
                CodonMappings = new[] { 25, 26, 27 },
                HumanDesignTheme = "Sacral authority - knowing what sustains you"
            },

            ["Action"] = new GameArchetype
            {
                GameType = "Action",
                CoreMechanics = new[] { "Real-time movement", "Reaction timing", "Energy expenditure", "Dynamic challenges" },
                ClassicExamples = new[] { "Pong", "Pac-Man", "Super Mario Bros" },
                CodonMappings = new[] { 1, 8, 33 },
                HumanDesignTheme = "Splenic authority - in-the-moment awareness"
            },

            ["Puzzle"] = new GameArchetype
            {
                GameType = "Puzzle",
                CoreMechanics = new[] { "Pattern matching", "Spatial reasoning", "Logic challenges", "Mental flexibility" },
                ClassicExamples = new[] { "Tetris", "Portal", "Monument Valley" },
                CodonMappings = new[] { 11, 43, 62 },
                HumanDesignTheme = "Mental authority - processing and deciding"
            },

            ["Builder"] = new GameArchetype
            {
                GameType = "Builder",
                CoreMechanics = new[] { "Construction", "Resource transformation", "Progressive complexity", "System design" },
                ClassicExamples = new[] { "SimCity", "Minecraft Creative", "Civilization" },
                CodonMappings = new[] { 3, 27, 50 },
                HumanDesignTheme = "Self-projected authority - manifesting vision"
            },

            ["RPG"] = new GameArchetype
            {
                GameType = "RPG",
                CoreMechanics = new[] { "Character progression", "Story integration", "Choice consequences", "Memory building" },
                ClassicExamples = new[] { "The Sims Life Goals", "Animal Crossing", "Stardew Valley" },
                CodonMappings = new[] { 13, 49, 55 },
                HumanDesignTheme = "Lunar authority - cycles of experience and integration"
            }
        };
    }

    #endregion
}

#region Life Experience Data Models

public class LifeExperienceResult
{
    public string LifeAspect { get; set; } = string.Empty;
    public string ExperienceName { get; set; } = string.Empty;
    
    // Experience mechanics adapted to agent's mind/body/heart nature
    public List<string> ExperienceMechanics { get; set; } = new();
    public List<string> LifeGuidelines { get; set; } = new();
    
    // Experience modulated by agent's natural flow
    public float IntensityLevel { get; set; }
    public float LifeRhythm { get; set; }
    public string ExperienceStyle { get; set; } = string.Empty;
    public string LifeSoundscape { get; set; } = string.Empty;
    
    // Sims life experience with hidden Human Design themes
    public string SimsLifeExperience { get; set; } = string.Empty;
    
    // Agent's English Communication about their experience
    public string AgentSpeech { get; set; } = string.Empty;
    
    // Context
    public int LatticePosition { get; set; }
    public string SentenceStructure { get; set; } = string.Empty;
    
    public string AgentId { get; set; } = string.Empty;
    public DateTime GeneratedAt { get; set; }
}

public class GameImplementationMap
{
    public int SentenceStructureId { get; set; }
    public string StructureName { get; set; } = string.Empty;
    public string PrimaryGameArchetype { get; set; } = string.Empty;
    public string[] SecondaryArchetypes { get; set; } = Array.Empty<string>();
    public string MechanicFocus { get; set; } = string.Empty;
    public string[] SimsThemes { get; set; } = Array.Empty<string>();
    public string Description { get; set; } = string.Empty;
}

public class GameArchetype
{
    public string GameType { get; set; } = string.Empty;
    public string[] CoreMechanics { get; set; } = Array.Empty<string>();
    public string[] ClassicExamples { get; set; } = Array.Empty<string>();
    public int[] CodonMappings { get; set; } = Array.Empty<int>();
    public string HumanDesignTheme { get; set; } = string.Empty;
}

#endregion