using ModularSimWorld.Components;

namespace ModularSimWorld.Components;

/// <summary>
/// Dream sequence engine that creates personalized dream experiences
/// based on astrological character traits
/// </summary>
public class DreamSequenceEngine
{
    private readonly Random _random;
    private readonly AstrologicalWeatherService _weatherService;

    public DreamSequenceEngine(AstrologicalWeatherService weatherService)
    {
        _random = new Random();
        _weatherService = weatherService;
        Console.WriteLine("[Dreams] Dream Sequence Engine initialized");
    }

    /// <summary>
    /// Generates a personalized dream sequence for an astrological character
    /// </summary>
    public async Task<DreamExperience> GenerateDreamSequence(AstrologicalCharacter character)
    {
        var currentWeather = await _weatherService.GetCurrentWeatherAsync();
        
        // Generate dream based on character traits and current astrological conditions
        var dreamType = DetermineDreamType(character, currentWeather);
        var dreamSetting = GenerateDreamSetting(character.BodyTraits, currentWeather);
        var dreamNarrative = GenerateDreamNarrative(character, dreamType, currentWeather);
        var dreamChoices = GenerateDreamChoices(character, dreamType);
        
        var dream = new DreamExperience
        {
            CharacterName = character.Name,
            DreamType = dreamType,
            Setting = dreamSetting,
            Narrative = dreamNarrative,
            CosmicInfluences = GetCosmicInfluences(character, currentWeather),
            AvailableChoices = dreamChoices,
            AstrologicalResonance = CalculateAstrologicalResonance(character, currentWeather),
            Timestamp = DateTime.UtcNow
        };

        Console.WriteLine($"[Dreams] Generated {dreamType} dream sequence for {character.Name}");
        return dream;
    }

    /// <summary>
    /// Processes a dream choice and generates the consequence
    /// </summary>
    public DreamConsequence ProcessDreamChoice(DreamExperience dream, int choiceIndex, AstrologicalCharacter character)
    {
        if (choiceIndex < 0 || choiceIndex >= dream.AvailableChoices.Count)
            return new DreamConsequence { Description = "The dream fades as indecision takes hold...", EffectType = "Neutral" };

        var choice = dream.AvailableChoices[choiceIndex];
        var consequence = GenerateConsequence(choice, character, dream);

        Console.WriteLine($"[Dreams] {character.Name} chose: {choice.Description}");
        return consequence;
    }

    #region Dream Generation Methods

    private DreamType DetermineDreamType(AstrologicalCharacter character, AstrologicalWeatherReport weather)
    {
        // Use character traits and current astrological weather to determine dream type
        var bodyWeight = character.BodyTraits.PhysicalStrength + character.BodyTraits.Agility;
        var mindWeight = character.MindTraits.Intellect + character.MindTraits.Intuition;
        var heartWeight = character.HeartTraits.Passion + character.HeartTraits.EmotionalDepth;

        // Factor in current lunar phase
        var lunarInfluence = weather.MoonPhase switch
        {
            MoonPhase.NewMoon => "Prophetic",
            MoonPhase.FullMoon => "Lucid", 
            MoonPhase.WaxingCrescent => "Adventure",
            MoonPhase.WaningCrescent => "Memory",
            _ => "Symbolic"
        };

        // Determine dominant trait and combine with lunar influence
        if (heartWeight >= bodyWeight && heartWeight >= mindWeight)
        {
            return lunarInfluence switch
            {
                "Prophetic" => DreamType.PsychicVision,
                "Lucid" => DreamType.EmotionalHealing,
                "Adventure" => DreamType.LoveQuest,
                "Memory" => DreamType.PastLifeMemory,
                _ => DreamType.EmotionalHealing
            };
        }
        else if (mindWeight >= bodyWeight)
        {
            return lunarInfluence switch
            {
                "Prophetic" => DreamType.CosmicRevelation,
                "Lucid" => DreamType.AstralJourney,
                "Adventure" => DreamType.IntellectualQuest,
                "Memory" => DreamType.AncientWisdom,
                _ => DreamType.SymbolicVision
            };
        }
        else
        {
            return lunarInfluence switch
            {
                "Prophetic" => DreamType.PhysicalTransformation,
                "Lucid" => DreamType.AdventureQuest,
                "Adventure" => DreamType.HeroicJourney,
                "Memory" => DreamType.WarriorMemory,
                _ => DreamType.AdventureQuest
            };
        }
    }

    private DreamSetting GenerateDreamSetting(TropicalBodyTraits bodyTraits, AstrologicalWeatherReport weather)
    {
        var element = bodyTraits.Element;
        var weatherElement = weather.DominantElement;

        // Combine character's natal element with current dominant element
        var settingDescription = (element, weatherElement) switch
        {
            (TropicalElement.Fire, ElementType.Fire) => "A volcanic landscape with rivers of liquid starlight",
            (TropicalElement.Fire, ElementType.Water) => "A misty volcano beside a cosmic ocean",
            (TropicalElement.Fire, ElementType.Earth) => "Ancient ruins glowing with inner fire beneath mountain peaks",
            (TropicalElement.Fire, ElementType.Air) => "Floating islands connected by bridges of flame",

            (TropicalElement.Water, ElementType.Water) => "An underwater palace with rooms filled with liquid moonbeams",
            (TropicalElement.Water, ElementType.Fire) => "A steaming hot spring surrounded by phoenixes",
            (TropicalElement.Water, ElementType.Earth) => "A mystical grotto with waterfalls of liquid crystal",
            (TropicalElement.Water, ElementType.Air) => "Clouds that rain upward into crystal pools",

            (TropicalElement.Earth, ElementType.Earth) => "A magnificent library carved from living stone, filled with glowing books",
            (TropicalElement.Earth, ElementType.Air) => "A floating garden with roots that reach between worlds",
            (TropicalElement.Earth, ElementType.Fire) => "A forge where mountains are shaped by celestial blacksmiths",
            (TropicalElement.Earth, ElementType.Water) => "An underground city with streets of flowing mercury",

            (TropicalElement.Air, ElementType.Air) => "A vast sky realm where thoughts take physical form",
            (TropicalElement.Air, ElementType.Earth) => "A tower that reaches from the earth's core to the stars",
            (TropicalElement.Air, ElementType.Fire) => "A windstorm filled with dancing flames and singing voices",
            (TropicalElement.Air, ElementType.Water) => "Clouds that hold entire oceans, with fish that fly through the air",

            _ => "A realm where the elements dance together in perfect harmony"
        };

        return new DreamSetting
        {
            Description = settingDescription,
            PrimaryElement = element.ToString(),
            SecondaryElement = weatherElement.ToString(),
            AmbientMood = GetAmbientMood(weather.MoonPhase, weather.EnergyLevel)
        };
    }

    private string GetAmbientMood(MoonPhase moonPhase, int energyLevel)
    {
        var baseEnergy = energyLevel > 70 ? "vibrant" : energyLevel > 40 ? "calm" : "mysterious";
        
        return moonPhase switch
        {
            MoonPhase.NewMoon => $"A {baseEnergy} darkness filled with infinite potential",
            MoonPhase.WaxingCrescent => $"A {baseEnergy} twilight where new possibilities emerge",
            MoonPhase.FirstQuarter => $"A {baseEnergy} evening of balance and decision",
            MoonPhase.WaxingGibbous => $"A {baseEnergy} dusk building toward revelation",
            MoonPhase.FullMoon => $"A {baseEnergy} brilliance that illuminates all hidden truths",
            MoonPhase.WaningGibbous => $"A {baseEnergy} afterglow of wisdom and gratitude",
            MoonPhase.ThirdQuarter => $"A {baseEnergy} dawn of release and letting go",
            MoonPhase.WaningCrescent => $"A {baseEnergy} pre-dawn stillness of reflection",
            _ => $"A {baseEnergy} atmosphere of cosmic mystery"
        };
    }

    private List<string> GenerateDreamNarrative(AstrologicalCharacter character, DreamType dreamType, AstrologicalWeatherReport weather)
    {
        var narrative = new List<string>();

        // Opening scene based on dream type
        narrative.Add(GetDreamOpening(dreamType, character.Name));
        
        // Middle development based on character traits
        narrative.AddRange(GetCharacterSpecificNarrative(character, dreamType, weather));
        
        // Climax based on current astrological conditions
        narrative.Add(GetAstrologicalClimax(weather, character));

        return narrative;
    }

    private string GetDreamOpening(DreamType dreamType, string characterName)
    {
        return dreamType switch
        {
            DreamType.AstralJourney => $"{characterName} feels their consciousness lifting from their physical form...",
            DreamType.PsychicVision => $"{characterName} sees through eyes that are not their own...",
            DreamType.EmotionalHealing => $"{characterName}'s heart begins to glow with warm, healing light...",
            DreamType.AdventureQuest => $"{characterName} stands at the edge of an unknown realm, adventure calling...",
            DreamType.CosmicRevelation => $"The universe opens like a vast book before {characterName}...",
            DreamType.LoveQuest => $"{characterName} feels drawn by an invisible thread of connection...",
            DreamType.PastLifeMemory => $"Memories that aren't quite {characterName}'s begin to surface...",
            DreamType.IntellectualQuest => $"{characterName} discovers a puzzle that spans dimensions...",
            DreamType.PhysicalTransformation => $"{characterName} feels their body beginning to change and evolve...",
            DreamType.HeroicJourney => $"{characterName} hears the call to become more than they ever imagined...",
            DreamType.AncientWisdom => $"Ancient voices whisper forgotten knowledge to {characterName}...",
            DreamType.WarriorMemory => $"{characterName} remembers battles fought across time and space...",
            DreamType.SymbolicVision => $"Symbols and archetypes dance around {characterName} in meaningful patterns...",
            _ => $"{characterName} finds themselves in a realm beyond ordinary dreams..."
        };
    }

    private List<string> GetCharacterSpecificNarrative(AstrologicalCharacter character, DreamType dreamType, AstrologicalWeatherReport weather)
    {
        var narrative = new List<string>();

        // Add body-based narrative (Tropical astrology influence)
        var bodyNarrative = character.BodyTraits.ZodiacSign switch
        {
            TropicalZodiacSign.Aries => "A ram with horns of pure starlight offers to guide the way forward.",
            TropicalZodiacSign.Taurus => "The earth beneath your feet pulses with ancient, steady rhythms.",
            TropicalZodiacSign.Gemini => "Twin butterflies with wings of light dance around you, offering different paths.",
            TropicalZodiacSign.Cancer => "A silver crab emerges from moonlit waters, carrying the wisdom of tides.",
            TropicalZodiacSign.Leo => "A golden lion's mane catches fire with the light of a thousand suns.",
            TropicalZodiacSign.Virgo => "Perfect crystals form in the air around you, each one containing a piece of truth.",
            TropicalZodiacSign.Libra => "Scales of justice appear, but they weigh dreams against reality.",
            TropicalZodiacSign.Scorpio => "A phoenix rises from ashes that smell of transformation and rebirth.",
            TropicalZodiacSign.Sagittarius => "An arrow of pure intention shoots across the cosmic sky.",
            TropicalZodiacSign.Capricorn => "A mountain goat climbs impossible peaks, each step creating new stars.",
            TropicalZodiacSign.Aquarius => "Water flows upward, carrying the hopes and dreams of humanity.",
            TropicalZodiacSign.Pisces => "Two fish swim in opposite directions through an ocean of liquid light.",
            _ => "Your zodiac essence manifests as pure energy around you."
        };
        narrative.Add(bodyNarrative);

        // Add mind-based narrative (Sidereal astrology influence)
        var mindNarrative = $"Your consciousness resonates with the {character.MindTraits.Nakshatra} constellation, ";
        mindNarrative += character.MindTraits.Nakshatra switch
        {
            Nakshatra.Ashwini => "bringing the healing energy of the cosmic horsemen.",
            Nakshatra.Bharani => "carrying the transformative power of creation and destruction.",
            Nakshatra.Krittika => "igniting the fire of spiritual purification.",
            Nakshatra.Rohini => "nurturing growth like the red star of creation.",
            Nakshatra.Mrigashira => "searching for truth like the cosmic deer.",
            _ => "connecting you to ancient stellar wisdom."
        };
        narrative.Add(mindNarrative);

        // Add heart-based narrative (Draconian astrology influence)
        var heartNarrative = $"The {character.HeartTraits.DragonEnergy} Dragon stirs within your heart, ";
        heartNarrative += character.HeartTraits.DragonEnergy switch
        {
            DragonEnergyType.FireDragon => "breathing flames of passion and courage into your spirit.",
            DragonEnergyType.WaterDragon => "flowing with deep emotional currents and intuitive wisdom.",
            DragonEnergyType.EarthDragon => "grounding you with ancient stability and enduring strength.",
            DragonEnergyType.AirDragon => "lifting your thoughts with winds of change and inspiration.",
            _ => "awakening your deepest spiritual essence."
        };
        narrative.Add(heartNarrative);

        return narrative;
    }

    private string GetAstrologicalClimax(AstrologicalWeatherReport weather, AstrologicalCharacter character)
    {
        var climax = "As the cosmic energies reach their peak, ";
        
        if (weather.EnergyLevel > 75)
        {
            climax += $"brilliant {weather.DominantElement} energy erupts around you, and you realize you must choose: ";
        }
        else if (weather.EnergyLevel > 40)
        {
            climax += $"steady {weather.DominantElement} currents flow through the dream, offering you a moment to decide: ";
        }
        else
        {
            climax += $"subtle {weather.DominantElement} whispers guide you to a crucial crossroads where you must: ";
        }

        return climax;
    }

    private List<DreamChoice> GenerateDreamChoices(AstrologicalCharacter character, DreamType dreamType)
    {
        var choices = new List<DreamChoice>();

        // Generate choices based on the three astrological systems
        var bodyChoice = GenerateBodyChoice(character.BodyTraits, dreamType);
        var mindChoice = GenerateMindChoice(character.MindTraits, dreamType);
        var heartChoice = GenerateHeartChoice(character.HeartTraits, dreamType);

        choices.AddRange(new[] { bodyChoice, mindChoice, heartChoice });

        // Add a fourth "intuitive" choice based on current astrological weather
        choices.Add(new DreamChoice
        {
            Description = "Trust your cosmic intuition and let the universe decide",
            ChoiceType = "Cosmic",
            RequiredTrait = "Universal Connection",
            PotentialOutcome = "Surrender to divine will and see what unfolds"
        });

        return choices;
    }

    private DreamChoice GenerateBodyChoice(TropicalBodyTraits bodyTraits, DreamType dreamType)
    {
        var actionVerb = bodyTraits.Element switch
        {
            TropicalElement.Fire => "Charge forward",
            TropicalElement.Earth => "Stand your ground",
            TropicalElement.Air => "Rise above",
            TropicalElement.Water => "Flow around",
            _ => "Move with"
        };

        return new DreamChoice
        {
            Description = $"{actionVerb} with the strength of {bodyTraits.ZodiacSign}",
            ChoiceType = "Physical",
            RequiredTrait = $"{bodyTraits.Element} Element",
            PotentialOutcome = $"Channel your physical {bodyTraits.Element.ToString().ToLower()} energy"
        };
    }

    private DreamChoice GenerateMindChoice(SiderealMindTraits mindTraits, DreamType dreamType)
    {
        var mentalAction = mindTraits.Intellect > mindTraits.Intuition 
            ? "Analyze the situation carefully"
            : "Trust your inner knowing";

        return new DreamChoice
        {
            Description = $"{mentalAction} using {mindTraits.Nakshatra} wisdom",
            ChoiceType = "Mental",
            RequiredTrait = $"{mindTraits.Nakshatra} Connection",
            PotentialOutcome = $"Apply the ancient wisdom of {mindTraits.Nakshatra}"
        };
    }

    private DreamChoice GenerateHeartChoice(DraconianHeartTraits heartTraits, DreamType dreamType)
    {
        var emotionalAction = heartTraits.Passion > heartTraits.Empathy
            ? "Follow your passionate heart"
            : "Act with compassionate understanding";

        return new DreamChoice
        {
            Description = $"{emotionalAction} guided by {heartTraits.DragonEnergy}",
            ChoiceType = "Emotional",
            RequiredTrait = $"{heartTraits.DragonEnergy} Energy",
            PotentialOutcome = $"Embody the power of the {heartTraits.DragonEnergy}"
        };
    }

    private List<string> GetCosmicInfluences(AstrologicalCharacter character, AstrologicalWeatherReport weather)
    {
        var influences = new List<string>();

        // Current planetary influences
        foreach (var planet in weather.PlanetaryHighlights.Take(3))
        {
            influences.Add($"{planet.Planet} in {planet.Sign}: {planet.Influence}");
        }

        // Character resonance with current conditions
        influences.Add($"Your {character.BodyTraits.Element} nature resonates with the current {weather.DominantElement} energy");
        influences.Add($"Moon in {weather.MoonSign} activates your {GetLunarConnection(character, weather.MoonSign)}");

        return influences;
    }

    private string GetLunarConnection(AstrologicalCharacter character, string moonSign)
    {
        // Find connections between character traits and current moon sign
        if (character.BodyTraits.ZodiacSign.ToString() == moonSign)
        {
            return "natal sun energy (heightened physical vitality)";
        }
        
        if (character.HeartTraits.NorthNode == moonSign || character.HeartTraits.SouthNode == moonSign)
        {
            return "karmic node (past-life memories may surface)";
        }

        return "subconscious emotional patterns";
    }

    private int CalculateAstrologicalResonance(AstrologicalCharacter character, AstrologicalWeatherReport weather)
    {
        var resonance = 50; // Base resonance

        // Element matching
        if (character.BodyTraits.Element.ToString() == weather.DominantElement.ToString())
        {
            resonance += 20;
        }

        // Moon phase influence
        resonance += weather.MoonPhase switch
        {
            MoonPhase.FullMoon => 15,
            MoonPhase.NewMoon => 10,
            _ => 5
        };

        // Energy level influence
        resonance += (weather.EnergyLevel - 50) / 10;

        return Math.Max(1, Math.Min(100, resonance));
    }

    private DreamConsequence GenerateConsequence(DreamChoice choice, AstrologicalCharacter character, DreamExperience dream)
    {
        var baseDescription = choice.ChoiceType switch
        {
            "Physical" => $"Drawing upon your {character.BodyTraits.Element} essence, you {choice.Description.ToLower()}. ",
            "Mental" => $"Your mind, attuned to {character.MindTraits.Nakshatra}, guides you as you {choice.Description.ToLower()}. ",
            "Emotional" => $"Your heart, empowered by {character.HeartTraits.DragonEnergy}, leads you to {choice.Description.ToLower()}. ",
            "Cosmic" => $"Surrendering to cosmic flow, you {choice.Description.ToLower()}. ",
            _ => $"You {choice.Description.ToLower()}. "
        };

        var outcome = GenerateOutcome(choice, dream.AstrologicalResonance);
        var effect = GetEffectType(choice, dream.AstrologicalResonance);

        return new DreamConsequence
        {
            Description = baseDescription + outcome,
            EffectType = effect,
            ResonanceGained = CalculateResonanceGain(choice, dream.AstrologicalResonance),
            NewInsight = GenerateInsight(choice, character)
        };
    }

    private string GenerateOutcome(DreamChoice choice, int resonance)
    {
        var success = resonance > 70 ? "remarkable" : resonance > 40 ? "moderate" : "subtle";
        
        return choice.ChoiceType switch
        {
            "Physical" => $"The {success} strength of your body manifests in the dream realm, creating tangible changes around you.",
            "Mental" => $"Your {success} intellectual clarity pierces through dream illusions, revealing hidden truths.",
            "Emotional" => $"The {success} power of your heart transforms the emotional landscape of the dream.",
            "Cosmic" => $"The universe responds with {success} synchronicity, aligning dream events with your highest good.",
            _ => $"A {success} shift occurs in the fabric of the dream."
        };
    }

    private string GetEffectType(DreamChoice choice, int resonance)
    {
        if (resonance > 70) return "Transformative";
        if (resonance > 40) return "Beneficial";
        return "Neutral";
    }

    private int CalculateResonanceGain(DreamChoice choice, int currentResonance)
    {
        return choice.ChoiceType switch
        {
            "Physical" => _random.Next(1, 6),
            "Mental" => _random.Next(1, 6), 
            "Emotional" => _random.Next(1, 6),
            "Cosmic" => _random.Next(5, 11),
            _ => _random.Next(1, 4)
        };
    }

    private string GenerateInsight(DreamChoice choice, AstrologicalCharacter character)
    {
        return choice.ChoiceType switch
        {
            "Physical" => $"You understand more deeply how your {character.BodyTraits.Element} nature shapes your physical experience.",
            "Mental" => $"The wisdom of {character.MindTraits.Nakshatra} reveals new layers of understanding about your mental patterns.",
            "Emotional" => $"Your {character.HeartTraits.DragonEnergy} energy shows you the true power of your emotional nature.",
            "Cosmic" => "You glimpse the vast web of cosmic connections that link all things together.",
            _ => "A subtle understanding about the nature of dreams and reality emerges."
        };
    }

    #endregion
}

#region Dream System Data Models

public class DreamExperience
{
    public string CharacterName { get; set; } = string.Empty;
    public DreamType DreamType { get; set; }
    public DreamSetting Setting { get; set; } = new();
    public List<string> Narrative { get; set; } = new();
    public List<string> CosmicInfluences { get; set; } = new();
    public List<DreamChoice> AvailableChoices { get; set; } = new();
    public int AstrologicalResonance { get; set; }
    public DateTime Timestamp { get; set; }
}

public class DreamSetting
{
    public string Description { get; set; } = string.Empty;
    public string PrimaryElement { get; set; } = string.Empty;
    public string SecondaryElement { get; set; } = string.Empty;
    public string AmbientMood { get; set; } = string.Empty;
}

public class DreamChoice
{
    public string Description { get; set; } = string.Empty;
    public string ChoiceType { get; set; } = string.Empty; // Physical, Mental, Emotional, Cosmic
    public string RequiredTrait { get; set; } = string.Empty;
    public string PotentialOutcome { get; set; } = string.Empty;
}

public class DreamConsequence
{
    public string Description { get; set; } = string.Empty;
    public string EffectType { get; set; } = string.Empty; // Transformative, Beneficial, Neutral
    public int ResonanceGained { get; set; }
    public string NewInsight { get; set; } = string.Empty;
}

public enum DreamType
{
    AstralJourney, PsychicVision, EmotionalHealing, AdventureQuest,
    CosmicRevelation, LoveQuest, PastLifeMemory, IntellectualQuest,
    PhysicalTransformation, HeroicJourney, AncientWisdom, WarriorMemory,
    SymbolicVision
}

#endregion