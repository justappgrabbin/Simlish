using System.Text.Json;

namespace ModularSimWorld.Components;

/// <summary>
/// Autonomous agent engine that creates agents based on user traits
/// Agents act independently with user guidance only
/// </summary>
public class AutonomousAgentEngine
{
    private readonly Random _random;
    private readonly AstrologicalWeatherService _weatherService;
    private readonly Dictionary<string, AutonomousAgent> _activeAgents;

    public AutonomousAgentEngine(AstrologicalWeatherService weatherService)
    {
        _random = new Random();
        _weatherService = weatherService;
        _activeAgents = new Dictionary<string, AutonomousAgent>();
        Console.WriteLine("[Agents] Autonomous Agent Engine initialized");
    }

    /// <summary>
    /// Creates an autonomous agent from an astrological character
    /// </summary>
    public AutonomousAgent CreateAgent(AstrologicalCharacter character)
    {
        var agent = new AutonomousAgent
        {
            Id = Guid.NewGuid().ToString(),
            Name = character.Name,
            BaseCharacter = character,
            CurrentState = AgentState.Idle,
            Energy = 100,
            Skills = InitializeSkills(character),
            PersonalityTraits = ExtractPersonalityTraits(character),
            DecisionMaking = CreateDecisionMakingProfile(character),
            Autonomy = 100, // Full autonomy initially
            GuidanceReceptivity = CalculateGuidanceReceptivity(character),
            CreatedAt = DateTime.UtcNow
        };

        _activeAgents[agent.Id] = agent;
        Console.WriteLine($"[Agents] Created autonomous agent: {agent.Name} (ID: {agent.Id})");
        return agent;
    }

    /// <summary>
    /// Drops agent into a scenario - can be called by external architect
    /// </summary>
    public async Task<ScenarioExperience> DropAgentIntoScenario(string agentId, LifeScenario scenario)
    {
        if (!_activeAgents.TryGetValue(agentId, out var agent))
            throw new ArgumentException($"Agent with ID {agentId} not found");

        Console.WriteLine($"[Scenario] Dropping {agent.Name} into: {scenario.Title}");

        // Create exaggerated version of the scenario based on agent traits
        var exaggeratedScenario = ExaggerateScenario(scenario, agent);
        
        // Start autonomous behavior
        agent.CurrentState = AgentState.InScenario;
        agent.CurrentScenario = exaggeratedScenario;

        var experience = new ScenarioExperience
        {
            AgentId = agentId,
            AgentName = agent.Name,
            Scenario = exaggeratedScenario,
            StartTime = DateTime.UtcNow,
            InitialAgentState = CloneAgentState(agent),
            GuidanceHistory = new List<GuidanceAction>(),
            AgentDecisions = new List<AgentDecision>(),
            SkillTests = new List<SkillTest>(),
            IsActive = true
        };

        // Start autonomous behavior loop
        _ = Task.Run(async () => await RunAutonomousBehavior(agent, experience));

        return experience;
    }

    /// <summary>
    /// User provides guidance (nudge) to agent - doesn't guarantee compliance
    /// </summary>
    public GuidanceResult ProvideGuidance(string agentId, string guidanceText, GuidanceType type)
    {
        if (!_activeAgents.TryGetValue(agentId, out var agent))
            return new GuidanceResult { Success = false, Message = "Agent not found" };

        if (agent.CurrentState != AgentState.InScenario)
            return new GuidanceResult { Success = false, Message = "Agent not in scenario" };

        // Agent may or may not follow guidance based on their traits
        var compliance = CalculateGuidanceCompliance(agent, guidanceText, type);
        var guidance = new GuidanceAction
        {
            Timestamp = DateTime.UtcNow,
            GuidanceText = guidanceText,
            Type = type,
            ComplianceLevel = compliance,
            AgentResponse = GenerateAgentResponse(agent, guidanceText, compliance)
        };

        // Add to current scenario experience
        if (agent.CurrentScenario != null)
        {
            // Find and update the experience (would need reference tracking in real implementation)
            Console.WriteLine($"[Guidance] {agent.Name} receives guidance: '{guidanceText}' (Compliance: {compliance}%)");
        }

        return new GuidanceResult 
        { 
            Success = true, 
            Message = guidance.AgentResponse,
            ComplianceLevel = compliance
        };
    }

    /// <summary>
    /// External system can inject scenarios - for architect agent integration
    /// </summary>
    public async Task<bool> InjectScenario(string agentId, string scenarioJson)
    {
        try
        {
            var scenario = JsonSerializer.Deserialize<LifeScenario>(scenarioJson);
            if (scenario == null) return false;

            await DropAgentIntoScenario(agentId, scenario);
            return true;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[Scenario] Failed to inject scenario: {ex.Message}");
            return false;
        }
    }

    #region Private Methods

    private Dictionary<SkillType, int> InitializeSkills(AstrologicalCharacter character)
    {
        var skills = new Dictionary<SkillType, int>();

        // Initialize based on astrological traits
        skills[SkillType.Communication] = (character.MindTraits.Intellect + character.HeartTraits.Empathy) / 2;
        skills[SkillType.Leadership] = (character.BodyTraits.PhysicalStrength + character.HeartTraits.Passion) / 2;
        skills[SkillType.Creativity] = character.MindTraits.Creativity;
        skills[SkillType.EmotionalIntelligence] = character.HeartTraits.EmotionalDepth;
        skills[SkillType.ProblemSolving] = (character.MindTraits.Intellect + character.MindTraits.Focus) / 2;
        skills[SkillType.Resilience] = (character.BodyTraits.Endurance + character.HeartTraits.EmotionalDepth) / 2;
        skills[SkillType.Adaptability] = character.MindTraits.Intuition;
        skills[SkillType.SocialSkills] = (character.HeartTraits.Empathy + character.MindTraits.Creativity) / 2;

        return skills;
    }

    private PersonalityProfile ExtractPersonalityTraits(AstrologicalCharacter character)
    {
        return new PersonalityProfile
        {
            Impulsiveness = character.BodyTraits.Element == TropicalElement.Fire ? 80 : 40,
            Patience = character.BodyTraits.Element == TropicalElement.Earth ? 80 : 40,
            SocialOrientation = character.BodyTraits.Element == TropicalElement.Air ? 80 : 40,
            EmotionalDepth = character.HeartTraits.EmotionalDepth,
            LogicalThinking = character.MindTraits.Intellect,
            IntuitiveThinking = character.MindTraits.Intuition,
            RiskTolerance = character.HeartTraits.Passion,
            StressResponse = GetStressResponse(character)
        };
    }

    private StressResponseType GetStressResponse(AstrologicalCharacter character)
    {
        return character.BodyTraits.Element switch
        {
            TropicalElement.Fire => StressResponseType.Fight,
            TropicalElement.Earth => StressResponseType.Endure,
            TropicalElement.Air => StressResponseType.Analyze,
            TropicalElement.Water => StressResponseType.Adapt,
            _ => StressResponseType.Analyze
        };
    }

    private DecisionMakingProfile CreateDecisionMakingProfile(AstrologicalCharacter character)
    {
        return new DecisionMakingProfile
        {
            PrimaryDecisionStyle = character.MindTraits.Intellect > character.MindTraits.Intuition 
                ? DecisionStyle.Analytical 
                : DecisionStyle.Intuitive,
            SecondaryDecisionStyle = character.HeartTraits.Passion > character.HeartTraits.Empathy
                ? DecisionStyle.Emotional
                : DecisionStyle.Social,
            DecisionSpeed = character.BodyTraits.Agility + character.MindTraits.Focus,
            RiskAssessment = character.MindTraits.Intellect,
            ValueSystem = ExtractValueSystem(character)
        };
    }

    private List<string> ExtractValueSystem(AstrologicalCharacter character)
    {
        var values = new List<string>();
        
        // Based on zodiac sign
        values.AddRange(character.BodyTraits.ZodiacSign switch
        {
            TropicalZodiacSign.Aries => new[] { "Independence", "Achievement", "Leadership" },
            TropicalZodiacSign.Taurus => new[] { "Stability", "Security", "Beauty" },
            TropicalZodiacSign.Gemini => new[] { "Knowledge", "Communication", "Variety" },
            TropicalZodiacSign.Cancer => new[] { "Family", "Nurturing", "Emotional Security" },
            TropicalZodiacSign.Leo => new[] { "Recognition", "Creativity", "Self-Expression" },
            TropicalZodiacSign.Virgo => new[] { "Perfection", "Service", "Health" },
            TropicalZodiacSign.Libra => new[] { "Harmony", "Justice", "Relationships" },
            TropicalZodiacSign.Scorpio => new[] { "Truth", "Transformation", "Intensity" },
            TropicalZodiacSign.Sagittarius => new[] { "Freedom", "Adventure", "Philosophy" },
            TropicalZodiacSign.Capricorn => new[] { "Achievement", "Responsibility", "Legacy" },
            TropicalZodiacSign.Aquarius => new[] { "Innovation", "Humanity", "Independence" },
            TropicalZodiacSign.Pisces => new[] { "Compassion", "Spirituality", "Creativity" },
            _ => new[] { "Balance", "Growth", "Understanding" }
        });

        return values;
    }

    private int CalculateGuidanceReceptivity(AstrologicalCharacter character)
    {
        // Higher empathy and lower impulsiveness = more receptive to guidance
        var empathy = character.HeartTraits.Empathy;
        var stubbornness = character.BodyTraits.Element == TropicalElement.Earth ? 80 : 40;
        var independence = character.BodyTraits.Element == TropicalElement.Fire ? 80 : 40;

        return Math.Max(10, Math.Min(90, empathy - (stubbornness + independence) / 2 + 50));
    }

    private LifeScenario ExaggerateScenario(LifeScenario scenario, AutonomousAgent agent)
    {
        // Create exaggerated version based on agent's weakest skills
        var weakestSkill = agent.Skills.OrderBy(s => s.Value).First();
        var intensityMultiplier = (100 - weakestSkill.Value) / 50.0; // Higher for weaker skills

        return new LifeScenario
        {
            Id = Guid.NewGuid().ToString(),
            Title = $"[INTENSIVE] {scenario.Title}",
            Description = $"{scenario.Description} (Intensity: {intensityMultiplier:F1}x)",
            Category = scenario.Category,
            DifficultyLevel = Math.Min(100, (int)(scenario.DifficultyLevel * intensityMultiplier)),
            TargetSkills = scenario.TargetSkills,
            TimeLimit = scenario.TimeLimit,
            SuccessMetrics = scenario.SuccessMetrics,
            Challenges = ExaggerateChallenges(scenario.Challenges, agent, intensityMultiplier),
            ContextualFactors = AddPersonalizedFactors(scenario.ContextualFactors, agent)
        };
    }

    private List<string> ExaggerateChallenges(List<string> challenges, AutonomousAgent agent, double multiplier)
    {
        var exaggerated = new List<string>();
        
        foreach (var challenge in challenges)
        {
            // Amplify challenges based on agent's personality weaknesses
            var amplified = agent.PersonalityTraits.Impulsiveness > 70 
                ? $"{challenge} (with time pressure and distractions)"
                : $"{challenge} (with complex decision points)";
                
            exaggerated.Add(amplified);
        }

        // Add personalized challenges based on astrological profile
        var personalChallenge = agent.BaseCharacter.BodyTraits.Element switch
        {
            TropicalElement.Fire => "Managing impulsive reactions under extreme pressure",
            TropicalElement.Earth => "Adapting quickly to rapidly changing circumstances",
            TropicalElement.Air => "Making decisive choices without overthinking",
            TropicalElement.Water => "Maintaining objectivity while emotions run high",
            _ => "Balancing multiple competing priorities"
        };
        
        exaggerated.Add(personalChallenge);
        return exaggerated;
    }

    private Dictionary<string, string> AddPersonalizedFactors(Dictionary<string, string> factors, AutonomousAgent agent)
    {
        var personalized = new Dictionary<string, string>(factors);
        
        // Add factors based on agent's astrological makeup
        personalized["AstrologicalInfluence"] = $"{agent.BaseCharacter.BodyTraits.ZodiacSign} nature affecting physical responses";
        personalized["MentalPattern"] = $"{agent.BaseCharacter.MindTraits.Nakshatra} influencing thought processes";
        personalized["EmotionalCore"] = $"{agent.BaseCharacter.HeartTraits.DragonEnergy} driving emotional reactions";
        
        return personalized;
    }

    private async Task RunAutonomousBehavior(AutonomousAgent agent, ScenarioExperience experience)
    {
        while (agent.CurrentState == AgentState.InScenario && experience.IsActive)
        {
            // Agent makes autonomous decisions based on their traits
            var decision = await MakeAutonomousDecision(agent);
            experience.AgentDecisions.Add(decision);

            // Process the decision and its effects
            var outcome = ProcessDecision(agent, decision);
            
            // Update agent state based on outcome
            UpdateAgentState(agent, outcome);

            // Check if scenario is complete
            if (IsScenarioComplete(agent, experience))
            {
                await CompleteScenario(agent, experience);
                break;
            }

            // Wait before next decision cycle
            await Task.Delay(_random.Next(2000, 5000));
        }
    }

    private async Task<AgentDecision> MakeAutonomousDecision(AutonomousAgent agent)
    {
        var weather = await _weatherService.GetCurrentWeatherAsync();
        
        // Agent decides based on personality, skills, and current astrological conditions
        var availableActions = GetAvailableActions(agent);
        var chosenAction = SelectActionBasedOnTraits(agent, availableActions, weather);

        return new AgentDecision
        {
            Timestamp = DateTime.UtcNow,
            Action = chosenAction,
            Reasoning = GenerateDecisionReasoning(agent, chosenAction),
            ConfidenceLevel = CalculateConfidence(agent, chosenAction),
            InfluencingFactors = GetInfluencingFactors(agent, weather)
        };
    }

    private List<string> GetAvailableActions(AutonomousAgent agent)
    {
        // Actions based on current scenario and agent state
        return agent.CurrentScenario?.Category switch
        {
            "Work" => new List<string> { "Analyze problem", "Consult team", "Take immediate action", "Request more information", "Delegate task" },
            "Relationships" => new List<string> { "Communicate directly", "Listen actively", "Seek compromise", "Take time to think", "Express emotions" },
            "Personal" => new List<string> { "Reflect on values", "Seek guidance", "Trust instincts", "Gather more data", "Take calculated risk" },
            "Crisis" => new List<string> { "Act immediately", "Assess situation", "Call for help", "Stay calm and observe", "Follow protocol" },
            _ => new List<string> { "Observe", "Act", "Reflect", "Communicate", "Adapt" }
        };
    }

    private string SelectActionBasedOnTraits(AutonomousAgent agent, List<string> actions, AstrologicalWeatherReport weather)
    {
        // Weight actions based on personality traits and current cosmic conditions
        var actionWeights = new Dictionary<string, double>();
        
        foreach (var action in actions)
        {
            var weight = CalculateActionWeight(agent, action, weather);
            actionWeights[action] = weight;
        }
        
        // Select action based on weighted random selection
        return SelectWeightedRandom(actionWeights);
    }

    private double CalculateActionWeight(AutonomousAgent agent, string action, AstrologicalWeatherReport weather)
    {
        var weight = 1.0;
        
        // Personality influences
        if (action.Contains("immediate") && agent.PersonalityTraits.Impulsiveness > 70) weight *= 2.0;
        if (action.Contains("analyze") && agent.PersonalityTraits.LogicalThinking > 70) weight *= 2.0;
        if (action.Contains("communicate") && agent.PersonalityTraits.SocialOrientation > 70) weight *= 2.0;
        
        // Astrological influences
        var elementBonus = agent.BaseCharacter.BodyTraits.Element.ToString() == weather.DominantElement.ToString() ? 1.5 : 1.0;
        weight *= elementBonus;
        
        // Energy level influences
        if (agent.Energy < 30 && action.Contains("Act")) weight *= 0.5;
        if (agent.Energy > 80 && action.Contains("immediate")) weight *= 1.5;
        
        return weight;
    }

    private string SelectWeightedRandom(Dictionary<string, double> weights)
    {
        var totalWeight = weights.Values.Sum();
        var randomValue = _random.NextDouble() * totalWeight;
        
        foreach (var kvp in weights)
        {
            randomValue -= kvp.Value;
            if (randomValue <= 0) return kvp.Key;
        }
        
        return weights.Keys.First();
    }

    private string GenerateDecisionReasoning(AutonomousAgent agent, string action)
    {
        var reasoning = $"As a {agent.BaseCharacter.BodyTraits.ZodiacSign} with {agent.BaseCharacter.HeartTraits.DragonEnergy} energy, ";
        
        reasoning += action switch
        {
            var a when a.Contains("immediate") => "I feel compelled to act quickly on my instincts.",
            var a when a.Contains("analyze") => "I need to understand the situation fully before proceeding.",
            var a when a.Contains("communicate") => "connecting with others feels like the right approach.",
            var a when a.Contains("reflect") => "I should look inward for the answer.",
            _ => "this action aligns with my natural tendencies."
        };
        
        return reasoning;
    }

    private int CalculateConfidence(AutonomousAgent agent, string action)
    {
        // Confidence based on relevant skills and personality alignment
        var relevantSkill = action switch
        {
            var a when a.Contains("communicate") => agent.Skills[SkillType.Communication],
            var a when a.Contains("analyze") => agent.Skills[SkillType.ProblemSolving],
            var a when a.Contains("immediate") => agent.Skills[SkillType.Leadership],
            _ => agent.Skills[SkillType.Adaptability]
        };
        
        return Math.Max(20, Math.Min(95, relevantSkill + _random.Next(-10, 11)));
    }

    private List<string> GetInfluencingFactors(AutonomousAgent agent, AstrologicalWeatherReport weather)
    {
        return new List<string>
        {
            $"Current energy level: {agent.Energy}%",
            $"Dominant element alignment: {weather.DominantElement}",
            $"Moon phase influence: {weather.MoonPhase}",
            $"Primary personality trait: {GetPrimaryTrait(agent)}",
            $"Stress response pattern: {agent.PersonalityTraits.StressResponse}"
        };
    }

    private string GetPrimaryTrait(AutonomousAgent agent)
    {
        var traits = new Dictionary<string, int>
        {
            ["Impulsive"] = agent.PersonalityTraits.Impulsiveness,
            ["Patient"] = agent.PersonalityTraits.Patience,
            ["Social"] = agent.PersonalityTraits.SocialOrientation,
            ["Logical"] = agent.PersonalityTraits.LogicalThinking,
            ["Intuitive"] = agent.PersonalityTraits.IntuitiveThinking
        };
        
        return traits.OrderByDescending(t => t.Value).First().Key;
    }

    private DecisionOutcome ProcessDecision(AutonomousAgent agent, AgentDecision decision)
    {
        var success = CalculateDecisionSuccess(agent, decision);
        var skillGain = CalculateSkillGain(decision, success);
        var energyCost = CalculateEnergyCost(decision);
        
        return new DecisionOutcome
        {
            Success = success,
            SkillGains = skillGain,
            EnergyCost = energyCost,
            ConsequenceDescription = GenerateConsequenceDescription(decision, success),
            LessonsLearned = GenerateLessons(decision, success)
        };
    }

    private bool CalculateDecisionSuccess(AutonomousAgent agent, AgentDecision decision)
    {
        var baseSuccessRate = decision.ConfidenceLevel;
        var skillBonus = GetRelevantSkillBonus(agent, decision.Action);
        var randomFactor = _random.Next(-20, 21);
        
        var totalScore = baseSuccessRate + skillBonus + randomFactor;
        return totalScore > 50;
    }

    private Dictionary<SkillType, int> CalculateSkillGain(AgentDecision decision, bool success)
    {
        var gains = new Dictionary<SkillType, int>();
        var baseGain = success ? 2 : 1; // Learn from both success and failure
        
        // Determine which skills were exercised
        var exercisedSkills = decision.Action switch
        {
            var a when a.Contains("communicate") => new[] { SkillType.Communication, SkillType.SocialSkills },
            var a when a.Contains("analyze") => new[] { SkillType.ProblemSolving },
            var a when a.Contains("immediate") => new[] { SkillType.Leadership },
            _ => new[] { SkillType.Adaptability }
        };
        
        foreach (var skill in exercisedSkills)
        {
            gains[skill] = baseGain;
        }
        
        return gains;
    }

    private int CalculateEnergyCost(AgentDecision decision)
    {
        return decision.Action switch
        {
            var a when a.Contains("immediate") => 15,
            var a when a.Contains("analyze") => 10,
            var a when a.Contains("communicate") => 8,
            var a when a.Contains("reflect") => 5,
            _ => 7
        };
    }

    private string GenerateConsequenceDescription(AgentDecision decision, bool success)
    {
        var outcome = success ? "succeeds" : "encounters challenges";
        return $"The agent's decision to '{decision.Action}' {outcome}, leading to new insights about their capabilities.";
    }

    private List<string> GenerateLessons(AgentDecision decision, bool success)
    {
        var lessons = new List<string>();
        
        if (success)
        {
            lessons.Add($"Successfully applying '{decision.Action}' builds confidence in this approach.");
        }
        else
        {
            lessons.Add($"The challenge with '{decision.Action}' reveals areas for growth.");
        }
        
        lessons.Add("Each decision shapes future behavior patterns.");
        return lessons;
    }

    private int GetRelevantSkillBonus(AutonomousAgent agent, string action)
    {
        var relevantSkill = action switch
        {
            var a when a.Contains("communicate") => SkillType.Communication,
            var a when a.Contains("analyze") => SkillType.ProblemSolving,
            var a when a.Contains("immediate") => SkillType.Leadership,
            _ => SkillType.Adaptability
        };
        
        return (agent.Skills[relevantSkill] - 50) / 5; // Convert to bonus/penalty
    }

    private void UpdateAgentState(AutonomousAgent agent, DecisionOutcome outcome)
    {
        // Update energy
        agent.Energy = Math.Max(0, Math.Min(100, agent.Energy - outcome.EnergyCost));
        
        // Update skills
        foreach (var skillGain in outcome.SkillGains)
        {
            agent.Skills[skillGain.Key] = Math.Min(100, agent.Skills[skillGain.Key] + skillGain.Value);
        }
        
        // Update experience
        agent.TotalExperience += outcome.Success ? 10 : 5;
    }

    private bool IsScenarioComplete(AutonomousAgent agent, ScenarioExperience experience)
    {
        // Scenario complete conditions
        var timeElapsed = DateTime.UtcNow - experience.StartTime;
        var hasTimeLimit = agent.CurrentScenario?.TimeLimit != null;
        var timeExpired = hasTimeLimit && timeElapsed > agent.CurrentScenario.TimeLimit;
        
        var lowEnergy = agent.Energy < 10;
        var sufficientDecisions = experience.AgentDecisions.Count >= 5;
        
        return timeExpired || lowEnergy || sufficientDecisions;
    }

    private async Task CompleteScenario(AutonomousAgent agent, ScenarioExperience experience)
    {
        agent.CurrentState = AgentState.Idle;
        agent.CurrentScenario = null;
        experience.IsActive = false;
        experience.EndTime = DateTime.UtcNow;
        experience.FinalAgentState = CloneAgentState(agent);
        
        Console.WriteLine($"[Scenario] Completed scenario for {agent.Name}. Total decisions: {experience.AgentDecisions.Count}");
        
        // Generate scenario report
        var report = GenerateScenarioReport(experience);
        Console.WriteLine($"[Report] {report}");
    }

    private AutonomousAgentState CloneAgentState(AutonomousAgent agent)
    {
        return new AutonomousAgentState
        {
            Energy = agent.Energy,
            Skills = new Dictionary<SkillType, int>(agent.Skills),
            TotalExperience = agent.TotalExperience,
            State = agent.CurrentState
        };
    }

    private string GenerateScenarioReport(ScenarioExperience experience)
    {
        var decisions = experience.AgentDecisions.Count;
        var successRate = experience.AgentDecisions.Count(d => d.ConfidenceLevel > 60) / (double)decisions * 100;
        var duration = experience.EndTime - experience.StartTime;
        
        return $"{experience.AgentName} completed {decisions} decisions with {successRate:F1}% confidence in {duration.TotalMinutes:F1} minutes";
    }

    private int CalculateGuidanceCompliance(AutonomousAgent agent, string guidance, GuidanceType type)
    {
        var baseCompliance = agent.GuidanceReceptivity;
        
        // Adjust based on guidance type and agent personality
        var adjustment = type switch
        {
            GuidanceType.Suggestion => 0,
            GuidanceType.Warning => 20,
            GuidanceType.Encouragement => 10,
            GuidanceType.Redirection => -10,
            _ => 0
        };
        
        // Agent's current energy affects compliance
        var energyFactor = agent.Energy < 30 ? -20 : agent.Energy > 80 ? 10 : 0;
        
        return Math.Max(10, Math.Min(90, baseCompliance + adjustment + energyFactor));
    }

    private string GenerateAgentResponse(AutonomousAgent agent, string guidance, int compliance)
    {
        var responseStyle = agent.PersonalityTraits.SocialOrientation > 70 ? "diplomatic" : "direct";
        
        if (compliance > 70)
        {
            return responseStyle == "diplomatic" 
                ? $"I appreciate the guidance and will consider it carefully."
                : $"That makes sense, I'll adjust my approach.";
        }
        else if (compliance > 40)
        {
            return responseStyle == "diplomatic"
                ? $"I understand your perspective, though I have some reservations."
                : $"I hear you, but I need to trust my instincts here.";
        }
        else
        {
            return responseStyle == "diplomatic"
                ? $"Thank you for the input, but I feel compelled to follow my own path."
                : $"I respectfully disagree with that approach.";
        }
    }

    public List<AutonomousAgent> GetActiveAgents() => _activeAgents.Values.ToList();

    #endregion
}

#region Data Models

public class AutonomousAgent
{
    public string Id { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public AstrologicalCharacter BaseCharacter { get; set; } = new();
    public AgentState CurrentState { get; set; }
    public LifeScenario? CurrentScenario { get; set; }
    public int Energy { get; set; } = 100;
    public Dictionary<SkillType, int> Skills { get; set; } = new();
    public PersonalityProfile PersonalityTraits { get; set; } = new();
    public DecisionMakingProfile DecisionMaking { get; set; } = new();
    public int Autonomy { get; set; } = 100;
    public int GuidanceReceptivity { get; set; } = 50;
    public int TotalExperience { get; set; } = 0;
    public DateTime CreatedAt { get; set; }
}

public class PersonalityProfile
{
    public int Impulsiveness { get; set; }
    public int Patience { get; set; }
    public int SocialOrientation { get; set; }
    public int EmotionalDepth { get; set; }
    public int LogicalThinking { get; set; }
    public int IntuitiveThinking { get; set; }
    public int RiskTolerance { get; set; }
    public StressResponseType StressResponse { get; set; }
}

public class DecisionMakingProfile
{
    public DecisionStyle PrimaryDecisionStyle { get; set; }
    public DecisionStyle SecondaryDecisionStyle { get; set; }
    public int DecisionSpeed { get; set; }
    public int RiskAssessment { get; set; }
    public List<string> ValueSystem { get; set; } = new();
}

public class LifeScenario
{
    public string Id { get; set; } = string.Empty;
    public string Title { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public string Category { get; set; } = string.Empty; // Work, Relationships, Personal, Crisis
    public int DifficultyLevel { get; set; }
    public List<SkillType> TargetSkills { get; set; } = new();
    public TimeSpan? TimeLimit { get; set; }
    public List<string> SuccessMetrics { get; set; } = new();
    public List<string> Challenges { get; set; } = new();
    public Dictionary<string, string> ContextualFactors { get; set; } = new();
}

public class ScenarioExperience
{
    public string AgentId { get; set; } = string.Empty;
    public string AgentName { get; set; } = string.Empty;
    public LifeScenario Scenario { get; set; } = new();
    public DateTime StartTime { get; set; }
    public DateTime EndTime { get; set; }
    public AutonomousAgentState InitialAgentState { get; set; } = new();
    public AutonomousAgentState FinalAgentState { get; set; } = new();
    public List<GuidanceAction> GuidanceHistory { get; set; } = new();
    public List<AgentDecision> AgentDecisions { get; set; } = new();
    public List<SkillTest> SkillTests { get; set; } = new();
    public bool IsActive { get; set; }
}

public class AgentDecision
{
    public DateTime Timestamp { get; set; }
    public string Action { get; set; } = string.Empty;
    public string Reasoning { get; set; } = string.Empty;
    public int ConfidenceLevel { get; set; }
    public List<string> InfluencingFactors { get; set; } = new();
}

public class GuidanceAction
{
    public DateTime Timestamp { get; set; }
    public string GuidanceText { get; set; } = string.Empty;
    public GuidanceType Type { get; set; }
    public int ComplianceLevel { get; set; }
    public string AgentResponse { get; set; } = string.Empty;
}

public class GuidanceResult
{
    public bool Success { get; set; }
    public string Message { get; set; } = string.Empty;
    public int ComplianceLevel { get; set; }
}

public class DecisionOutcome
{
    public bool Success { get; set; }
    public Dictionary<SkillType, int> SkillGains { get; set; } = new();
    public int EnergyCost { get; set; }
    public string ConsequenceDescription { get; set; } = string.Empty;
    public List<string> LessonsLearned { get; set; } = new();
}

public class AutonomousAgentState
{
    public int Energy { get; set; }
    public Dictionary<SkillType, int> Skills { get; set; } = new();
    public int TotalExperience { get; set; }
    public AgentState State { get; set; }
}

public class SkillTest
{
    public SkillType Skill { get; set; }
    public string Challenge { get; set; } = string.Empty;
    public int InitialLevel { get; set; }
    public int FinalLevel { get; set; }
    public bool Passed { get; set; }
}

public enum AgentState
{
    Idle, InScenario, Resting, Learning
}

public enum SkillType
{
    Communication, Leadership, Creativity, EmotionalIntelligence,
    ProblemSolving, Resilience, Adaptability, SocialSkills, Empathy
}

public enum StressResponseType
{
    Fight, Flight, Freeze, Endure, Analyze, Adapt
}

public enum DecisionStyle
{
    Analytical, Intuitive, Emotional, Social
}

public enum GuidanceType
{
    Suggestion, Warning, Encouragement, Redirection
}

#endregion