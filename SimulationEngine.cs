using FSO.SimAntics.Interface;
using FSO.SimAntics.Interface.Models;

namespace ModularSimWorld.Components;

/// <summary>
/// Core simulation engine using FSO.SimAntics.Interface
/// </summary>
public class SimulationEngine
{
    private readonly List<SimObject> _objects;
    private readonly List<SimAction> _actionQueue;
    private readonly Dictionary<string, GameObject> _gameObjects;
    private bool _isRunning;
    private int _tickCount;
    
    public SimulationEngine()
    {
        _objects = new List<SimObject>();
        _actionQueue = new List<SimAction>();
        _gameObjects = new Dictionary<string, GameObject>();
        _isRunning = false;
        _tickCount = 0;
        Console.WriteLine("[Simulation] SimulationEngine initialized with FSO.SimAntics.Interface");
    }
    
    /// <summary>
    /// Starts the simulation
    /// </summary>
    public void Start()
    {
        if (!_isRunning)
        {
            _isRunning = true;
            Console.WriteLine("[Simulation] Simulation started");
        }
    }
    
    /// <summary>
    /// Stops the simulation
    /// </summary>
    public void Stop()
    {
        if (_isRunning)
        {
            _isRunning = false;
            Console.WriteLine("[Simulation] Simulation stopped");
        }
    }
    
    /// <summary>
    /// Processes one simulation tick
    /// </summary>
    public void Tick()
    {
        if (!_isRunning) return;
        
        _tickCount++;
        
        // Process pending actions
        ProcessActions();
        
        // Update simulation objects
        UpdateObjects();
        
        if (_tickCount % 100 == 0) // Log every 100 ticks
        {
            Console.WriteLine($"[Simulation] Tick {_tickCount} - Objects: {_objects.Count}, Actions: {_actionQueue.Count}");
        }
    }
    
    /// <summary>
    /// Adds a simulation object with FSO GameObject integration
    /// </summary>
    public void AddObject(SimObject obj)
    {
        _objects.Add(obj);
        
        // Create a FSO GameObject (simulated since we don't have real OBDJ/Resource files)
        // This demonstrates integration with FSO.SimAntics.Interface.Models
        var gameObject = new GameObject(
            GUID: (ulong)obj.Id.GetHashCode(),
            OBJ: null!, // Would be a real IOBDJ in actual implementation
            Resource: null! // Would be a real IGameIffResource in actual implementation
        );
        
        _gameObjects[obj.Id] = gameObject;
        Console.WriteLine($"[Simulation] Added object: {obj.Name} (ID: {obj.Id}) with FSO GameObject GUID: {gameObject.GUID}");
    }
    
    /// <summary>
    /// Queues an action for execution
    /// </summary>
    public void QueueAction(SimAction action)
    {
        _actionQueue.Add(action);
        Console.WriteLine($"[Simulation] Queued action: {action.Type} for object {action.TargetObjectId}");
    }
    
    private void ProcessActions()
    {
        var actionsToRemove = new List<SimAction>();
        
        foreach (var action in _actionQueue)
        {
            // Simulate action processing
            if (ProcessAction(action))
            {
                actionsToRemove.Add(action);
            }
        }
        
        foreach (var action in actionsToRemove)
        {
            _actionQueue.Remove(action);
        }
    }
    
    private bool ProcessAction(SimAction action)
    {
        // Simulate action execution time
        action.ProcessingTicks++;
        
        if (action.ProcessingTicks >= action.RequiredTicks)
        {
            Console.WriteLine($"[Simulation] Completed action: {action.Type} for object {action.TargetObjectId}");
            return true; // Action completed
        }
        
        return false; // Action still processing
    }
    
    private void UpdateObjects()
    {
        foreach (var obj in _objects)
        {
            // Simulate object state updates
            obj.LastUpdateTick = _tickCount;
        }
    }
    
    /// <summary>
    /// Gets FSO GameObject for a simulation object
    /// </summary>
    public GameObject? GetGameObject(string objectId)
    {
        return _gameObjects.TryGetValue(objectId, out var gameObject) ? gameObject : null;
    }
    
    /// <summary>
    /// Gets GameObject information for display
    /// </summary>
    public string GetGameObjectInfo(string objectId)
    {
        if (_gameObjects.TryGetValue(objectId, out var gameObject))
        {
            return $"FSO GameObject GUID: {gameObject.GUID}";
        }
        return "No FSO GameObject found";
    }

    public bool IsRunning => _isRunning;
    public int TickCount => _tickCount;
    public int ObjectCount => _objects.Count;
    public int ActionCount => _actionQueue.Count;
}

/// <summary>
/// Represents a simulation object
/// </summary>
public class SimObject
{
    public string Id { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public SimObjectType Type { get; set; }
    public Vector3D Position { get; set; }
    public int LastUpdateTick { get; set; }
}

/// <summary>
/// Represents a simulation action
/// </summary>
public class SimAction
{
    public string Id { get; set; } = string.Empty;
    public SimActionType Type { get; set; }
    public string TargetObjectId { get; set; } = string.Empty;
    public string ActorId { get; set; } = string.Empty;
    public int RequiredTicks { get; set; } = 5;
    public int ProcessingTicks { get; set; } = 0;
}

/// <summary>
/// Types of simulation objects
/// </summary>
public enum SimObjectType
{
    Furniture,
    Appliance,
    Decoration,
    Plant,
    Vehicle
}

/// <summary>
/// Types of simulation actions
/// </summary>
public enum SimActionType
{
    Move,
    Interact,
    Use,
    Examine,
    Purchase,
    Sell
}