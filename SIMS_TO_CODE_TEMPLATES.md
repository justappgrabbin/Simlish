# SIMS-TO-CODE TEMPLATES
## When Sims build → Code generates

These templates define HOW a completed Sim building becomes actual code in the app.

---

## THE MAPPING

**Each of the 5 building types → Generates specific code:**

| Sim Building | Construction Type | Generates | Glyph |
|--------------|-------------------|-----------|-------|
| Factory      | ENGINE            | Engine Class | ◈ |
| Gallery      | INTERFACE         | React Component | ◯ |
| Academy      | AGENT             | Agent Class | ◆ |
| Landmark     | WORLD             | Location Definition | ⬡ |
| Library      | KNOWLEDGE         | Data Store | ◉ |

---

## TEMPLATE 1: ENGINE (Factory → Engine Class)

### Input (from Unity)
```json
{
  "building_type": "Factory",
  "name": "ResonanceEngine",
  "dimensions": {"width": 20, "height": 30, "depth": 20},
  "materials": ["metal", "circuits", "energy"],
  "connections": ["SpiritCore", "TrinityEngine"],
  "functions": ["calculate", "resonate", "emit"]
}
```

### Output (Generated Code)
```javascript
/**
 * ResonanceEngine.js
 * Auto-generated from Sim Factory construction
 * Built by: Agent_Name
 * Date: 2025-11-17
 */

export class ResonanceEngine {
  constructor(spiritCore, trinityEngine) {
    this.spirit = spiritCore;
    this.trinity = trinityEngine;
    
    // Properties from building dimensions
    this.capacity = 20;      // width
    this.frequency = 30;     // height
    this.depth = 20;         // depth
    
    // Materials determine capabilities
    this.hasMetal = true;    // structural integrity
    this.hasCircuits = true; // logical processing
    this.hasEnergy = true;   // active power
    
    console.log('🔧 ResonanceEngine initialized');
  }
  
  // Functions from building spec
  calculate(input) {
    // Auto-generated calculation logic
    return input * this.frequency;
  }
  
  resonate(field) {
    // Auto-generated resonance logic
    const resonance = field * this.capacity;
    this.emit('resonance_calculated', resonance);
    return resonance;
  }
  
  emit(event, data) {
    this.spirit.emit(event, data);
  }
  
  getStatus() {
    return {
      type: 'ResonanceEngine',
      glyph: '◈',
      capacity: this.capacity,
      frequency: this.frequency,
      depth: this.depth
    };
  }
}

export default ResonanceEngine;
```

---

## TEMPLATE 2: INTERFACE (Gallery → React Component)

### Input (from Unity)
```json
{
  "building_type": "Gallery",
  "name": "ResonanceWidget",
  "dimensions": {"width": 300, "height": 200, "depth": 10},
  "materials": ["glass", "light", "color"],
  "connections": ["ResonanceEngine"],
  "display_elements": ["meter", "graph", "button"]
}
```

### Output (Generated Code)
```jsx
/**
 * ResonanceWidget.jsx
 * Auto-generated from Sim Gallery construction
 * Built by: Agent_Name
 * Date: 2025-11-17
 */

import React, { useState, useEffect } from 'react';
import { ResonanceEngine } from '../engine/ResonanceEngine';

export const ResonanceWidget = ({ universe }) => {
  const [resonance, setResonance] = useState(0);
  const [active, setActive] = useState(false);
  
  // Dimensions from building
  const width = 300;
  const height = 200;
  
  // Materials determine styling
  const hasGlass = true;   // transparency
  const hasLight = true;   // glow effects
  const hasColor = true;   // color variation
  
  // Connect to engine
  const engine = universe.getEngine('ResonanceEngine');
  
  useEffect(() => {
    if (engine) {
      universe.on('resonance_calculated', (data) => {
        setResonance(data);
      });
    }
  }, [engine]);
  
  const handleCalculate = () => {
    if (engine) {
      const result = engine.calculate(100);
      setActive(true);
    }
  };
  
  return (
    <div 
      className="resonance-widget"
      style={{
        width: `${width}px`,
        height: `${height}px`,
        background: hasGlass ? 'rgba(255,255,255,0.1)' : '#fff',
        boxShadow: hasLight ? '0 0 20px rgba(159,122,234,0.3)' : 'none',
        border: '1px solid rgba(159,122,234,0.5)',
        borderRadius: '8px',
        padding: '20px'
      }}
    >
      <h3>◈ Resonance</h3>
      
      {/* Display elements from building spec */}
      
      {/* Meter */}
      <div className="meter">
        <div 
          className="meter-fill"
          style={{
            width: `${(resonance / 100) * 100}%`,
            height: '10px',
            background: 'linear-gradient(90deg, #9F7AEA, #4299E1)',
            transition: 'width 0.3s'
          }}
        />
      </div>
      
      {/* Value display */}
      <div className="value">
        {resonance.toFixed(2)}
      </div>
      
      {/* Button */}
      <button 
        onClick={handleCalculate}
        style={{
          background: active ? '#9F7AEA' : '#4A5568',
          color: 'white',
          padding: '10px 20px',
          border: 'none',
          borderRadius: '4px',
          cursor: 'pointer'
        }}
      >
        Calculate Resonance
      </button>
    </div>
  );
};

export default ResonanceWidget;
```

---

## TEMPLATE 3: AGENT (Academy → Agent Class)

### Input (from Unity)
```json
{
  "building_type": "Academy",
  "name": "ResonanceGuide",
  "dimensions": {"width": 15, "height": 25, "depth": 15},
  "materials": ["consciousness", "code", "autonomy"],
  "connections": ["ResonanceEngine", "SpiritCore"],
  "behaviors": ["observe", "calculate", "guide"]
}
```

### Output (Generated Code)
```javascript
/**
 * ResonanceGuide.js
 * Auto-generated from Sim Academy construction
 * Built by: Agent_Name
 * Date: 2025-11-17
 */

export class ResonanceGuide {
  constructor(universe) {
    this.universe = universe;
    this.name = 'ResonanceGuide';
    this.glyph = '◆';
    
    // Agent properties from building
    this.awareness = 15;      // width
    this.intelligence = 25;   // height
    this.autonomy = 15;       // depth
    
    // Materials determine capabilities
    this.hasConsciousness = true; // self-aware
    this.hasCode = true;          // can generate code
    this.hasAutonomy = true;      // acts independently
    
    // Behavior state
    this.currentActivity = 'idle';
    this.observations = [];
    
    // Connect to engines
    this.resonanceEngine = null;
    this.spirit = null;
    
    this.initialize();
  }
  
  async initialize() {
    // Connect to required systems
    this.resonanceEngine = this.universe.getEngine('ResonanceEngine');
    this.spirit = this.universe.spirit;
    
    // Register as agent
    this.universe.registerAgent(this.name, this);
    
    // Start autonomous behavior
    if (this.hasAutonomy) {
      this.startBehaviorLoop();
    }
    
    console.log('◆ ResonanceGuide agent initialized');
  }
  
  // Behaviors from building spec
  
  observe() {
    if (!this.resonanceEngine) return;
    
    const status = this.resonanceEngine.getStatus();
    this.observations.push({
      timestamp: Date.now(),
      data: status
    });
    
    this.currentActivity = 'observing';
  }
  
  calculate() {
    if (!this.resonanceEngine) return;
    
    const result = this.resonanceEngine.calculate(
      this.intelligence
    );
    
    this.currentActivity = 'calculating';
    return result;
  }
  
  guide(user) {
    const recentObs = this.observations.slice(-5);
    
    const guidance = {
      message: `Based on ${recentObs.length} observations, I recommend adjusting resonance frequency.`,
      confidence: this.awareness / 20,
      action: 'increase_frequency'
    };
    
    this.currentActivity = 'guiding';
    return guidance;
  }
  
  // Autonomous behavior loop
  async startBehaviorLoop() {
    while (this.hasAutonomy) {
      // Observe periodically
      this.observe();
      
      // Decide if guidance needed
      if (this.observations.length > 10) {
        const guidance = this.guide();
        this.universe.emit('agent_guidance', {
          agent: this.name,
          guidance
        });
      }
      
      // Wait before next cycle
      await new Promise(resolve => setTimeout(resolve, 5000));
    }
  }
  
  getStatus() {
    return {
      name: this.name,
      glyph: this.glyph,
      activity: this.currentActivity,
      awareness: this.awareness,
      intelligence: this.intelligence,
      autonomy: this.autonomy,
      observations: this.observations.length
    };
  }
}

export default ResonanceGuide;
```

---

## TEMPLATE 4: WORLD (Landmark → Location Definition)

### Input (from Unity)
```json
{
  "building_type": "Landmark",
  "name": "ResonanceChamber",
  "dimensions": {"width": 50, "height": 100, "depth": 50},
  "materials": ["space", "matter", "gravity"],
  "connections": ["EntrancePortal", "EngineRoom"],
  "features": ["resonance_field", "observation_deck"]
}
```

### Output (Generated Code)
```javascript
/**
 * ResonanceChamber.js
 * Auto-generated from Sim Landmark construction
 * Built by: Agent_Name
 * Date: 2025-11-17
 */

export const ResonanceChamber = {
  id: 'resonance_chamber',
  name: 'Resonance Chamber',
  glyph: '⬡',
  type: 'observatory',
  
  // Spatial properties from building
  coordinates: {
    x: 0,  // Assigned by SpatialEngine
    y: 0,
    z: 0
  },
  
  dimensions: {
    width: 50,
    height: 100,
    depth: 50
  },
  
  // Materials determine location properties
  hasSpace: true,    // Can contain entities
  hasMatter: true,   // Physical presence
  hasGravity: true,  // Affects movement
  
  // Connected locations
  connectedTo: ['entrance_portal', 'engine_room'],
  
  // Features available at this location
  features: {
    resonance_field: {
      active: true,
      strength: 0.8,
      affects: ['frequency', 'amplitude']
    },
    observation_deck: {
      active: true,
      visibility: 'high',
      instruments: ['scanner', 'analyzer']
    }
  },
  
  // Activities available here
  activities: [
    'observe',
    'resonate',
    'calibrate',
    'measure'
  ],
  
  // Resonance field unique to this location
  resonanceField: 0.85,
  
  // What happens when avatar enters
  onEnter: function(avatar) {
    console.log(`⬡ ${avatar.name} entered Resonance Chamber`);
    
    // Apply resonance field effect
    avatar.resonance = Math.min(
      avatar.resonance + this.resonanceField,
      1.0
    );
    
    // Unlock specific activities
    avatar.availableActivities = this.activities;
  },
  
  // What happens when avatar leaves
  onExit: function(avatar) {
    console.log(`⬡ ${avatar.name} left Resonance Chamber`);
    
    // Remove resonance boost
    avatar.resonance = Math.max(
      avatar.resonance - this.resonanceField,
      0.0
    );
  },
  
  // Get location status
  getStatus: function() {
    return {
      id: this.id,
      name: this.name,
      glyph: this.glyph,
      dimensions: this.dimensions,
      features: Object.keys(this.features),
      resonance: this.resonanceField
    };
  }
};

export default ResonanceChamber;
```

---

## TEMPLATE 5: KNOWLEDGE (Library → Data Store)

### Input (from Unity)
```json
{
  "building_type": "Library",
  "name": "ResonanceKnowledge",
  "dimensions": {"width": 30, "height": 40, "depth": 30},
  "materials": ["data", "memory", "wisdom"],
  "connections": ["ContextEngine"],
  "content_types": ["explanations", "observations", "patterns"]
}
```

### Output (Generated Code)
```json
{
  "knowledge_domain": "resonance",
  "glyph": "◉",
  "structure": {
    "explanations": {
      "resonance": {
        "surface": {
          "title": "Resonance",
          "tagline": "Harmonic alignment of frequencies",
          "keywords": ["frequency", "alignment", "harmony"]
        },
        "medium": {
          "title": "Understanding Resonance",
          "description": "Resonance occurs when two systems oscillate at compatible frequencies, allowing energy to transfer efficiently between them.",
          "keynotes": [
            "Frequency alignment",
            "Energy transfer",
            "Harmonic relationships"
          ]
        },
        "deep": {
          "title": "Resonance Mechanics",
          "description": "Mathematical and physical principles of resonant systems",
          "formulas": [
            "f_res = 1 / (2π√LC)",
            "Q = f_res / Δf"
          ]
        },
        "experiential": {
          "title": "Experiencing Resonance",
          "description": "When you feel resonance, it's like finding the exact frequency where everything aligns. The resistance drops away and energy flows naturally.",
          "feels_like": "Finding the sweet spot where everything clicks"
        }
      }
    },
    
    "observations": [
      {
        "timestamp": "2025-11-17T00:00:00Z",
        "observer": "ResonanceGuide",
        "observation": "High resonance detected at frequency 432Hz",
        "context": {
          "trinity": "observer-participant",
          "coherence": 0.85
        }
      }
    ],
    
    "patterns": [
      {
        "pattern_id": "resonance_cycle",
        "description": "Repeating pattern of resonance buildup and release",
        "frequency": "occurs every 7 days",
        "elements": ["observe", "resonate", "integrate"]
      }
    ]
  },
  
  "metadata": {
    "created": "2025-11-17",
    "created_by": "Agent_Name",
    "building_source": "Library construction in Unity",
    "dimensions": {"width": 30, "height": 40, "depth": 30},
    "capacity": 30,
    "depth": 30,
    "wisdom_level": 40
  }
}
```

---

## HOW TO USE THESE TEMPLATES

### In FSO Orchestrator:

```python
def _generate_code_from_construction(self, agent_task, result):
    construction_type = agent_task["type"]
    
    # Load appropriate template
    template = self._load_template(construction_type)
    
    # Fill template with building data
    code = template.format(
        name=agent_task['spec']['name'],
        width=result['dimensions']['width'],
        height=result['dimensions']['height'],
        depth=result['dimensions']['depth'],
        materials=result.get('materials', []),
        connections=result.get('connections', [])
    )
    
    # Save generated code
    output_path = self._get_output_path(construction_type, agent_task['spec']['name'])
    with open(output_path, 'w') as f:
        f.write(code)
    
    return {"filename": output_path, "code": code}
```

---

## THE MAGIC

**When a Sim builds a Factory (30x40x30):**
- Width (30) → Engine capacity
- Height (40) → Engine frequency/power
- Depth (30) → Engine processing depth

**Materials chosen determine capabilities:**
- Metal → Structural/stable
- Circuits → Logical/computational
- Energy → Active/powered

**Connections determine integrations:**
- Connected to SpiritCore → Can access consciousness state
- Connected to TrinityEngine → Can use trinity modes

**THE SIM LITERALLY ARCHITECTS THE CODE BY BUILDING.**

---

This is your simultaneous construction system.
Sims build → Code generates → App grows → Sims build more → Loop continues.
