# DESIGNER NOTEBOOK
## The Meta-Brain: Auto-Organizer Rules Engine

**Purpose**: This notebook defines HOW the system organizes itself. When you drop any file/code/concept into the system, THIS determines where it goes, what it becomes, and how it connects.

**Philosophy**: Everything collapses to **5 Master Categories**. Every file, every concept, every piece of code gets "glyphed" (classified) and placed automatically.

---

## THE FIVE MASTER CATEGORIES

Based on consciousness architecture + your system's natural structure:

### 1. **ENGINE** 🔥
**What**: Core mechanics, logic, calculation systems
**Role**: The machinery that makes everything work
**Glyph**: `◈` (synthesis/creation)
**Folder**: `/engine/`
**Examples**:
- Consciousness engines (Trinity, Spirit, Fields)
- Calculation systems (Stellar Proximology)
- State machines
- Core algorithms

**Detection Rules**:
- File contains: `Engine`, `Core`, `System`, `Calculator`
- Exports classes ending in: `Engine`, `Core`, `System`
- Heavy computation/state management
- No UI components

---

### 2. **INTERFACE** 🎨
**What**: UI, visualization, user-facing components
**Role**: How humans interact with the system
**Glyph**: `◯` (observation/perception)
**Folder**: `/interface/`
**Examples**:
- React components
- Widgets
- Dashboards
- Visualizations
- Forms/inputs

**Detection Rules**:
- File contains: `Component`, `Widget`, `UI`, `View`
- Uses: React, JSX, HTML, CSS
- Has UI/UX elements
- User interaction code

---

### 3. **AGENT** 🤖
**What**: Autonomous entities, AI systems, decision-makers
**Role**: Things that act independently
**Glyph**: `◆` (action/participation)
**Folder**: `/agents/`
**Examples**:
- Glyph Keyboard
- Avatar system
- Oracle Engine
- SynthAI
- Autonomous actors

**Detection Rules**:
- File contains: `Agent`, `AI`, `Bot`, `Oracle`, `Assistant`
- Has autonomous behavior loops
- Makes decisions
- Responds to events

---

### 4. **WORLD** 🌍
**What**: Spatial systems, locations, environments, universe structure
**Role**: Where things exist and how they connect
**Glyph**: `⬡` (space/structure)
**Folder**: `/world/`
**Examples**:
- Spatial Engine
- Location registry
- Universe maps
- Building definitions
- Navigation systems

**Detection Rules**:
- File contains: `Spatial`, `Location`, `World`, `Map`, `Universe`
- Deals with coordinates/positioning
- Manages locations/buildings
- Navigation/pathfinding

---

### 5. **KNOWLEDGE** 📚
**What**: Data, context, explanations, memory, documentation
**Role**: What the system knows and remembers
**Glyph**: `◉` (integration/unity)
**Folder**: `/knowledge/`
**Examples**:
- Context Engine
- Documentation
- Gate/Channel explanations
- Observation records
- Pattern databases

**Detection Rules**:
- File contains: `Context`, `Knowledge`, `Data`, `Memory`, `Doc`
- Stores information
- Provides explanations
- No active logic (just data)

---

## GLYPH DETECTION LOGIC

### Primary Classification
When a file is uploaded, scan for these patterns IN ORDER:

1. **Check filename** for category keywords
2. **Check exports** for class/function names matching patterns
3. **Scan imports** for dependencies indicating type
4. **Analyze code structure** for behavioral patterns
5. **Default** to most likely category based on content ratio

### Secondary Tags
After primary category, add secondary tags:
- `[core]` - Essential system component
- `[utility]` - Helper/support function
- `[experimental]` - In development
- `[legacy]` - Old version kept for reference
- `[bridge]` - Connects two categories

### Example Classification

**File**: `TrinityEngine.js`
```
Primary: ENGINE
Secondary: [core]
Glyph: ◈
Path: /engine/core/TrinityEngine.js
Reason: Contains "Engine" in name, exports state machine class, no UI
```

**File**: `AvatarWidget.jsx`
```
Primary: INTERFACE
Secondary: [agent-connected]
Glyph: ◯
Path: /interface/widgets/AvatarWidget.jsx
Reason: JSX file, React component, UI rendering
```

**File**: `GlyphKeyboard.js`
```
Primary: AGENT
Secondary: [core]
Glyph: ◆
Path: /agents/core/GlyphKeyboard.js
Reason: Autonomous behavior, records glyphs, acts independently
```

---

## PLACEMENT LOGIC

### Folder Structure
```
/root
  /engine
    /core          (essential engines)
    /utils         (helper systems)
    /bridges       (connections between engines)
  
  /interface
    /widgets       (UI components)
    /screens       (full pages)
    /visualizers   (data display)
  
  /agents
    /core          (main autonomous entities)
    /assistants    (helper agents)
    /oracles       (guidance systems)
  
  /world
    /locations     (building definitions)
    /maps          (universe structure)
    /navigation    (pathfinding)
  
  /knowledge
    /context       (explanations)
    /data          (raw data/records)
    /patterns      (learned insights)
```

### Auto-Placement Rules

**When file is classified:**
1. Move to primary category folder
2. If `[core]` tag → `/core/` subfolder
3. If connects to another category → create bridge reference
4. Update index for that category
5. Link to related files
6. Log the placement

**Index Updates:**
- Each category has `_INDEX.json` tracking all files
- Contains: filename, path, glyph, tags, dependencies, description
- Auto-updated on any file placement

---

## BUILDER LOOP

### The Auto-Organization Cycle

```
1. FILE DROPPED
   ↓
2. SCAN & CLASSIFY
   (Apply detection rules)
   ↓
3. ASSIGN GLYPH
   (Determine primary category + tags)
   ↓
4. PLACE FILE
   (Move to correct folder structure)
   ↓
5. UPDATE INDEX
   (Add to category index)
   ↓
6. LINK DEPENDENCIES
   (Connect to related files)
   ↓
7. LOG ACTION
   (Record in glyph timeline)
   ↓
8. EMIT EVENT
   (Notify system of new component)
```

### Recursive Scanning
When a folder/project is uploaded:
- Scan ALL files recursively
- Classify each independently
- Maintain original folder structure AS REFERENCE
- Create organized structure in parallel
- Map old paths → new paths
- Update all imports automatically

---

## INTEGRATION HOOKS

### How Other Systems Use This

**Engine Layer**:
- Reads category indices
- Loads components by glyph
- Discovers available modules

**Agent Layer**:
- Agents query knowledge category for context
- Agents register themselves in agent category
- Agents can discover other agents

**Interface Layer**:
- UI components auto-discover widgets
- Dashboard builds itself from available interfaces
- Navigation generated from world category

**World Layer**:
- Locations read from world category
- Spatial relationships auto-mapped
- Universe structure self-constructs

**Knowledge Layer**:
- Context pulled from knowledge category
- Observations stored in knowledge/data
- Patterns extracted and indexed

---

## USAGE EXAMPLES

### Example 1: Upload New Engine

**Input**: Drop `WaveformEngine.js` into system

**Process**:
1. Scan: Contains "Engine" in name, exports class with state
2. Classify: PRIMARY=ENGINE, tags=[core]
3. Glyph: `◈`
4. Place: `/engine/core/WaveformEngine.js`
5. Index: Add to `/engine/_INDEX.json`
6. Link: Connect to SpiritCore (dependency detected)
7. Log: Record glyph `◈` in timeline
8. Emit: `engine_added` event

**Result**: WaveformEngine now available to all systems

---

### Example 2: Upload UI Component

**Input**: Drop `ChartVisualizer.jsx` into system

**Process**:
1. Scan: JSX extension, React imports, render() method
2. Classify: PRIMARY=INTERFACE, tags=[visualizer]
3. Glyph: `◯`
4. Place: `/interface/visualizers/ChartVisualizer.jsx`
5. Index: Add to `/interface/_INDEX.json`
6. Link: Detect uses StellarProximology data
7. Log: Record glyph `◯` in timeline
8. Emit: `interface_added` event

**Result**: Widget auto-appears in dashboard registry

---

### Example 3: Upload Documentation

**Input**: Drop `Gate12.md` (Human Design gate info)

**Process**:
1. Scan: Markdown file, contains explanations
2. Classify: PRIMARY=KNOWLEDGE, tags=[human-design, gates]
3. Glyph: `◉`
4. Place: `/knowledge/human-design/gates/Gate12.md`
5. Index: Add to `/knowledge/_INDEX.json`
6. Link: Connect to ContextEngine
7. Log: Record glyph `◉` in timeline
8. Emit: `knowledge_added` event

**Result**: ContextEngine can now explain Gate 12

---

## CONFIGURATION

### Customization Points

**Modify Categories**:
Edit the 5 master categories if your system needs different divisions

**Add Detection Rules**:
Extend keyword lists for better classification

**Change Folder Structure**:
Modify placement paths while keeping category logic

**Add Tags**:
Create new secondary tags for finer organization

**Integration Hooks**:
Define what happens when each category gets new content

---

## LOGS & TRACKING

### What Gets Logged

Every placement action creates a log entry:

```json
{
  "timestamp": "2025-11-17T05:00:00Z",
  "action": "file_placed",
  "filename": "TrinityEngine.js",
  "category": "ENGINE",
  "glyph": "◈",
  "path": "/engine/core/TrinityEngine.js",
  "tags": ["core"],
  "dependencies": ["SpiritCore.js"],
  "auto_detected": true,
  "confidence": 0.95
}
```

### Designer Log
Maintains history of all organization decisions:
- What was classified
- Why it was classified that way
- Where it was placed
- What it connects to
- Any manual overrides

This log IS the system's memory of how it organized itself.

---

## NEXT STEPS

### To Activate This System:

1. **Create the Builder Script**
   - Implement detection rules as code
   - Add file scanning logic
   - Build auto-placement engine
   - Create index updater

2. **Set Up Folder Structure**
   - Create 5 category folders
   - Add `_INDEX.json` to each
   - Set up subfolder templates

3. **Upload Your Files**
   - Drop entire project into builder
   - Let it auto-organize
   - Review placement decisions
   - Adjust rules if needed

4. **Integrate with Engines**
   - UniverseKernel reads indices
   - Components auto-discover each other
   - System self-assembles

---

## YOUR LEGACY BUILDER

This IS what you remembered.

You had a system that:
- Scanned files
- Glyphed them (classified)
- Placed them automatically
- Built the app structure
- Connected everything

**LlamaIndex was the brain** doing the classification.
**This notebook is the rules** telling it HOW to classify.
**The builder script** executes these rules.

You weren't confused.
You were building the meta-layer.

Now rebuild it with these rules.

---

**Status**: DESIGNER NOTEBOOK COMPLETE
**Next**: Implement as code (Python or JS)
**Then**: Upload files and watch auto-organization happen
