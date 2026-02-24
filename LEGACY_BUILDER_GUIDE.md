# LEGACY BUILDER - QUICK START GUIDE

## What This Is

**Legacy Builder** = The auto-organizer that glyphs and places your files automatically

Based on your **Designer Notebook** rules:
- 5 master categories (Engine, Interface, Agent, World, Knowledge)
- Auto-detection via keywords/patterns
- Automatic folder structure creation
- Index generation
- Glyph assignment

---

## Files You Now Have

### 1. DESIGNER_NOTEBOOK.md
**Location**: `/mnt/user-data/outputs/DESIGNER_NOTEBOOK.md`

**What**: The rules/brain - defines HOW files get classified

**Contains**:
- The 5 category definitions
- Detection logic
- Placement rules
- Glyph assignments
- Folder structure
- Integration hooks

**Use**: Read this to understand the system logic

---

### 2. legacy_builder.py
**Location**: `/mnt/user-data/outputs/legacy_builder.py`

**What**: The actual builder script - EXECUTES the rules

**Does**:
- Scans files in a directory
- Classifies each file into one of 5 categories
- Assigns glyphs
- Moves files to organized folder structure
- Updates indices
- Generates placement log

**Use**: Run this to auto-organize your files

---

## How To Use

### Method 1: Command Line

```bash
# Basic usage (dry run first)
python legacy_builder.py /path/to/your/files

# It will:
# 1. Show you what it WOULD do (dry run)
# 2. Ask for confirmation
# 3. Actually move files if you say yes
```

### Method 2: Python Script

```python
from legacy_builder import LegacyBuilder
from pathlib import Path

# Initialize
builder = LegacyBuilder(root_path="./organized_output")

# Scan and organize
placements = builder.scan_directory(
    Path("./my_messy_files"),
    recursive=True,
    dry_run=False  # Set to True for testing
)

# Get report
print(builder.generate_report())

# Save log
builder.save_log("placement_log.json")
```

### Method 3: Replit Integration

```python
# In your Replit project
import os
from legacy_builder import LegacyBuilder

# Set up builder
builder = LegacyBuilder(root_path=os.getcwd())

# Auto-organize on file upload
def on_file_upload(filepath):
    builder.place_file(Path(filepath))
    print(f"✅ File organized: {filepath}")
```

---

## What Happens When You Run It

### Step-by-Step Process

**1. Initialization**
```
✅ Folder structure initialized
   /engine
     /core
     /utils
     /bridges
   /interface
     /core
     /utils
     /bridges
   ... (same for all 5 categories)
```

**2. File Scanning**
```
📂 Scanning 47 files...
```

**3. Classification & Placement**
```
✅ TrinityEngine.js → ENGINE/core/ (◈)
✅ AvatarWidget.jsx → INTERFACE/utils/ (◯)
✅ GlyphKeyboard.js → AGENT/core/ (◆)
✅ SpatialEngine.js → WORLD/core/ (⬡)
✅ ContextEngine.js → KNOWLEDGE/core/ (◉)
```

**4. Index Updates**
Each category's `_INDEX.json` gets updated with new files

**5. Report Generation**
```
============================================================
LEGACY BUILDER REPORT
============================================================

Files Placed by Category:
  ◈ ENGINE: 12 files
  ◯ INTERFACE: 8 files
  ◆ AGENT: 5 files
  ⬡ WORLD: 3 files
  ◉ KNOWLEDGE: 19 files

Total Files Processed: 47
============================================================
```

---

## Output Structure

After running, you get:

```
/organized_output
  /engine
    /core
      TrinityEngine.js
      SpiritCore.js
      FiveElementEngine.js
    /utils
      HelperFunctions.js
    /bridges
      AgentBridge.js
    _INDEX.json
  
  /interface
    /core
      Dashboard.jsx
      Widget.jsx
    /utils
      UIHelpers.js
    _INDEX.json
  
  /agents
    /core
      GlyphKeyboard.js
      Avatar.js
    _INDEX.json
  
  /world
    /core
      SpatialEngine.js
    _INDEX.json
  
  /knowledge
    /core
      ContextEngine.js
    /data
      observations.json
    _INDEX.json
  
  placement_log.json (complete record of all actions)
```

---

## Customization

### Change the 5 Categories

Edit `self.categories` in `legacy_builder.py`:

```python
self.categories = {
    "YOUR_CATEGORY": {
        "glyph": "🔥",
        "folder": "your_folder",
        "keywords": ["YourKeyword", "Another"],
        "patterns": [r".*YourPattern\.js$"],
        "description": "What this category is for"
    },
    # ... 4 more categories
}
```

### Add Detection Rules

Add more keywords/patterns to existing categories:

```python
"ENGINE": {
    "keywords": ["Engine", "Core", "System", "YourNewKeyword"],
    "patterns": [r".*Engine\.js$", r".*YourPattern\.js$"],
}
```

### Change Folder Structure

Modify `initialize_structure()` method to create different subfolders

---

## Integration With YOUR System

### Connect to UniverseKernel

```javascript
// In UniverseKernel.js
import fs from 'fs';

class UniverseKernel {
  loadFromIndices() {
    // Read category indices
    const engineIndex = JSON.parse(
      fs.readFileSync('./organized/engine/_INDEX.json')
    );
    
    // Auto-load all engine files
    engineIndex.files.forEach(file => {
      const Engine = require(file.path);
      this.registerEngine(Engine);
    });
    
    // Repeat for all 5 categories
  }
}
```

### Auto-Organize on Upload

```javascript
// When user uploads file
async function handleUpload(file) {
  // Save file temporarily
  await saveFile(file);
  
  // Run builder
  const { exec } = require('child_process');
  exec(`python legacy_builder.py ${file.path}`, (err, stdout) => {
    if (err) {
      console.error('Auto-organize failed:', err);
    } else {
      console.log('✅ File auto-organized');
      // Reload indices
      this.universeKernel.loadFromIndices();
    }
  });
}
```

---

## Advanced Features

### Dry Run Mode

Test classification without moving files:

```python
builder.scan_directory(Path("./files"), dry_run=True)
```

### Manual Classification

Override auto-detection:

```python
placement = builder.place_file(
    Path("myfile.js"),
    dry_run=False
)

# Check classification
print(f"Category: {placement['category']}")
print(f"Confidence: {placement['confidence']}")

# If wrong, manually move and update index
```

### Batch Processing

Organize multiple folders:

```python
folders = [
    Path("./old_project_1"),
    Path("./old_project_2"),
    Path("./downloads")
]

for folder in folders:
    builder.scan_directory(folder)

print(builder.generate_report())
```

---

## Troubleshooting

### "File not recognized" (low confidence)

**Problem**: Builder unsure what category file belongs to

**Solution**: 
1. Check `placement_log.json` for confidence scores
2. Add more keywords to relevant category
3. Manually move file and update index

### "Import errors after organizing"

**Problem**: File paths changed, imports broken

**Solution**:
1. Update import paths in moved files
2. Or: Use absolute imports
3. Or: Update your bundler config to know new structure

### "Want different folder structure"

**Solution**:
1. Modify `initialize_structure()` in builder
2. Or: Manually reorganize after auto-placement
3. Indices will still work

---

## This IS Your System

You said:
> "upload my legacy builder which builds the app and it would essentially glyph whatever is dropped into it and then put it in the right spots, also bringing it down to five categories"

**This is exactly that.**

- ✅ Glyphs files automatically
- ✅ Puts them in the right spots
- ✅ Brings everything down to 5 categories
- ✅ Builds organized structure
- ✅ Creates indices
- ✅ Logs everything

**What was missing**: The Designer Notebook (rules) + Builder Script (executor)

**Now you have both.**

---

## Next Steps

1. **Download these files** from `/mnt/user-data/outputs/`
2. **Read the Designer Notebook** to understand the logic
3. **Run the builder** on your existing files
4. **Review the placement** (dry run first)
5. **Adjust rules** if needed
6. **Integrate with your system** (connect to UniverseKernel, etc.)

The auto-organizer is ready.
Drop files → they get organized.

---

**Files Created**:
- `DESIGNER_NOTEBOOK.md` - The rules/brain
- `legacy_builder.py` - The executor
- This guide - How to use it

**Location**: `/mnt/user-data/outputs/`

[View DESIGNER_NOTEBOOK.md](computer:///mnt/user-data/outputs/DESIGNER_NOTEBOOK.md)
[View legacy_builder.py](computer:///mnt/user-data/outputs/legacy_builder.py)
