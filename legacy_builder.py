#!/usr/bin/env python3
"""
LegacyBuilder - The Auto-Organizer
Implements the Designer Notebook rules to automatically classify and place files

This is your "Builder that builds the app."
Drop files in → they get glyphed, classified, and placed automatically.
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class LegacyBuilder:
    def __init__(self, root_path: str = "."):
        self.root = Path(root_path)
        
        # The Five Master Categories
        self.categories = {
            "ENGINE": {
                "glyph": "◈",
                "folder": "engine",
                "keywords": ["Engine", "Core", "System", "Calculator", "State", "Manager"],
                "patterns": [r".*Engine\.js$", r".*Core\.js$", r".*System\.js$"],
                "description": "Core mechanics and logic"
            },
            "INTERFACE": {
                "glyph": "◯",
                "folder": "interface",
                "keywords": ["Component", "Widget", "UI", "View", "Screen", "Display"],
                "patterns": [r".*\.jsx$", r".*Component\.js$", r".*Widget\.js$"],
                "description": "User-facing UI components"
            },
            "AGENT": {
                "glyph": "◆",
                "folder": "agents",
                "keywords": ["Agent", "AI", "Bot", "Oracle", "Assistant", "Keyboard"],
                "patterns": [r".*Agent\.js$", r".*AI\.js$", r".*Oracle\.js$"],
                "description": "Autonomous entities"
            },
            "WORLD": {
                "glyph": "⬡",
                "folder": "world",
                "keywords": ["Spatial", "Location", "World", "Map", "Universe", "Navigation"],
                "patterns": [r".*Spatial\.js$", r".*Map\.js$", r".*Location\.js$"],
                "description": "Spatial systems and environments"
            },
            "KNOWLEDGE": {
                "glyph": "◉",
                "folder": "knowledge",
                "keywords": ["Context", "Knowledge", "Data", "Memory", "Doc", "Explanation"],
                "patterns": [r".*Context\.js$", r".*\.md$", r".*Data\.js$"],
                "description": "Information and documentation"
            }
        }
        
        # Secondary tags
        self.tags = {
            "core": ["Core", "Engine", "Manager"],
            "utility": ["Utils", "Helper", "Tool"],
            "experimental": ["Experimental", "Draft", "WIP"],
            "legacy": ["Legacy", "Old", "Deprecated"],
            "bridge": ["Bridge", "Connector", "Link"]
        }
        
        # Placement log
        self.placement_log = []
        
        # Initialize folder structure
        self.initialize_structure()
    
    def initialize_structure(self):
        """Create the 5 category folder structure"""
        for category, config in self.categories.items():
            folder = self.root / config["folder"]
            folder.mkdir(parents=True, exist_ok=True)
            
            # Create subfolders
            (folder / "core").mkdir(exist_ok=True)
            (folder / "utils").mkdir(exist_ok=True)
            (folder / "bridges").mkdir(exist_ok=True)
            
            # Create index file if doesn't exist
            index_path = folder / "_INDEX.json"
            if not index_path.exists():
                with open(index_path, 'w') as f:
                    json.dump({
                        "category": category,
                        "glyph": config["glyph"],
                        "description": config["description"],
                        "files": []
                    }, f, indent=2)
        
        print("✅ Folder structure initialized")
    
    def classify_file(self, filepath: Path) -> Tuple[str, List[str], float]:
        """
        Classify a file into one of the 5 categories
        Returns: (category, tags, confidence)
        """
        filename = filepath.name
        content = self._read_file(filepath)
        
        scores = {}
        
        for category, config in self.categories.items():
            score = 0.0
            
            # Check filename patterns
            for pattern in config["patterns"]:
                if re.match(pattern, filename):
                    score += 0.4
                    break
            
            # Check keywords in filename
            for keyword in config["keywords"]:
                if keyword.lower() in filename.lower():
                    score += 0.2
                    break
            
            # Check keywords in content
            if content:
                keyword_count = sum(1 for kw in config["keywords"] if kw in content)
                score += min(keyword_count * 0.1, 0.4)
            
            scores[category] = score
        
        # Get highest scoring category
        best_category = max(scores, key=scores.get)
        confidence = scores[best_category]
        
        # Determine secondary tags
        tags = self._determine_tags(filename, content)
        
        # Default to ENGINE if confidence too low
        if confidence < 0.2:
            best_category = "ENGINE"
            tags.append("unknown")
            confidence = 0.5
        
        return best_category, tags, confidence
    
    def _determine_tags(self, filename: str, content: str) -> List[str]:
        """Determine secondary tags for a file"""
        tags = []
        
        for tag, keywords in self.tags.items():
            for keyword in keywords:
                if keyword.lower() in filename.lower():
                    tags.append(tag)
                    break
                if content and keyword in content:
                    tags.append(tag)
                    break
        
        return list(set(tags))  # Remove duplicates
    
    def _read_file(self, filepath: Path) -> Optional[str]:
        """Safely read file content"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read(1000)  # Read first 1000 chars
        except:
            return None
    
    def place_file(self, filepath: Path, dry_run: bool = False) -> Dict:
        """
        Place a file in the correct category folder
        
        Args:
            filepath: Path to file to place
            dry_run: If True, don't actually move files
        
        Returns:
            Placement info dict
        """
        # Classify the file
        category, tags, confidence = self.classify_file(filepath)
        
        # Determine subfolder
        subfolder = "core" if "core" in tags else "utils"
        
        # Build target path
        category_config = self.categories[category]
        target_folder = self.root / category_config["folder"] / subfolder
        target_path = target_folder / filepath.name
        
        # Create placement record
        placement = {
            "timestamp": datetime.now().isoformat(),
            "filename": filepath.name,
            "source_path": str(filepath),
            "target_path": str(target_path),
            "category": category,
            "glyph": category_config["glyph"],
            "tags": tags,
            "confidence": confidence,
            "dry_run": dry_run
        }
        
        # Actually move the file (if not dry run)
        if not dry_run:
            try:
                import shutil
                shutil.copy2(filepath, target_path)
                print(f"✅ {filepath.name} → {category}/{subfolder}/ ({category_config['glyph']})")
            except Exception as e:
                print(f"❌ Error moving {filepath.name}: {e}")
                placement["error"] = str(e)
        else:
            print(f"🔍 [DRY RUN] {filepath.name} → {category}/{subfolder}/ ({category_config['glyph']})")
        
        # Update category index
        if not dry_run:
            self._update_index(category, placement)
        
        # Log placement
        self.placement_log.append(placement)
        
        return placement
    
    def _update_index(self, category: str, placement: Dict):
        """Update the category's index file"""
        category_config = self.categories[category]
        index_path = self.root / category_config["folder"] / "_INDEX.json"
        
        try:
            with open(index_path, 'r') as f:
                index = json.load(f)
            
            # Add file entry
            file_entry = {
                "filename": placement["filename"],
                "path": placement["target_path"],
                "glyph": placement["glyph"],
                "tags": placement["tags"],
                "added": placement["timestamp"],
                "confidence": placement["confidence"]
            }
            
            # Check if file already exists in index
            existing = next((f for f in index["files"] if f["filename"] == placement["filename"]), None)
            if existing:
                index["files"].remove(existing)
            
            index["files"].append(file_entry)
            
            with open(index_path, 'w') as f:
                json.dump(index, f, indent=2)
        
        except Exception as e:
            print(f"⚠️ Could not update index: {e}")
    
    def scan_directory(self, directory: Path, recursive: bool = True, dry_run: bool = False) -> List[Dict]:
        """
        Scan and classify all files in a directory
        
        Args:
            directory: Directory to scan
            recursive: Scan subdirectories
            dry_run: Don't actually move files
        
        Returns:
            List of placement records
        """
        placements = []
        
        # Get all files
        if recursive:
            files = [f for f in directory.rglob("*") if f.is_file()]
        else:
            files = [f for f in directory.glob("*") if f.is_file()]
        
        print(f"\n📂 Scanning {len(files)} files...\n")
        
        # Classify and place each file
        for filepath in files:
            # Skip hidden files and indices
            if filepath.name.startswith('.') or filepath.name == "_INDEX.json":
                continue
            
            placement = self.place_file(filepath, dry_run=dry_run)
            placements.append(placement)
        
        return placements
    
    def generate_report(self) -> str:
        """Generate a summary report of all placements"""
        report = []
        report.append("\n" + "="*60)
        report.append("LEGACY BUILDER REPORT")
        report.append("="*60 + "\n")
        
        # Count by category
        category_counts = {}
        for placement in self.placement_log:
            cat = placement["category"]
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        report.append("Files Placed by Category:")
        for category, count in sorted(category_counts.items()):
            glyph = self.categories[category]["glyph"]
            report.append(f"  {glyph} {category}: {count} files")
        
        report.append(f"\nTotal Files Processed: {len(self.placement_log)}")
        
        # Low confidence warnings
        low_confidence = [p for p in self.placement_log if p["confidence"] < 0.5]
        if low_confidence:
            report.append(f"\n⚠️ {len(low_confidence)} files with low confidence:")
            for p in low_confidence[:5]:  # Show first 5
                report.append(f"  - {p['filename']} → {p['category']} ({p['confidence']:.2f})")
        
        report.append("\n" + "="*60 + "\n")
        
        return "\n".join(report)
    
    def save_log(self, output_path: str = "placement_log.json"):
        """Save placement log to file"""
        with open(output_path, 'w') as f:
            json.dump(self.placement_log, f, indent=2)
        print(f"💾 Log saved to {output_path}")


def main():
    """Example usage"""
    import sys
    
    # Initialize builder
    builder = LegacyBuilder(root_path="./organized")
    
    # Check if directory argument provided
    if len(sys.argv) > 1:
        scan_dir = Path(sys.argv[1])
    else:
        print("Usage: python legacy_builder.py <directory_to_scan>")
        print("\nOr import and use programmatically:")
        print("  from legacy_builder import LegacyBuilder")
        print("  builder = LegacyBuilder()")
        print("  builder.scan_directory(Path('./my_files'))")
        return
    
    # Scan directory (dry run first)
    print("🔍 Running dry run...")
    builder.scan_directory(scan_dir, dry_run=True)
    
    # Ask for confirmation
    response = input("\n✅ Proceed with actual file placement? (y/n): ")
    
    if response.lower() == 'y':
        builder.placement_log = []  # Clear dry run log
        builder.scan_directory(scan_dir, dry_run=False)
        print(builder.generate_report())
        builder.save_log()
    else:
        print("❌ Cancelled")


if __name__ == "__main__":
    main()
