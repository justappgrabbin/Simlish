"""
CONTEXT KERNEL
The "How Does This Fit?" Explainer

When you look at ANY component, this shows:
- What it is (glyph + category)
- What it connects to
- How it fits in the system
- Where it goes
- What depends on it

This is the pop-up context system.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

class ContextKernel:
    """
    Explains connections and placement for any component
    Shows the "how does this fit?" context
    """
    
    def __init__(self, organized_root: str = "./organized"):
        self.root = Path(organized_root)
        
        # The 5 category glyphs
        self.glyphs = {
            "engine": "◈",
            "interface": "◯", 
            "agent": "◆",
            "world": "⬡",
            "knowledge": "◉"
        }
        
        # Load all indices
        self.indices = self._load_indices()
        
        # Build connection map
        self.connection_map = self._build_connection_map()
        
        print("🔍 ContextKernel initialized")
    
    def _load_indices(self) -> Dict:
        """Load all category indices"""
        indices = {}
        
        for category in self.glyphs.keys():
            index_path = self.root / category / "_INDEX.json"
            if index_path.exists():
                with open(index_path, 'r') as f:
                    indices[category] = json.load(f)
        
        return indices
    
    def _build_connection_map(self) -> Dict:
        """Build map of what connects to what"""
        connections = {}
        
        # Scan all files for imports/dependencies
        for category, index in self.indices.items():
            for file_entry in index.get("files", []):
                filename = file_entry["filename"]
                filepath = file_entry["path"]
                
                # Detect connections by scanning imports
                connected_to = self._detect_connections(filepath)
                
                connections[filename] = {
                    "category": category,
                    "glyph": self.glyphs[category],
                    "path": filepath,
                    "connects_to": connected_to,
                    "connected_from": []  # Will populate
                }
        
        # Populate reverse connections
        for filename, data in connections.items():
            for connected_file in data["connects_to"]:
                if connected_file in connections:
                    connections[connected_file]["connected_from"].append(filename)
        
        return connections
    
    def _detect_connections(self, filepath: str) -> List[str]:
        """Detect what a file imports/connects to"""
        connections = []
        
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Look for imports (JavaScript style)
            import_patterns = [
                r"import .* from ['\"](.*)['\"]",
                r"require\(['\"](.*)['\"]\)",
            ]
            
            import re
            for pattern in import_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    # Extract filename from path
                    imported_file = Path(match).name
                    if imported_file and not imported_file.startswith('.'):
                        connections.append(imported_file)
        
        except:
            pass
        
        return connections
    
    # ===== CONTEXT QUERIES =====
    
    def get_context(self, filename: str) -> Dict:
        """
        Get complete context for a component
        This is what shows in the pop-up
        """
        if filename not in self.connection_map:
            return {"error": f"Component '{filename}' not found"}
        
        component = self.connection_map[filename]
        
        context = {
            "component": filename,
            "category": component["category"],
            "glyph": component["glyph"],
            "path": component["path"],
            
            "what_it_is": self._explain_what_it_is(component),
            "what_it_connects_to": self._explain_connections_to(component),
            "what_depends_on_it": self._explain_dependencies(component),
            "where_it_goes": self._explain_placement(component),
            "how_to_use": self._explain_usage(component),
            
            "visualization": self._generate_connection_diagram(filename)
        }
        
        return context
    
    def _explain_what_it_is(self, component: Dict) -> str:
        """Explain what this component is"""
        category = component["category"]
        
        explanations = {
            "engine": "Core system component that powers functionality",
            "interface": "Visual component users interact with",
            "agent": "Autonomous entity that acts independently",
            "world": "Location or spatial structure in the universe",
            "knowledge": "Data, context, or information storage"
        }
        
        return f"{component['glyph']} {explanations.get(category, 'Component')}"
    
    def _explain_connections_to(self, component: Dict) -> List[Dict]:
        """Explain what this connects to"""
        connections = []
        
        for connected_file in component["connects_to"]:
            if connected_file in self.connection_map:
                connected = self.connection_map[connected_file]
                connections.append({
                    "file": connected_file,
                    "glyph": connected["glyph"],
                    "category": connected["category"],
                    "why": self._explain_connection_reason(
                        component["category"],
                        connected["category"]
                    )
                })
        
        return connections
    
    def _explain_connection_reason(self, from_cat: str, to_cat: str) -> str:
        """Explain WHY two categories connect"""
        reasons = {
            ("interface", "engine"): "Uses engine to display data",
            ("agent", "engine"): "Uses engine to process logic",
            ("agent", "world"): "Navigates locations in world",
            ("interface", "world"): "Displays world information",
            ("engine", "knowledge"): "Reads context and data",
            ("agent", "knowledge"): "Learns from stored patterns",
            ("interface", "agent"): "Shows agent status/actions"
        }
        
        return reasons.get((from_cat, to_cat), "Provides functionality")
    
    def _explain_dependencies(self, component: Dict) -> List[Dict]:
        """Explain what depends on this component"""
        dependencies = []
        
        for dependent_file in component["connected_from"]:
            if dependent_file in self.connection_map:
                dependent = self.connection_map[dependent_file]
                dependencies.append({
                    "file": dependent_file,
                    "glyph": dependent["glyph"],
                    "category": dependent["category"]
                })
        
        return dependencies
    
    def _explain_placement(self, component: Dict) -> Dict:
        """Explain where this component goes"""
        category = component["category"]
        
        placement = {
            "category_folder": f"/{category}/",
            "subfolder": self._determine_subfolder(component),
            "full_path": component["path"],
            "reason": self._explain_placement_reason(component)
        }
        
        return placement
    
    def _determine_subfolder(self, component: Dict) -> str:
        """Determine subfolder based on component type"""
        path = component["path"]
        
        if "/core/" in path:
            return "core"
        elif "/utils/" in path:
            return "utils"
        elif "/bridges/" in path:
            return "bridges"
        else:
            return "root"
    
    def _explain_placement_reason(self, component: Dict) -> str:
        """Explain WHY it's placed where it is"""
        category = component["category"]
        subfolder = self._determine_subfolder(component)
        
        reasons = {
            ("engine", "core"): "Essential system engine",
            ("engine", "utils"): "Helper functionality for engines",
            ("interface", "core"): "Primary user interface component",
            ("interface", "utils"): "UI utility/helper",
            ("agent", "core"): "Main autonomous agent",
            ("world", "core"): "Primary location/space",
            ("knowledge", "core"): "Core data storage"
        }
        
        return reasons.get((category, subfolder), f"Belongs in {category}")
    
    def _explain_usage(self, component: Dict) -> Dict:
        """Explain how to use this component"""
        category = component["category"]
        filename = Path(component["path"]).stem  # name without extension
        
        usage_templates = {
            "engine": {
                "import": f"import {{{filename}Engine}} from '{component['path']}'",
                "initialize": f"const engine = new {filename}Engine(config)",
                "use": f"const result = engine.someMethod()"
            },
            "interface": {
                "import": f"import {{{filename}}} from '{component['path']}'",
                "use": f"<{filename} universe={{universe}} />"
            },
            "agent": {
                "import": f"import {{{filename}}} from '{component['path']}'",
                "register": f"universe.registerAgent('{filename}', new {filename}(universe))"
            },
            "world": {
                "import": f"import {{{filename}}} from '{component['path']}'",
                "register": f"spatial.registerLocation('{filename}', {filename})"
            },
            "knowledge": {
                "import": f"import {filename}Data from '{component['path']}'",
                "use": f"const context = {filename}Data.explanations"
            }
        }
        
        return usage_templates.get(category, {})
    
    def _generate_connection_diagram(self, filename: str) -> str:
        """Generate ASCII diagram of connections"""
        if filename not in self.connection_map:
            return ""
        
        component = self.connection_map[filename]
        glyph = component["glyph"]
        
        lines = []
        lines.append(f"\n{glyph} {filename}")
        lines.append("│")
        
        # Connections TO
        if component["connects_to"]:
            lines.append("├─ USES:")
            for conn_file in component["connects_to"]:
                if conn_file in self.connection_map:
                    conn_glyph = self.connection_map[conn_file]["glyph"]
                    lines.append(f"│  └─ {conn_glyph} {conn_file}")
        
        # Connections FROM
        if component["connected_from"]:
            lines.append("├─ USED BY:")
            for dep_file in component["connected_from"]:
                if dep_file in self.connection_map:
                    dep_glyph = self.connection_map[dep_file]["glyph"]
                    lines.append(f"│  └─ {dep_glyph} {dep_file}")
        
        return "\n".join(lines)
    
    # ===== CATEGORY QUERIES =====
    
    def get_category_context(self, category: str) -> Dict:
        """Get context for entire category"""
        if category not in self.glyphs:
            return {"error": f"Category '{category}' not found"}
        
        components = [
            comp for comp in self.connection_map.values()
            if comp["category"] == category
        ]
        
        return {
            "category": category,
            "glyph": self.glyphs[category],
            "component_count": len(components),
            "components": [c["path"] for c in components],
            "total_connections": sum(
                len(c["connects_to"]) for c in components
            )
        }
    
    # ===== SEARCH =====
    
    def search_by_connection(self, target_file: str) -> List[str]:
        """Find all components that connect to target"""
        results = []
        
        for filename, component in self.connection_map.items():
            if target_file in component["connects_to"]:
                results.append(filename)
        
        return results
    
    def search_by_category(self, category: str) -> List[str]:
        """Get all components in category"""
        return [
            filename for filename, comp in self.connection_map.items()
            if comp["category"] == category
        ]
    
    # ===== POP-UP GENERATION =====
    
    def generate_popup_html(self, filename: str) -> str:
        """Generate HTML for context pop-up"""
        context = self.get_context(filename)
        
        if "error" in context:
            return f"<div class='error'>{context['error']}</div>"
        
        html = f"""
        <div class='context-popup'>
            <div class='header'>
                <span class='glyph'>{context['glyph']}</span>
                <h3>{context['component']}</h3>
                <span class='category'>{context['category'].upper()}</span>
            </div>
            
            <div class='section'>
                <h4>What It Is</h4>
                <p>{context['what_it_is']}</p>
            </div>
            
            <div class='section'>
                <h4>Connects To</h4>
                <ul>
                    {''.join(f"<li>{conn['glyph']} {conn['file']} - {conn['why']}</li>" 
                             for conn in context['what_it_connects_to'])}
                </ul>
            </div>
            
            <div class='section'>
                <h4>Depended On By</h4>
                <ul>
                    {''.join(f"<li>{dep['glyph']} {dep['file']}</li>" 
                             for dep in context['what_depends_on_it'])}
                </ul>
            </div>
            
            <div class='section'>
                <h4>Placement</h4>
                <code>{context['where_it_goes']['full_path']}</code>
                <p>{context['where_it_goes']['reason']}</p>
            </div>
            
            <div class='section'>
                <h4>How To Use</h4>
                <pre>{json.dumps(context['how_to_use'], indent=2)}</pre>
            </div>
            
            <div class='diagram'>
                <pre>{context['visualization']}</pre>
            </div>
        </div>
        """
        
        return html


# ===== USAGE EXAMPLE =====

if __name__ == "__main__":
    # Initialize
    kernel = ContextKernel(organized_root="./organized")
    
    # Get context for a component
    context = kernel.get_context("TrinityEngine.js")
    
    print("\n" + "="*60)
    print("CONTEXT KERNEL EXAMPLE")
    print("="*60)
    print(f"\n{context['what_it_is']}")
    print(f"\nPath: {context['where_it_goes']['full_path']}")
    print(f"Reason: {context['where_it_goes']['reason']}")
    
    print("\nConnects to:")
    for conn in context['what_it_connects_to']:
        print(f"  {conn['glyph']} {conn['file']} - {conn['why']}")
    
    print(context['visualization'])
    
    # Generate pop-up HTML
    html = kernel.generate_popup_html("TrinityEngine.js")
    with open("context_popup.html", 'w') as f:
        f.write(html)
    
    print("\n✅ Pop-up HTML generated: context_popup.html")
