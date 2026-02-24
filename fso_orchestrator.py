"""
FSO ORCHESTRATOR
Full-Stack Orchestrator - The bridge between User Universe and Agent Universe

When user builds in app → Sims build in Unity
When Sims build in Unity → App features generate

This coordinates SIMULTANEOUS CONSTRUCTION across universes.
"""

import json
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from enum import Enum

class UniverseType(Enum):
    USER = "user"
    AGENT = "agent"

class ConstructionType(Enum):
    ENGINE = "engine"       # Core system building
    INTERFACE = "interface" # UI widget building  
    AGENT = "agent"         # Agent entity building
    WORLD = "world"         # Location/space building
    KNOWLEDGE = "knowledge" # Data/context building

class FSOOrchestrator:
    """
    Orchestrates construction between User Universe (app) and Agent Universe (Sims/Unity)
    
    The FSO (Full-Stack Orchestrator) ensures that:
    - User actions trigger Agent construction
    - Agent construction generates User features
    - Both universes stay in sync
    - Construction happens simultaneously
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Construction queues for each universe
        self.user_queue = []
        self.agent_queue = []
        
        # Active constructions
        self.active_constructions = {}
        
        # Bridge status
        self.bridge_active = False
        self.sync_interval = 1.0  # seconds
        
        # Construction history
        self.history = []
        
        # Unity connection (placeholder - would connect to actual Unity)
        self.unity_connection = None
        
        # Sims → Code mappings
        self.building_to_code = {
            "engine": {
                "sims_building": "Factory",
                "generates": "EngineClass",
                "template": "engine_template.js"
            },
            "interface": {
                "sims_building": "Gallery",
                "generates": "ReactComponent",
                "template": "component_template.jsx"
            },
            "agent": {
                "sims_building": "Academy",
                "generates": "AgentClass",
                "template": "agent_template.js"
            },
            "world": {
                "sims_building": "Landmark",
                "generates": "LocationDefinition",
                "template": "location_template.js"
            },
            "knowledge": {
                "sims_building": "Library",
                "generates": "DataStore",
                "template": "knowledge_template.json"
            }
        }
        
        print("🌉 FSO Orchestrator initialized")
    
    # ===== USER → AGENT SYNC =====
    
    def user_builds(self, construction_type: str, spec: Dict) -> Dict:
        """
        User initiates construction in app
        This triggers Sim construction in Unity
        """
        construction_id = f"user_{datetime.now().timestamp()}"
        
        construction = {
            "id": construction_id,
            "source": UniverseType.USER.value,
            "type": construction_type,
            "spec": spec,
            "status": "initiated",
            "timestamp": datetime.now().isoformat(),
            "agent_task": None
        }
        
        # Add to user queue
        self.user_queue.append(construction)
        
        # Create corresponding agent task
        agent_task = self._create_agent_task(construction)
        construction["agent_task"] = agent_task["id"]
        
        # Add to agent queue
        self.agent_queue.append(agent_task)
        
        # Record in history
        self.history.append({
            "event": "user_initiated",
            "construction_id": construction_id,
            "type": construction_type,
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"👤 USER builds {construction_type}: {spec.get('name', 'unnamed')}")
        print(f"🤖 → Assigned to AGENT task: {agent_task['id']}")
        
        return construction
    
    def _create_agent_task(self, user_construction: Dict) -> Dict:
        """
        Convert user construction into agent building task
        """
        construction_type = user_construction["type"]
        mapping = self.building_to_code.get(construction_type, {})
        
        agent_task = {
            "id": f"agent_{datetime.now().timestamp()}",
            "source": UniverseType.AGENT.value,
            "type": construction_type,
            "building_type": mapping.get("sims_building", "Generic"),
            "spec": {
                "name": user_construction["spec"].get("name", "Unnamed"),
                "dimensions": self._calculate_dimensions(user_construction),
                "location": self._assign_location(construction_type),
                "materials": self._determine_materials(construction_type)
            },
            "status": "queued",
            "user_construction_id": user_construction["id"],
            "timestamp": datetime.now().isoformat()
        }
        
        return agent_task
    
    # ===== AGENT → USER SYNC =====
    
    def agent_completes_construction(self, agent_task_id: str, result: Dict) -> Dict:
        """
        Agent (Sim) completes construction in Unity
        This generates code/features in user app
        """
        # Find the agent task
        agent_task = next((t for t in self.agent_queue if t["id"] == agent_task_id), None)
        
        if not agent_task:
            print(f"⚠️ Agent task {agent_task_id} not found")
            return None
        
        # Update task status
        agent_task["status"] = "completed"
        agent_task["result"] = result
        
        # Generate code from agent construction
        generated_code = self._generate_code_from_construction(agent_task, result)
        
        # Update corresponding user construction
        user_construction_id = agent_task.get("user_construction_id")
        if user_construction_id:
            user_construction = next(
                (c for c in self.user_queue if c["id"] == user_construction_id),
                None
            )
            if user_construction:
                user_construction["status"] = "completed"
                user_construction["generated_code"] = generated_code
        
        # Record in history
        self.history.append({
            "event": "agent_completed",
            "agent_task_id": agent_task_id,
            "type": agent_task["type"],
            "generated": generated_code.get("filename"),
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"🤖 AGENT completed {agent_task['building_type']}")
        print(f"📄 → Generated: {generated_code.get('filename')}")
        
        return generated_code
    
    def _generate_code_from_construction(self, agent_task: Dict, result: Dict) -> Dict:
        """
        Convert completed Sim building into actual code
        This is where Unity construction → App feature happens
        """
        construction_type = agent_task["type"]
        mapping = self.building_to_code.get(construction_type, {})
        
        # Get template
        template = mapping.get("template", "default.js")
        
        # Generate code based on building properties
        generated = {
            "type": mapping.get("generates", "GenericClass"),
            "filename": f"{agent_task['spec']['name']}.js",
            "template": template,
            "properties": {
                "name": agent_task['spec']['name'],
                "dimensions": result.get("dimensions", {}),
                "materials": result.get("materials", []),
                "connections": result.get("connections", []),
                "glyph": self._get_glyph_for_type(construction_type)
            },
            "category": construction_type,
            "path": self._get_output_path(construction_type, agent_task['spec']['name'])
        }
        
        return generated
    
    # ===== SIMULTANEOUS CONSTRUCTION =====
    
    async def sync_construction(self):
        """
        Main sync loop - keeps both universes in sync
        Runs continuously while bridge is active
        """
        self.bridge_active = True
        
        print("🌉 Bridge activated - Starting sync loop")
        
        while self.bridge_active:
            # Check user queue for new constructions
            for construction in self.user_queue:
                if construction["status"] == "initiated":
                    # Send to Unity (placeholder)
                    await self._send_to_unity(construction["agent_task"])
                    construction["status"] = "in_progress"
            
            # Check agent queue for completed constructions
            for task in self.agent_queue:
                if task["status"] == "completed":
                    # Generate code
                    # (In real system, Unity would callback here)
                    pass
            
            # Wait before next sync
            await asyncio.sleep(self.sync_interval)
    
    async def _send_to_unity(self, agent_task: Dict):
        """
        Send construction task to Unity/Sims
        In real system, this would use FSO protocol to communicate
        """
        # Placeholder - would send via websocket/gRPC/etc
        print(f"  📤 Sending to Unity: {agent_task['building_type']}")
        
        # Simulate construction time
        await asyncio.sleep(2.0)
        
        # Simulate completion (in real system, Unity would callback)
        result = {
            "dimensions": {"width": 10, "height": 20, "depth": 10},
            "materials": ["wood", "stone"],
            "connections": []
        }
        
        self.agent_completes_construction(agent_task["id"], result)
    
    # ===== CONSTRUCTION COORDINATION =====
    
    def coordinate_build(self, name: str, construction_type: str, spec: Dict) -> Dict:
        """
        High-level API for coordinated construction
        User says "build X" → This handles both universes
        """
        print(f"\n🏗️ COORDINATING CONSTRUCTION: {name}")
        print(f"   Type: {construction_type}")
        print(f"   Spec: {spec}")
        
        # Initiate in user universe
        construction = self.user_builds(construction_type, {
            "name": name,
            **spec
        })
        
        # Agent construction happens automatically via sync loop
        
        return {
            "user_construction": construction,
            "agent_task": construction["agent_task"],
            "status": "coordinated"
        }
    
    # ===== HELPERS =====
    
    def _calculate_dimensions(self, construction: Dict) -> Dict:
        """Calculate building dimensions based on spec"""
        spec = construction["spec"]
        
        # Simple calculation - could be much more sophisticated
        return {
            "width": spec.get("width", 10),
            "height": spec.get("height", 20),
            "depth": spec.get("depth", 10)
        }
    
    def _assign_location(self, construction_type: str) -> Dict:
        """Assign location in agent universe for building"""
        # Would use SpatialEngine to find available spot
        return {
            "x": 0,
            "y": 0,
            "z": 0,
            "sector": f"{construction_type}_zone"
        }
    
    def _determine_materials(self, construction_type: str) -> List[str]:
        """Determine what materials Sim needs to build"""
        materials_by_type = {
            "engine": ["metal", "circuits", "energy"],
            "interface": ["glass", "light", "color"],
            "agent": ["consciousness", "code", "autonomy"],
            "world": ["space", "matter", "gravity"],
            "knowledge": ["data", "memory", "wisdom"]
        }
        return materials_by_type.get(construction_type, ["generic"])
    
    def _get_glyph_for_type(self, construction_type: str) -> str:
        """Get glyph symbol for construction type"""
        glyphs = {
            "engine": "◈",
            "interface": "◯",
            "agent": "◆",
            "world": "⬡",
            "knowledge": "◉"
        }
        return glyphs.get(construction_type, "○")
    
    def _get_output_path(self, category: str, name: str) -> str:
        """Get output path for generated code"""
        return f"./organized/{category}/core/{name}.js"
    
    # ===== STATUS & REPORTING =====
    
    def get_status(self) -> Dict:
        """Get current bridge status"""
        return {
            "bridge_active": self.bridge_active,
            "user_queue_length": len(self.user_queue),
            "agent_queue_length": len(self.agent_queue),
            "active_constructions": len(self.active_constructions),
            "total_history": len(self.history)
        }
    
    def get_construction_status(self, construction_id: str) -> Optional[Dict]:
        """Get status of specific construction"""
        # Check user queue
        construction = next(
            (c for c in self.user_queue if c["id"] == construction_id),
            None
        )
        
        if construction:
            return {
                "id": construction_id,
                "status": construction["status"],
                "type": construction["type"],
                "agent_task": construction.get("agent_task"),
                "generated_code": construction.get("generated_code")
            }
        
        return None
    
    def generate_report(self) -> str:
        """Generate construction report"""
        report = []
        report.append("\n" + "="*60)
        report.append("FSO ORCHESTRATOR REPORT")
        report.append("="*60 + "\n")
        
        # Count by type
        type_counts = {}
        for event in self.history:
            if event["event"] == "user_initiated":
                t = event["type"]
                type_counts[t] = type_counts.get(t, 0) + 1
        
        report.append("Constructions by Type:")
        for construction_type, count in sorted(type_counts.items()):
            glyph = self._get_glyph_for_type(construction_type)
            report.append(f"  {glyph} {construction_type}: {count}")
        
        report.append(f"\nTotal Constructions: {len(type_counts)}")
        report.append(f"User Queue: {len(self.user_queue)} items")
        report.append(f"Agent Queue: {len(self.agent_queue)} items")
        
        report.append("\n" + "="*60 + "\n")
        
        return "\n".join(report)
    
    def export_history(self, output_path: str = "fso_history.json"):
        """Export construction history"""
        with open(output_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"💾 History saved to {output_path}")


# ===== USAGE EXAMPLE =====

async def main():
    """Example usage of FSO Orchestrator"""
    
    # Initialize orchestrator
    fso = FSOOrchestrator()
    
    # Start sync loop (in background)
    sync_task = asyncio.create_task(fso.sync_construction())
    
    # Simulate user building something
    print("\n" + "="*60)
    print("EXAMPLE: User builds a new widget")
    print("="*60)
    
    result = fso.coordinate_build(
        name="DashboardWidget",
        construction_type="interface",
        spec={
            "width": 300,
            "height": 200,
            "color": "blue"
        }
    )
    
    # Wait for construction
    await asyncio.sleep(3)
    
    # Check status
    status = fso.get_construction_status(result["user_construction"]["id"])
    print(f"\n✅ Construction Status: {status['status']}")
    
    # Generate report
    print(fso.generate_report())
    
    # Stop sync loop
    fso.bridge_active = False
    await sync_task


if __name__ == "__main__":
    asyncio.run(main())
