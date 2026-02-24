/**
 * MAIN APP WITH POV EXPERIENCE
 * 
 * Click houses to enter agent's consciousness!
 */

import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid, Environment, Stats } from '@react-three/drei';
import { useEffect, useState } from 'react';
import { useGame, useGameLoop, generateHousingDistrict } from './integration/useGame';
import { GameWorld3D } from './integration/GameComponentsWithPOV';
import { AgentPOVExperience } from './integration/AgentPOVExperience';
import { ConsciousAgent } from './agents/ConsciousAgent';

export function App() {
  const addAgent = useGame(state => state.addAgent);
  const addHouse = useGame(state => state.addHouse);
  const assignHome = useGame(state => state.assignHome);
  const agents = useGame(state => state.agents);
  const houses = useGame(state => state.houses);
  const selectedAgentId = useGame(state => state.selectedAgentId);
  
  // POV mode state
  const [povMode, setPovMode] = useState(false);
  const [povAgentId, setPovAgentId] = useState<string | null>(null);
  
  // Initialize game on mount
  useEffect(() => {
    const newHouses = generateHousingDistrict(0, 0, 8);
    newHouses.forEach(house => addHouse(house));
    
    const agentNames = [
      'Celestial', 'Aria', 'Nova', 'Zephyr', 'Luna', 
      'Sol', 'Atlas', 'Echo'
    ];
    
    agentNames.forEach((name, i) => {
      const angle = (i / agentNames.length) * Math.PI * 2;
      const radius = 5;
      
      addAgent({
        id: `agent_${i}`,
        name,
        position: {
          x: Math.cos(angle) * radius,
          y: 0,
          z: Math.sin(angle) * radius
        }
      });
    });
    
    setTimeout(() => {
      const currentAgents = useGame.getState().agents;
      const currentHouses = useGame.getState().houses;
      
      currentAgents.forEach((agent, i) => {
        if (i < currentHouses.length) {
          assignHome(agent.id, currentHouses[i].id);
        }
      });
    }, 100);
  }, []);
  
  const selectedAgent = agents.find(a => a.id === selectedAgentId);
  const povAgent = agents.find(a => a.id === povAgentId);
  
  const handleEnterPOV = (agentId: string) => {
    setPovAgentId(agentId);
    setPovMode(true);
  };
  
  const handleExitPOV = () => {
    setPovMode(false);
    setPovAgentId(null);
  };
  
  return (
    <div style={{ width: '100vw', height: '100vh', background: '#0a0a1a' }}>
      {/* POV Experience (full screen overlay) */}
      {povMode && povAgent && (
        <AgentPOVExperience 
          agent={povAgent}
          onExit={handleExitPOV}
        />
      )}
      
      {/* 3D Canvas (hidden when in POV mode) */}
      <div style={{ 
        display: povMode ? 'none' : 'block',
        width: '100%',
        height: '100%'
      }}>
        <Canvas
          shadows
          camera={{ position: [15, 15, 15], fov: 60 }}
        >
          <color attach="background" args={['#0a0a1a']} />
          <fog attach="fog" args={['#0a0a1a', 20, 60]} />
          
          <Environment preset="night" />
          
          <Grid
            args={[50, 50]}
            cellSize={1}
            cellThickness={0.5}
            cellColor="#6366f1"
            sectionSize={5}
            sectionThickness={1}
            sectionColor="#8b5cf6"
            fadeDistance={50}
            fadeStrength={1}
            infiniteGrid
          />
          
          <GameWorld3D onEnterAgentPOV={handleEnterPOV} />
          
          <GameLoop />
          
          <OrbitControls
            enablePan
            enableZoom
            enableRotate
            minDistance={5}
            maxDistance={50}
            target={[0, 0, 0]}
          />
          
          <Stats />
        </Canvas>
        
        <UIOverlay agent={selectedAgent} />
        
        {/* Instructions overlay */}
        <div style={{
          position: 'absolute',
          bottom: 20,
          left: '50%',
          transform: 'translateX(-50%)',
          background: 'rgba(0,0,0,0.8)',
          color: 'white',
          padding: '15px 30px',
          borderRadius: '10px',
          fontSize: '14px',
          border: '1px solid rgba(255,255,255,0.3)',
          textAlign: 'center'
        }}>
          🏠 Click on houses with gold orbs to experience agent's consciousness
        </div>
      </div>
    </div>
  );
}

// =======================
// GAME LOOP
// =======================

function GameLoop() {
  const updateAgents = useGame(state => state.updateAgents);
  
  useFrame((state, delta) => {
    updateAgents(delta * 1000);
  });
  
  return null;
}

// =======================
// UI OVERLAY
// =======================

function UIOverlay({ agent }: { agent?: ConsciousAgent }) {
  if (!agent) return null;
  
  const drives = agent.getDrives();
  const traits = agent.getTraits();
  const mood = agent.getMood();
  const hdState = agent.getHDChart();
  
  return (
    <div style={{
      position: 'absolute',
      top: 20,
      right: 20,
      background: 'rgba(0, 0, 0, 0.8)',
      color: 'white',
      padding: '20px',
      borderRadius: '10px',
      fontFamily: 'monospace',
      fontSize: '12px',
      minWidth: '300px',
      maxHeight: '80vh',
      overflow: 'auto'
    }}>
      <h2 style={{ margin: '0 0 10px 0', color: agent.color }}>{agent.name}</h2>
      
      <div style={{ marginBottom: '15px' }}>
        <h3 style={{ fontSize: '14px', margin: '5px 0' }}>Human Design</h3>
        <div>Type: {hdState.phenotype?.bodyType}</div>
        <div>Archetype: {hdState.currentArchetype}</div>
        <div>Consciousness: {(hdState.consciousnessLevel * 100).toFixed(0)}%</div>
        <div>Gates: {hdState.activeGates.length}/64</div>
      </div>
      
      <div style={{ marginBottom: '15px' }}>
        <h3 style={{ fontSize: '14px', margin: '5px 0' }}>Drives</h3>
        {Object.entries(drives).map(([key, value]) => (
          <DriveBar key={key} label={key} value={value} />
        ))}
      </div>
      
      <div style={{ marginBottom: '15px' }}>
        <h3 style={{ fontSize: '14px', margin: '5px 0' }}>Traits</h3>
        <div>Energy: {(traits.energyLevel * 100).toFixed(0)}%</div>
        <div>Creativity: {(traits.creativity * 100).toFixed(0)}%</div>
        <div>Intuition: {(traits.intuition * 100).toFixed(0)}%</div>
        <div>Assertiveness: {(traits.assertiveness * 100).toFixed(0)}%</div>
      </div>
      
      <div style={{ marginBottom: '15px' }}>
        <h3 style={{ fontSize: '14px', margin: '5px 0' }}>Mood (PAD)</h3>
        <div>Valence: {mood.valence > 0 ? '😊' : '😔'} {mood.valence.toFixed(2)}</div>
        <div>Arousal: {mood.arousal > 0.5 ? '⚡' : '😴'} {mood.arousal.toFixed(2)}</div>
        <div>Dominance: {mood.dominance > 0.5 ? '💪' : '🤝'} {mood.dominance.toFixed(2)}</div>
      </div>
      
      {agent.currentAction && (
        <div>
          <h3 style={{ fontSize: '14px', margin: '5px 0' }}>Current Action</h3>
          <div style={{ color: agent.color, fontWeight: 'bold' }}>
            {agent.currentAction.type}
          </div>
        </div>
      )}
      
      {agent.home && (
        <div style={{ marginTop: '10px', opacity: 0.7 }}>
          {agent.isAtHome ? '🏠 At Home' : '🚶 Away from Home'}
        </div>
      )}
      
      {agent.isAtHome && (
        <div style={{
          marginTop: '15px',
          padding: '10px',
          background: 'rgba(255, 215, 0, 0.2)',
          borderRadius: '5px',
          border: '1px solid rgba(255, 215, 0, 0.5)',
          fontSize: '11px'
        }}>
          💡 Click their house to experience their consciousness!
        </div>
      )}
    </div>
  );
}

function DriveBar({ label, value }: { label: string; value: number }) {
  const percentage = value * 100;
  
  return (
    <div style={{ marginBottom: '5px' }}>
      <div style={{ fontSize: '10px', marginBottom: '2px' }}>
        {label}: {percentage.toFixed(0)}%
      </div>
      <div style={{
        width: '100%',
        height: '6px',
        background: 'rgba(255,255,255,0.1)',
        borderRadius: '3px',
        overflow: 'hidden'
      }}>
        <div style={{
          width: `${percentage}%`,
          height: '100%',
          background: `hsl(${percentage * 1.2}, 70%, 50%)`,
          transition: 'width 0.3s'
        }} />
      </div>
    </div>
  );
}

export default App;
