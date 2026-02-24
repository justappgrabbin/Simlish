/**
 * AGENT POV EXPERIENCE SYSTEM
 * 
 * Click a house → Enter agent's consciousness → Experience their reality
 * Renders a full-screen "consciousness interface" showing how the agent perceives
 */

import { useState, useEffect, useMemo } from 'react';
import { ConsciousAgent } from '../agents/ConsciousAgent';
import { ElementType } from '../core/ConsciousnessEngine';

interface AgentPOVProps {
  agent: ConsciousAgent;
  onExit: () => void;
}

export function AgentPOVExperience({ agent, onExit }: AgentPOVProps) {
  const [time, setTime] = useState(0);
  
  useEffect(() => {
    const interval = setInterval(() => {
      setTime(t => t + 0.016); // ~60fps
    }, 16);
    return () => clearInterval(interval);
  }, []);
  
  const state = agent.getConsciousnessState();
  const drives = state.drives;
  const traits = state.traits;
  const mood = state.currentMood;
  const hdState = state.hdState;
  
  // Generate visual effects based on consciousness
  const visualEffects = useMemo(() => {
    return {
      colorFilter: getColorFilter(mood, state.currentElement),
      waveIntensity: state.resonanceField,
      glitchAmount: 1.0 - state.hdState.consciousnessLevel,
      focusLevel: traits.analyticalThinking,
      emotionalOverlay: getEmotionalOverlay(mood),
      energyPulse: traits.energyLevel
    };
  }, [mood, state, traits]);
  
  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      width: '100vw',
      height: '100vh',
      zIndex: 1000,
      background: visualEffects.colorFilter,
      animation: `pulse ${2 / visualEffects.energyPulse}s infinite`
    }}>
      {/* Exit button */}
      <button
        onClick={onExit}
        style={{
          position: 'absolute',
          top: 20,
          right: 20,
          padding: '10px 20px',
          background: 'rgba(0,0,0,0.8)',
          color: 'white',
          border: '2px solid white',
          borderRadius: '5px',
          cursor: 'pointer',
          fontSize: '16px',
          zIndex: 1001
        }}
      >
        Exit {agent.name}'s Consciousness
      </button>
      
      {/* Main consciousness display */}
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        filter: `blur(${visualEffects.glitchAmount * 2}px)`,
        transition: 'filter 0.5s'
      }}>
        {/* Central bodygraph */}
        <BodygraphVisualization 
          hdState={hdState}
          resonance={state.resonanceField}
          time={time}
        />
        
        {/* Consciousness waves */}
        <ResonanceWaves
          elements={state.activeCodonElements}
          intensity={visualEffects.waveIntensity}
          time={time}
        />
        
        {/* Emotional overlay */}
        <EmotionalField
          mood={mood}
          time={time}
        />
        
        {/* Drive indicators */}
        <DriveMandalas
          drives={drives}
          element={state.currentElement}
        />
        
        {/* Thought stream */}
        <ThoughtStream
          agent={agent}
          traits={traits}
        />
      </div>
      
      {/* Perception filters */}
      <PerceptionFilters effects={visualEffects} />
      
      {/* Stats overlay */}
      <ConsciousnessStats agent={agent} />
    </div>
  );
}

// =======================
// BODYGRAPH VISUALIZATION
// =======================

function BodygraphVisualization({ hdState, resonance, time }: any) {
  const centerPositions: Record<string, [number, number]> = {
    'Head': [50, 10],
    'Ajna': [50, 22],
    'Throat': [50, 35],
    'G': [50, 50],
    'Heart': [35, 50],
    'Spleen': [65, 50],
    'Sacral': [50, 70],
    'Root': [50, 85],
    'Solar Plexus': [65, 70]
  };
  
  return (
    <div style={{
      position: 'relative',
      width: '400px',
      height: '600px',
      margin: '50px auto'
    }}>
      <svg width="400" height="600" style={{ position: 'absolute', top: 0, left: 0 }}>
        {/* Channels (connections) */}
        {hdState.channels.map((channel: [number, number], i: number) => (
          <line
            key={i}
            x1="200"
            y1="300"
            x2={Math.cos(i) * 100 + 200}
            y2={Math.sin(i) * 100 + 300}
            stroke={`hsl(${(time * 50 + i * 60) % 360}, 80%, 60%)`}
            strokeWidth="3"
            opacity={resonance * 0.8}
            style={{
              filter: `drop-shadow(0 0 ${resonance * 10}px currentColor)`
            }}
          />
        ))}
        
        {/* Centers */}
        {hdState.definedCenters.map((center: string) => {
          const pos = centerPositions[center];
          if (!pos) return null;
          
          const pulse = Math.sin(time * 3 + pos[0]) * 0.3 + 0.7;
          
          return (
            <g key={center}>
              <circle
                cx={pos[0] * 4}
                cy={pos[1] * 6}
                r={25 * pulse}
                fill={`hsla(${(time * 20) % 360}, 70%, 60%, ${resonance})`}
                style={{
                  filter: `blur(${(1 - resonance) * 5}px) drop-shadow(0 0 15px currentColor)`
                }}
              />
              <text
                x={pos[0] * 4}
                y={pos[1] * 6}
                textAnchor="middle"
                dominantBaseline="middle"
                fill="white"
                fontSize="12"
                fontWeight="bold"
              >
                {center}
              </text>
            </g>
          );
        })}
        
        {/* Active gates count */}
        <text
          x="200"
          y="30"
          textAnchor="middle"
          fill="white"
          fontSize="24"
          fontWeight="bold"
          style={{
            filter: 'drop-shadow(0 0 10px rgba(255,255,255,0.8))'
          }}
        >
          {hdState.activeGates.length} GATES ACTIVE
        </text>
      </svg>
    </div>
  );
}

// =======================
// RESONANCE WAVES
// =======================

function ResonanceWaves({ elements, intensity, time }: any) {
  return (
    <svg
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        opacity: intensity * 0.5
      }}
    >
      {elements.slice(0, 5).map((element: any, i: number) => {
        const freq = element.frequency || 1;
        const phase = time * freq * 0.5 + i;
        
        return (
          <g key={i}>
            {/* Horizontal wave */}
            <path
              d={generateWavePath(phase, 'horizontal', i)}
              stroke={`hsla(${(i * 60 + time * 30) % 360}, 80%, 60%, ${intensity})`}
              strokeWidth="2"
              fill="none"
              style={{
                filter: 'blur(1px) drop-shadow(0 0 5px currentColor)'
              }}
            />
            
            {/* Vertical wave */}
            <path
              d={generateWavePath(phase + Math.PI / 2, 'vertical', i)}
              stroke={`hsla(${(i * 60 + time * 30 + 180) % 360}, 80%, 60%, ${intensity * 0.5})`}
              strokeWidth="2"
              fill="none"
              style={{
                filter: 'blur(1px) drop-shadow(0 0 5px currentColor)'
              }}
            />
          </g>
        );
      })}
    </svg>
  );
}

function generateWavePath(phase: number, direction: 'horizontal' | 'vertical', offset: number): string {
  const points: string[] = [];
  const steps = 50;
  
  if (direction === 'horizontal') {
    for (let i = 0; i <= steps; i++) {
      const x = (i / steps) * window.innerWidth;
      const y = window.innerHeight / 2 + 
                Math.sin(phase + (i / steps) * Math.PI * 4) * 100 +
                offset * 30;
      points.push(i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`);
    }
  } else {
    for (let i = 0; i <= steps; i++) {
      const y = (i / steps) * window.innerHeight;
      const x = window.innerWidth / 2 + 
                Math.sin(phase + (i / steps) * Math.PI * 4) * 100 +
                offset * 30;
      points.push(i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`);
    }
  }
  
  return points.join(' ');
}

// =======================
// EMOTIONAL FIELD
// =======================

function EmotionalField({ mood, time }: any) {
  // Map mood to visual properties
  const hue = ((mood.valence + 1) / 2) * 120; // Red to green
  const saturation = mood.arousal * 100;
  const particles = Math.floor(mood.arousal * 50) + 10;
  
  return (
    <div style={{
      position: 'absolute',
      top: 0,
      left: 0,
      width: '100%',
      height: '100%',
      pointerEvents: 'none',
      overflow: 'hidden'
    }}>
      {Array.from({ length: particles }).map((_, i) => {
        const x = (Math.sin(time * 0.5 + i) * 0.5 + 0.5) * 100;
        const y = (Math.cos(time * 0.3 + i * 0.5) * 0.5 + 0.5) * 100;
        const size = 5 + Math.sin(time * 2 + i) * 3;
        
        return (
          <div
            key={i}
            style={{
              position: 'absolute',
              left: `${x}%`,
              top: `${y}%`,
              width: `${size}px`,
              height: `${size}px`,
              borderRadius: '50%',
              background: `hsla(${hue}, ${saturation}%, 60%, 0.6)`,
              filter: `blur(${mood.dominance * 5}px)`,
              boxShadow: `0 0 ${10 + mood.arousal * 20}px hsla(${hue}, ${saturation}%, 60%, 0.8)`
            }}
          />
        );
      })}
    </div>
  );
}

// =======================
// DRIVE MANDALAS
// =======================

function DriveMandalas({ drives, element }: any) {
  const driveArray = Object.entries(drives);
  
  return (
    <div style={{
      position: 'absolute',
      bottom: 50,
      left: '50%',
      transform: 'translateX(-50%)',
      display: 'flex',
      gap: '20px'
    }}>
      {driveArray.map(([name, value]: [string, any], i) => (
        <div
          key={name}
          style={{
            width: '80px',
            height: '80px',
            borderRadius: '50%',
            background: `conic-gradient(
              hsla(${i * 50}, 80%, 60%, ${value}) 0deg ${value * 360}deg,
              rgba(0,0,0,0.2) ${value * 360}deg 360deg
            )`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative',
            border: '2px solid rgba(255,255,255,0.3)',
            boxShadow: `0 0 20px hsla(${i * 50}, 80%, 60%, ${value * 0.8})`
          }}
        >
          <div style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
            background: 'rgba(0,0,0,0.8)',
            width: '60px',
            height: '60px',
            borderRadius: '50%',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center'
          }}>
            <div style={{ fontSize: '10px', color: 'white', opacity: 0.7 }}>
              {name}
            </div>
            <div style={{ fontSize: '16px', color: 'white', fontWeight: 'bold' }}>
              {Math.floor(value * 100)}%
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

// =======================
// THOUGHT STREAM
// =======================

function ThoughtStream({ agent, traits }: any) {
  const [thoughts, setThoughts] = useState<string[]>([]);
  
  useEffect(() => {
    const generateThought = () => {
      const possibleThoughts = [
        agent.currentAction ? `I am ${agent.currentAction.type.toLowerCase()}...` : null,
        agent.isAtHome ? "I feel safe here." : "I should head home soon.",
        traits.creativity > 0.7 ? "I want to create something beautiful." : null,
        traits.intuition > 0.7 ? "Something feels... different." : null,
        traits.energyLevel > 1.5 ? "I have so much energy!" : null,
        traits.energyLevel < 0.5 ? "I need to rest..." : null,
        agent.nearbyAgents.length > 0 ? `There are ${agent.nearbyAgents.length} others nearby.` : "I am alone.",
        "What is my purpose?",
        "The resonance is strong today.",
        `My ${agent.getHDChart().currentArchetype} nature guides me.`
      ].filter(Boolean) as string[];
      
      const thought = possibleThoughts[Math.floor(Math.random() * possibleThoughts.length)];
      
      setThoughts(prev => {
        const newThoughts = [thought, ...prev].slice(0, 5);
        return newThoughts;
      });
    };
    
    const interval = setInterval(generateThought, 3000);
    generateThought(); // Initial thought
    
    return () => clearInterval(interval);
  }, [agent, traits]);
  
  return (
    <div style={{
      position: 'absolute',
      top: 100,
      left: 50,
      maxWidth: '400px'
    }}>
      {thoughts.map((thought, i) => (
        <div
          key={i}
          style={{
            background: 'rgba(0,0,0,0.7)',
            color: 'white',
            padding: '10px 15px',
            margin: '10px 0',
            borderRadius: '10px',
            fontSize: '14px',
            opacity: 1 - i * 0.15,
            transform: `translateX(${i * 10}px)`,
            transition: 'all 0.5s',
            border: '1px solid rgba(255,255,255,0.2)',
            fontStyle: 'italic'
          }}
        >
          {thought}
        </div>
      ))}
    </div>
  );
}

// =======================
// PERCEPTION FILTERS
// =======================

function PerceptionFilters({ effects }: any) {
  return (
    <>
      {/* Color filter overlay */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        background: effects.colorFilter,
        mixBlendMode: 'multiply',
        opacity: 0.3,
        pointerEvents: 'none'
      }} />
      
      {/* Vignette */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        background: 'radial-gradient(circle, transparent 30%, rgba(0,0,0,0.8) 100%)',
        pointerEvents: 'none',
        opacity: effects.focusLevel
      }} />
    </>
  );
}

// =======================
// CONSCIOUSNESS STATS
// =======================

function ConsciousnessStats({ agent }: any) {
  const state = agent.getConsciousnessState();
  
  return (
    <div style={{
      position: 'absolute',
      top: 80,
      right: 20,
      background: 'rgba(0,0,0,0.8)',
      padding: '15px',
      borderRadius: '10px',
      color: 'white',
      fontSize: '12px',
      border: '1px solid rgba(255,255,255,0.3)',
      minWidth: '200px'
    }}>
      <div style={{ fontSize: '16px', marginBottom: '10px', fontWeight: 'bold' }}>
        {agent.name}'s Consciousness
      </div>
      <div style={{ marginBottom: '5px' }}>
        Level: {(state.hdState.consciousnessLevel * 100).toFixed(0)}%
      </div>
      <div style={{ marginBottom: '5px' }}>
        Resonance: {(state.resonanceField * 100).toFixed(0)}%
      </div>
      <div style={{ marginBottom: '5px' }}>
        Element: {ElementType[state.currentElement]}
      </div>
      <div style={{ marginBottom: '5px' }}>
        Archetype: {state.hdState.currentArchetype}
      </div>
      <div style={{ marginBottom: '5px', opacity: 0.7, fontSize: '10px', marginTop: '10px' }}>
        Active Codons: {state.activeCodonElements.length}
      </div>
    </div>
  );
}

// =======================
// HELPER FUNCTIONS
// =======================

function getColorFilter(mood: any, element: ElementType): string {
  const hue = ((mood.valence + 1) / 2) * 120;
  
  const elementColors = {
    [ElementType.EARTH]: '40, 20%',
    [ElementType.WATER]: '200, 60%',
    [ElementType.AIR]: '180, 40%',
    [ElementType.FIRE]: '15, 70%',
    [ElementType.AETHER]: '280, 50%'
  };
  
  const [elementHue, elementSat] = elementColors[element].split(', ');
  
  return `linear-gradient(135deg, 
    hsla(${hue}, 60%, 20%, 0.3),
    hsla(${elementHue}, ${elementSat}, 20%, 0.4)
  )`;
}

function getEmotionalOverlay(mood: any): string {
  if (mood.valence < -0.5) return 'rgba(200, 50, 50, 0.2)'; // Sad = red
  if (mood.valence > 0.5) return 'rgba(50, 200, 120, 0.2)'; // Happy = green
  return 'rgba(100, 100, 100, 0.1)'; // Neutral = gray
}
