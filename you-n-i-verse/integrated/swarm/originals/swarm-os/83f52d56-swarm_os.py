#!/usr/bin/env python3
"""
SWARM-OS: Dimensional Agent Swarm Engine
========================================
Tiny eyes carrying words self-organize into neighborhoods,
exchange warmth, ascend a 5-dimensional stack, and crystallize lessons.

Dimensional Stack:
  D1 = Impulse    (raw movement, desire)
  D2 = Polarity   (attraction/repulsion)
  D3 = Witness    (observation, awareness)
  D4 = Context    (neighborhoods, structure)
  D5 = Meaning    (lessons, crystallization)

Usage:
  python swarm_os.py [--agents N] [--steps N] [--seed S] [--output DIR]
"""

import random
import argparse
import json
import os
from datetime import datetime


class DimensionalAgent:
    """A tiny eye — minimal perceiving agent in 5-D space."""

    def __init__(self, agent_id, position_5d, word_vector, warmth=1.0):
        self.id = agent_id
        self.pos = list(position_5d)  # [D1, D2, D3, D4, D5]
        self.words = list(word_vector)
        self.warmth = warmth
        self.neighbors = []
        self.layer = 0
        self.energy = random.uniform(0.5, 2.0)
        self.age = 0
        self.lessons_generated = []
        self.word_origin = {}

    def sense(self, field):
        """D3 witness: sense local field gradient."""
        return field.gradient_at(self.pos)

    def move(self, force_5d, dt=0.1):
        """D1 impulse: respond to force with energy-modulated amplitude."""
        amplitude = 0.12 * (1 + self.energy)
        for i in range(5):
            self.pos[i] += force_5d[i] * amplitude * dt
            self.pos[i] = max(-6, min(6, self.pos[i]))

    def polarize(self, other):
        """D2 polarity: compute attraction/repulsion force."""
        diff = [a - b for a, b in zip(self.pos, other.pos)]
        dist = sum(d*d for d in diff) ** 0.5
        warmth_attraction = (self.warmth * other.warmth) / (1 + dist)
        similarity = len(set(self.words) & set(other.words)) / max(
            len(set(self.words) | set(other.words)), 1)
        repulsion = similarity * 0.4
        return warmth_attraction - repulsion, diff

    def exchange_warmth(self, other):
        """Warminess exchange: diffusion + noise + bounds."""
        diff = self.warmth - other.warmth
        flow = 0.15 * diff
        self.warmth -= flow * 0.5
        other.warmth += flow * 0.5
        self.warmth += random.gauss(0, 0.015)
        other.warmth += random.gauss(0, 0.015)
        self.warmth = max(0.15, min(3.0, self.warmth))
        other.warmth = max(0.15, min(3.0, other.warmth))

    def share_words(self, other):
        """Distribute words to resonant neighbors."""
        shared = set(self.words) & set(other.words)
        resonance = len(shared) / max(len(set(self.words) | set(other.words)), 1)
        if resonance > 0.15:
            new_for_other = [w for w in self.words if w not in other.words][:2]
            new_for_self = [w for w in other.words if w not in self.words][:2]
            other.words.extend(new_for_other)
            self.words.extend(new_for_self)
            for w in new_for_other:
                other.word_origin[w] = self.id
            for w in new_for_self:
                self.word_origin[w] = other.id
            return True
        return False

    def burn_energy(self):
        self.energy = max(0.1, self.energy - 0.02)
        self.energy += self.warmth * 0.008
        self.age += 1

    def generate_lesson(self):
        """D5 meaning: crystallize experience into lesson."""
        if self.layer == 4 and len(self.words) > 5:
            key_words = random.sample(self.words, min(4, len(self.words)))
            lesson = f"When {' and '.join(key_words[:2])} resonate, {key_words[2]} emerges."
            if len(key_words) > 3:
                lesson += f" This is how {key_words[3]} teaches."
            self.lessons_generated.append(lesson)
            return lesson
        return None


class SwarmField:
    """The 5-layer field that agents inhabit and ascend."""

    LAYER_NAMES = ['D1 Impulse', 'D2 Polarity', 'D3 Witness', 
                   'D4 Context', 'D5 Meaning']
    LAYER_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']

    def __init__(self, size=5):
        self.layers = [[] for _ in range(size)]
        self.history = []
        self.time = 0
        self.births = 0
        self.lessons = []

    def add_agent(self, agent, layer=0):
        self.layers[layer].append(agent)
        agent.layer = layer

    def all_agents(self):
        return [a for layer in self.layers for a in layer]

    def gradient_at(self, pos_5d):
        grad = [0.0] * 5
        for layer_agents in self.layers:
            for agent in layer_agents:
                diff = [p - a for p, a in zip(pos_5d, agent.pos)]
                dist = sum(d*d for d in diff) ** 0.5 + 0.01
                strength = agent.warmth / (dist ** 2)
                for i in range(5):
                    grad[i] += strength * diff[i] * 0.06
        return grad

    def inject_energy(self):
        agents = self.all_agents()
        if agents:
            for a in random.sample(agents, min(4, len(agents))):
                a.energy += random.uniform(0.3, 0.7)

    def replenish_warmth(self):
        """Sun: inject warmth to compensate for diffusion loss."""
        agents = self.all_agents()
        if agents:
            for a in random.sample(agents, min(8, len(agents))):
                a.warmth += random.uniform(0.15, 0.35)

    def spawn_from_warm_cluster(self):
        """Birth: warm clusters spawn new agents."""
        agents = self.all_agents()
        warm = [a for a in agents if a.warmth > 1.5]
        if warm and len(agents) < 80:
            parent = random.choice(warm)
            child_pos = [p + random.gauss(0, 0.4) for p in parent.pos]
            child_words = random.sample(parent.words, min(3, len(parent.words)))
            child = DimensionalAgent(f"eye_b{self.births}", child_pos, 
                                    child_words, warmth=parent.warmth * 0.4)
            self.add_agent(child, layer=0)
            self.births += 1

    def step(self, dt=0.1):
        self.time += 1
        agents = self.all_agents()

        # Environmental drivers
        if self.time % 3 == 0:
            self.inject_energy()
            self.replenish_warmth()
        if self.time % 8 == 0:
            self.spawn_from_warm_cluster()

        # Movement (D1)
        for agent in agents:
            grad = agent.sense(self)
            impulse = [g + random.gauss(0, 0.12) for g in grad]
            agent.move(impulse, dt)
            agent.burn_energy()

        # Neighborhood formation (D4)
        for agent in agents:
            distances = []
            for other in agents:
                if agent is other:
                    continue
                _, diff = agent.polarize(other)
                dist = sum(d*d for d in diff) ** 0.5
                distances.append((dist, other))
            distances.sort()
            agent.neighbors = [o for _, o in distances[:4]]

        # Exchange (D2 + warmth + words)
        for agent in agents:
            for neighbor in agent.neighbors:
                agent.exchange_warmth(neighbor)
                agent.share_words(neighbor)

        # Ascension/descension through layers
        for agent in agents[:]:
            if (agent.warmth > 1.5 and agent.age > 10 and 
                agent.energy < 1.0 and agent.layer < 4):
                self.layers[agent.layer].remove(agent)
                agent.layer += 1
                self.layers[agent.layer].append(agent)
                agent.warmth *= 0.6
                agent.age = 0
            elif agent.warmth < 0.5 and agent.layer > 0:
                self.layers[agent.layer].remove(agent)
                agent.layer -= 1
                self.layers[agent.layer].append(agent)

        # D5 lesson generation
        for agent in self.layers[4]:
            if random.random() < 0.4:
                lesson = agent.generate_lesson()
                if lesson:
                    self.lessons.append({
                        'time': self.time,
                        'agent': agent.id,
                        'lesson': lesson
                    })

        self.history.append({
            'time': self.time,
            'avg_warmth': sum(a.warmth for a in agents) / len(agents),
            'layer_dist': [len(l) for l in self.layers],
            'total_agents': len(agents),
            'births': self.births,
            'lessons': len(self.lessons)
        })


class NeighborhoodCrystallizer:
    """Renders swarm state into pictures, videos, and lessons."""

    def __init__(self, field):
        self.field = field

    def render_picture(self, filename=None, show=True):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        colors = SwarmField.LAYER_COLORS
        labels = SwarmField.LAYER_NAMES

        # 1: Spatial map
        ax = axes[0,0]
        for li, layer in enumerate(self.field.layers):
            if not layer:
                continue
            xs = [a.pos[0] for a in layer]
            ys = [a.pos[1] for a in layer]
            ws = [a.warmth * 60 for a in layer]
            ax.scatter(xs, ys, s=ws, c=colors[li], alpha=0.6,
                      label=f"{labels[li]} ({len(layer)})",
                      edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Impulse (D1)')
        ax.set_ylabel('Polarity (D2)')
        ax.set_title('Swarm Neighborhoods')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2: Layer evolution
        ax = axes[0,1]
        h = self.field.history
        times = [s['time'] for s in h]
        for i in range(5):
            ax.plot(times, [s['layer_dist'][i] for s in h],
                   color=colors[i], label=labels[i], linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Agents')
        ax.set_title('Dimensional Ascension')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3: Warmth by layer
        ax = axes[0,2]
        agents = self.field.all_agents()
        ax.scatter([a.layer for a in agents], [a.warmth for a in agents],
                  c=[colors[a.layer] for a in agents], s=80, alpha=0.6,
                  edgecolors='black')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Warmth')
        ax.set_title('Warmth by Layer')
        ax.set_xticks(range(5))
        ax.set_xticklabels([l.replace(' ', '\n') for l in labels])
        ax.grid(True, alpha=0.3)

        # 4: Population
        ax = axes[1,0]
        ax.plot(times, [s['total_agents'] for s in h], 'b-', label='Population', linewidth=2)
        ax.plot(times, [s['births'] for s in h], 'g--', label='Births', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Count')
        ax.set_title('Swarm Growth')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5: Lessons
        ax = axes[1,1]
        ax.plot(times, [s['lessons'] for s in h], 'purple', linewidth=2)
        ax.fill_between(times, [s['lessons'] for s in h], alpha=0.3, color='purple')
        ax.set_xlabel('Time')
        ax.set_ylabel('Lessons')
        ax.set_title('Meaning Crystallization (D5)')
        ax.grid(True, alpha=0.3)

        # 6: Vocabulary
        ax = axes[1,2]
        layer_words = {}
        for li, layer in enumerate(self.field.layers):
            words = []
            for a in layer:
                words.extend(a.words)
            layer_words[li] = len(set(words))
        ax.bar(range(5), [layer_words.get(i, 0) for i in range(5)],
               color=colors, edgecolor='black')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Unique Words')
        ax.set_title('Vocabulary Diversity')
        ax.set_xticks(range(5))
        ax.set_xticklabels([l.replace(' ', '\n') for l in labels])
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved: {filename}")
        if show:
            plt.show()
        plt.close()

    def render_video(self, output_dir, fps=10):
        """Generate video frames showing swarm evolution."""
        import matplotlib.pyplot as plt

        frames_dir = os.path.join(output_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)

        h = self.field.history
        n_frames = min(50, len(h))
        step = max(1, len(h) // n_frames)

        colors = SwarmField.LAYER_COLORS
        labels = SwarmField.LAYER_NAMES

        for fi, hi in enumerate(range(0, len(h), step)):
            fig, ax = plt.subplots(figsize=(10, 8))

            # Reconstruct approximate positions from history
            # (In full implementation, store positions in history)
            for li, layer in enumerate(self.field.layers):
                if not layer:
                    continue
                xs = [a.pos[0] + random.gauss(0, 0.1) for a in layer]
                ys = [a.pos[1] + random.gauss(0, 0.1) for a in layer]
                ws = [a.warmth * 50 for a in layer]
                ax.scatter(xs, ys, s=ws, c=colors[li], alpha=0.6,
                          label=labels[li], edgecolors='black', linewidth=0.5)

            ax.set_xlim(-7, 7)
            ax.set_ylim(-7, 7)
            ax.set_xlabel('Impulse (D1)')
            ax.set_ylabel('Polarity (D2)')
            ax.set_title(f'Swarm Evolution — t={h[hi]["time"]}')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.savefig(os.path.join(frames_dir, f'frame_{fi:04d}.png'),
                       dpi=100, bbox_inches='tight')
            plt.close()

        print(f"Generated {fi+1} frames in {frames_dir}")
        print("To compile: ffmpeg -framerate 10 -i frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p swarm.mp4")

    def extract_lesson(self):
        h = self.field.history
        if len(h) < 2:
            return {"status": "insufficient data"}

        fl = h[-1]['layer_dist']

        if fl[4] > 0:
            emergence = "MEANING CRYSTALLIZATION"
            lesson = "D5 agents are generating lessons from swarm experience."
        elif fl[3] > 0:
            emergence = "CONTEXT STABILIZATION"
            lesson = "Persistent neighborhoods have formed in D4."
        elif fl[2] > 0:
            emergence = "WITNESS EMERGENCE"
            lesson = "Agents in D3 are observing swarm dynamics."
        elif fl[1] > 0:
            emergence = "POLARITY ORGANIZATION"
            lesson = "Attraction/repulsion structuring the swarm."
        else:
            emergence = "IMPULSE EXPLORATION"
            lesson = "Swarm is in raw exploration phase."

        return {
            'emergence': emergence,
            'lesson': lesson,
            'final_state': {
                'population': h[-1]['total_agents'],
                'layer_distribution': fl,
                'avg_warmth': round(h[-1]['avg_warmth'], 3),
                'births': h[-1]['births'],
                'lessons_generated': h[-1]['lessons']
            },
            'sample_lessons': [l['lesson'] for l in self.field.lessons[:10]]
        }


def run_swarm(n_agents=40, n_steps=250, seed=42, output_dir='./swarm_output'):
    """Run a complete swarm simulation."""
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== SWARM-OS: {n_agents} agents, {n_steps} steps, seed={seed} ===")

    field = SwarmField(size=5)

    words_pool = [
        "see", "witness", "warm", "connect", "build", "learn", "share", "grow",
        "tiny", "vast", "near", "far", "pulse", "field", "swarm", "self",
        "emerge", "crystallize", "ascend", "resonate", "teach", "nurture",
        "weave", "spiral", "bloom", "root", "flow", "glow"
    ]

    for i in range(n_agents):
        pos = [random.gauss(0, 1.5) for _ in range(5)]
        words = random.sample(words_pool, k=5)
        agent = DimensionalAgent(f"eye_{i:03d}", pos, words,
                                warmth=random.uniform(0.5, 2.0))
        field.add_agent(agent, layer=0)

    for step in range(n_steps):
        field.step()
        if step % 50 == 0:
            s = field.history[-1]
            print(f"t={s['time']:3d} | pop={s['total_agents']:2d} "
                  f"warmth={s['avg_warmth']:.2f} | layers={s['layer_dist']} "
                  f"| births={s['births']} lessons={s['lessons']}")

    crystal = NeighborhoodCrystallizer(field)
    result = crystal.extract_lesson()

    print(f"\n=== RESULT ===")
    print(f"Emergence: {result['emergence']}")
    print(f"Lesson: {result['lesson']}")
    print(f"State: {result['final_state']}")

    # Save outputs
    pic_file = os.path.join(output_dir, 'swarm_state.png')
    crystal.render_picture(pic_file, show=False)

    json_file = os.path.join(output_dir, 'swarm_result.json')
    with open(json_file, 'w') as f:
        json.dump(result, f, indent=2)

    lessons_file = os.path.join(output_dir, 'lessons.txt')
    with open(lessons_file, 'w') as f:
        f.write(f"SWARM-OS Lessons — {datetime.now()}\n")
        f.write(f"Emergence: {result['emergence']}\n\n")
        for l in field.lessons:
            f.write(f"[t={l['time']}] {l['agent']}: {l['lesson']}\n")

    print(f"\nOutputs saved to {output_dir}/")
    print(f"  - swarm_state.png")
    print(f"  - swarm_result.json")
    print(f"  - lessons.txt")

    return field, result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SWARM-OS Dimensional Swarm Simulator')
    parser.add_argument('--agents', type=int, default=40, help='Initial agent count')
    parser.add_argument('--steps', type=int, default=250, help='Simulation steps')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='./swarm_output', help='Output directory')
    args = parser.parse_args()

    run_swarm(args.agents, args.steps, args.seed, args.output)
