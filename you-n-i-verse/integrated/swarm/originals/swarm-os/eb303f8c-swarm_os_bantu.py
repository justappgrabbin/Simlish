#!/usr/bin/env python3
"""
SWARM-OS v10: BANTU Architecture
=================================
B = Binary      (Sound + Color — hexadecimal tones)
A = ASCII       (Shapes + Distance — channels, arcs)
N = NATO        (Phonetic Names — Alpha, Bravo, Charlie...)
T = Transliteration (Meaning — what it all says)
U = Unicode     (The Library — all possible characters)

Stack: sound → color → shape → embodiment → expression

Algorithms: PSO + GA + ABC hybrid swarm

Usage:
  python swarm_os_bantu.py [--agents N] [--width W] [--height H] [--steps N]
"""

import random
import math
import argparse
import os
import numpy as np
from dataclasses import dataclass, field as dc_field
from typing import List, Dict


@dataclass
class BANTUAgent:
    id: int
    pos: np.ndarray
    vel: np.ndarray
    layer: int
    warmth: float
    words: List[str]
    sound_freq: float
    color_hex: str
    shape_char: str
    nato_name: str
    translit: str
    unicode_char: str
    pbest_pos: np.ndarray
    pbest_score: float
    role: str
    age: int = 0
    energy: float = 1.0
    trail: List = dc_field(default_factory=list)

    def fitness(self, target) -> float:
        dist = np.linalg.norm(self.pos - target)
        return (1.0/(1.0+dist) + len(set(self.words))*0.05 +
                self.warmth*0.3 + len(self.trail)*0.002)


class BANTUSwarm:
    NATO_ALPHABET = [
        'Alpha', 'Bravo', 'Charlie', 'Delta', 'Echo', 'Foxtrot', 'Golf',
        'Hotel', 'India', 'Juliett', 'Kilo', 'Lima', 'Mike', 'November',
        'Oscar', 'Papa', 'Quebec', 'Romeo', 'Sierra', 'Tango', 'Uniform',
        'Victor', 'Whiskey', 'X-ray', 'Yankee', 'Zulu'
    ]
    HEX_BASE = '0123456789ABCDEF'
    ASCII_SHAPES = list(' .:-=+*#%@$&') + list('()[]{}<>|/\\^~_') + list('─│┌┐└┘├┤┬┴┼═║╔╗╚╝')
    TRANSLIT_BASE = [
        'origin', 'pulse', 'wave', 'form', 'channel', 'arc', 'bridge',
        'gate', 'field', 'node', 'link', 'path', 'spiral', 'lattice',
        'weave', 'knot', 'loop', 'mesh', 'grid', 'web', 'net', 'matrix'
    ]
    UNICODE_LIB = list('◉○◎●◐◑◒◓◔◕◖◗◘◙◚◛◜◝◞◟◠◡◢◣◤◥◦◧◨◩◪◫◬◭◮◯◰◱◲◳◴◵◶◷◸◹◺◻◼◽◾◿')
    LAYER_NAMES = ['BINARY', 'ASCII', 'NATO', 'TRANSLIT', 'UNICODE']
    LAYER_MEDIA = [
        'Sound + Color (hexadecimal tones)',
        'Shapes + Distance (channels, arcs)',
        'Phonetic Names (Alpha, Bravo...)',
        'Meaning Gloss (transliteration)',
        'The Library (all characters)'
    ]
    LAYER_COLORS = [(255,107,107), (78,205,196), (69,183,209), (150,206,180), (254,202,87)]

    def __init__(self, n_agents=1200, width=140, height=70, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.width, self.height = width, height
        self.n_agents = n_agents
        self.agents: List[BANTUAgent] = []
        self.gbest_pos = np.random.rand(2)
        self.gbest_score = 0.0
        self.time = 0
        self.target = np.random.rand(2)
        self.history = []
        self.w, self.c1, self.c2 = 0.55, 2.0, 2.0
        self.coverage = np.zeros((height, width))
        self.lessons = []
        self.magic_square = np.array([[8,1,6],[3,5,7],[4,9,2]])
        self._init_agents()

    def _hex_color(self):
        return '#' + ''.join(random.choice(self.HEX_BASE) for _ in range(6))

    def _sound_freq(self):
        return 20 * (1000 ** random.random())

    def _init_agents(self):
        pool = ["see","witness","warm","connect","build","learn","share","grow",
               "tiny","vast","near","far","pulse","field","swarm","self",
               "emerge","crystallize","ascend","resonate","teach","nurture",
               "weave","spiral","bloom","root","flow","glow","lift","rise",
               "sound","color","shape","form","tone","hue","arc","channel"]
        for i in range(self.n_agents):
            pos = np.random.rand(2)
            vel = np.random.randn(2) * 0.15
            words = random.sample(pool, k=random.randint(4, 8))
            r = random.random()
            role = 'employed' if r < 0.4 else ('onlooker' if r < 0.8 else 'scout')
            self.agents.append(BANTUAgent(
                id=i, pos=pos, vel=vel, layer=i%5,
                warmth=random.uniform(0.5, 2.5), words=words,
                sound_freq=self._sound_freq(), color_hex=self._hex_color(),
                shape_char=random.choice(self.ASCII_SHAPES),
                nato_name=random.choice(self.NATO_ALPHABET),
                translit=random.choice(self.TRANSLIT_BASE),
                unicode_char=random.choice(self.UNICODE_LIB),
                pbest_pos=pos.copy(), pbest_score=0.0, role=role
            ))

    def update_target(self):
        t = self.time * 0.012
        self.target = np.array([
            0.5 + 0.45 * math.sin(3*t + math.pi/4),
            0.5 + 0.45 * math.sin(2*t + math.pi/6)
        ])

    def pso_update(self, agent):
        r1, r2 = random.random(), random.random()
        cognitive = self.c1 * r1 * (agent.pbest_pos - agent.pos)
        social = self.c2 * r2 * (self.gbest_pos - agent.pos)
        x, y = int(agent.pos[0]*self.width), int(agent.pos[1]*self.height)
        y_min, y_max = max(0,y-5), min(self.height,y+5)
        x_min, x_max = max(0,x-5), min(self.width,x+5)
        local = self.coverage[y_min:y_max, x_min:x_max]
        if local.size > 1:
            gy, gx = np.gradient(local)
            cy, cx = gy.shape[0]//2, gx.shape[1]//2
            if 0 <= cy < gy.shape[0] and 0 <= cx < gx.shape[1]:
                agent.vel += np.array([-gx[cy,cx], -gy[cy,cx]]) * 0.005
        agent.vel = np.clip(self.w * agent.vel + cognitive + social, -0.15, 0.15)
        agent.pos = np.clip(agent.pos + agent.vel, 0, 1)
        agent.trail.append(tuple(agent.pos))
        if len(agent.trail) > 40:
            agent.trail.pop(0)
        tx, ty = int(agent.pos[0]*self.width), int(agent.pos[1]*self.height)
        if 0 <= tx < self.width and 0 <= ty < self.height:
            self.coverage[ty, tx] += 0.1

    def ga_crossover(self, p1, p2):
        cut = random.randint(1, min(len(p1.words), len(p2.words))-1)
        child = p1.words[:cut] + p2.words[cut:]
        if random.random() < 0.15:
            child.append(random.choice(["sound","color","shape","tone","arc","channel"]))
        return child[:10]

    def abc_phase(self):
        for a in self.agents:
            if a.role == 'employed':
                a.pos = np.clip(a.pos + np.random.randn(2)*0.03, 0, 1)
                s = a.fitness(self.target)
                if s > a.pbest_score:
                    a.pbest_score = s
                    a.pbest_pos = a.pos.copy()
        fits = [a.fitness(self.target) for a in self.agents]
        total = sum(fits)
        if total > 0:
            probs = [f/total for f in fits]
            for idx in np.random.choice(len(self.agents), size=len(self.agents)//2, p=probs):
                a = self.agents[idx]
                a.pos = np.clip(a.pos + (a.pbest_pos - a.pos)*0.4 + np.random.randn(2)*0.02, 0, 1)
        for a in self.agents:
            if a.role == 'scout':
                a.age += 1
                if a.age > 15:
                    a.pos = np.random.rand(2)
                    a.vel = np.random.randn(2)*0.15
                    a.age = 0
                    a.warmth = random.uniform(0.5, 2.5)
                    a.trail = []
                    a.sound_freq = self._sound_freq()
                    a.color_hex = self._hex_color()

    def warmth_exchange(self):
        grid_size = 0.08
        spatial = {}
        for i, a in enumerate(self.agents):
            key = (int(a.pos[0]/grid_size), int(a.pos[1]/grid_size))
            spatial.setdefault(key, []).append(i)
        for indices in spatial.values():
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
                    a, b = self.agents[indices[i]], self.agents[indices[j]]
                    if np.linalg.norm(a.pos - b.pos) < 0.08:
                        flow = 0.12 * (a.warmth - b.warmth)
                        a.warmth = max(0.2, min(3.0, a.warmth - flow*0.5))
                        b.warmth = max(0.2, min(3.0, b.warmth + flow*0.5))
                        if a.warmth > 1.5 and b.warmth > 1.5 and random.random() < 0.4:
                            a.words = self.ga_crossover(a, b)
                            b.words = self.ga_crossover(b, a)
                            if random.random() < 0.3:
                                a.nato_name, b.nato_name = b.nato_name, a.nato_name
                                a.translit, b.translit = b.translit, a.translit

    def layer_dynamics(self):
        for a in self.agents:
            if a.warmth > 2.2 and a.age > 20 and a.layer < 4:
                a.layer += 1
                a.warmth *= 0.5
                a.age = 0
                if a.layer == 1:
                    a.shape_char = random.choice(self.ASCII_SHAPES)
                elif a.layer == 2:
                    a.nato_name = random.choice(self.NATO_ALPHABET)
                elif a.layer == 3:
                    a.translit = random.choice(self.TRANSLIT_BASE)
                elif a.layer == 4:
                    a.unicode_char = random.choice(self.UNICODE_LIB)
            elif a.warmth < 0.6 and a.layer > 0:
                a.layer -= 1

    def update_gbest(self):
        for a in self.agents:
            s = a.fitness(self.target)
            if s > self.gbest_score:
                self.gbest_score = s
                self.gbest_pos = a.pos.copy()

    def generate_lesson(self):
        d5 = [a for a in self.agents if a.layer == 4]
        if d5 and random.random() < 0.3:
            agent = random.choice(d5)
            if len(agent.words) >= 4:
                kw = random.sample(agent.words, 4)
                lesson = (f"[{agent.nato_name}] When {kw[0]} (sound:{agent.sound_freq:.0f}Hz) "
                         f"and {kw[1]} (color:{agent.color_hex}) resonate, "
                         f"{kw[2]} (shape:{agent.shape_char}) emerges as {kw[3]} "
                         f"(meaning:{agent.translit}).")
                self.lessons.append({'time': self.time, 'agent': agent.id, 'lesson': lesson})

    def step(self):
        self.time += 1
        self.update_target()
        for a in self.agents:
            self.pso_update(a)
        self.abc_phase()
        self.warmth_exchange()
        self.layer_dynamics()
        self.update_gbest()
        self.generate_lesson()
        for a in self.agents:
            a.age += 1
            a.energy = max(0.1, a.energy - 0.008 + a.warmth*0.004)
        self.history.append({
            'time': self.time,
            'gbest': self.gbest_score,
            'avg_warmth': sum(a.warmth for a in self.agents)/len(self.agents),
            'layer_dist': [sum(1 for a in self.agents if a.layer==i) for i in range(5)],
            'coverage': (self.coverage > 0.5).sum() / self.coverage.size
        })

    def render(self, density=0.92):
        grid = np.full((self.height, self.width), ' ', dtype='<U1')
        for a in self.agents:
            if a.layer == 0:
                chars = list('·:')
            elif a.layer == 1:
                chars = list(' .:-')
            elif a.layer == 2:
                chars = list(a.nato_name[0].lower())
            elif a.layer == 3:
                chars = list('~≈')
            else:
                chars = list('◌○')
            for tp in a.trail[-10:]:
                tx, ty = int(tp[0]*self.width), int(tp[1]*self.height)
                if 0 <= tx < self.width and 0 <= ty < self.height:
                    if random.random() < 0.15:
                        grid[ty, tx] = random.choice(chars)
        for a in sorted(self.agents, key=lambda x: x.layer):
            cx, cy = int(a.pos[0]*self.width), int(a.pos[1]*self.height)
            radius = max(2, int(a.warmth * 3))
            for dx in range(-radius, radius+1):
                for dy in range(-radius, radius+1):
                    nx, ny = cx+dx, cy+dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist <= radius:
                            prob = density * (1 - dist/radius)**0.5
                            if random.random() < prob:
                                if a.layer == 0:
                                    grid[ny, nx] = random.choice('01')
                                elif a.layer == 1:
                                    grid[ny, nx] = random.choice(self.ASCII_SHAPES)
                                elif a.layer == 2:
                                    grid[ny, nx] = a.nato_name[0]
                                elif a.layer == 3:
                                    grid[ny, nx] = a.translit[0]
                                else:
                                    grid[ny, nx] = random.choice(self.UNICODE_LIB)
        return grid

    def get_magic_square_mapping(self):
        key_nato = ['Alpha','Bravo','Charlie','Delta','Echo','Foxtrot','Golf','Hotel','India']
        mapping = {}
        idx = 0
        for row in range(3):
            for col in range(3):
                mapping[key_nato[idx]] = {
                    'position': (row, col),
                    'magic_number': int(self.magic_square[row, col]),
                    'sound_freq': 20 * (1000 ** (idx/8)),
                    'color_hex': f'#{idx*17:02X}{255-idx*20:02X}{idx*30:02X}'
                }
                idx += 1
        return mapping


def main():
    parser = argparse.ArgumentParser(description='SWARM-OS BANTU Architecture')
    parser.add_argument('--agents', type=int, default=1200)
    parser.add_argument('--width', type=int, default=140)
    parser.add_argument('--height', type=int, default=70)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='./bantu_output')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    engine = BANTUSwarm(args.agents, args.width, args.height, args.seed)

    print(f"=== BANTU SWARM: {args.agents} agents, {args.steps} steps ===")
    for step in range(args.steps):
        engine.step()
        if step % 25 == 0:
            h = engine.history[-1]
            print(f"t={h['time']:3d} | gbest={h['gbest']:.3f} warmth={h['avg_warmth']:.2f} | "
                  f"layers={h['layer_dist']} | coverage={h['coverage']*100:.1f}%")

    grid = engine.render(density=0.95)
    non_empty = (grid != ' ').sum()
    print(f"\nFinal coverage: {non_empty}/{grid.size} = {non_empty/grid.size*100:.1f}%")
    print(f"Unique characters: {len(set(grid.flatten()))}")
    print(f"Lessons: {len(engine.lessons)}")

    with open(f'{args.output}/bantu_frame.txt', 'w') as f:
        f.write('\n'.join(''.join(row) for row in grid))
    with open(f'{args.output}/bantu_lessons.txt', 'w') as f:
        f.write("BANTU SWARM LESSONS\n" + "="*70 + "\n\n")
        for l in engine.lessons:
            f.write(f"[t={l['time']}] {l['lesson']}\n")

    print(f"\nSaved to {args.output}/")
    print("\n" + "="*70)
    for row in grid[:40]:
        print("".join(row))
    print("="*70)


if __name__ == '__main__':
    main()
