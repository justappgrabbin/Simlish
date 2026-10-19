#!/usr/bin/env python3
"""Read-only extraction of Alexis's own HD source material into data/her-sources.json.

Run on the Synthia box (paths below are box-local copies; originals untouched):
  python3 tools/extract-her-sources.py && node tools/extract-synthai-rule.mjs
Nothing here is required at runtime; the module only reads the generated JSON (as .mjs).
"""
import json, re, csv, hashlib, os, subprocess, sys
W = '/workspace'
P = {
  'blackbook': f'{W}/blackbook/bb_gates.json',
  'rave': f'{W}/rave2001/combined.txt',
  'rutt': f'{W}/zhouyi/rutt_hex.json',
  'kingwen': f'{W}/run-existing/foundation/Synthia-State-Space-Foundation/vendor/browser/src/merged/kingwen.js',
  'centersChannels': f'{W}/run-existing/foundation/Synthia-State-Space-Foundation/vendor/browser/src/merged/centers-channels.js',
  'hdjs': f'{W}/run-existing/foundation/Synthia-State-Space-Foundation/vendor/browser/src/state-space/human-design.js',
  'codex': f'{W}/scratch/hdm-src/drive/cb/HumanDesign_GateCodon_Codex_FULL.csv',
  'centersDoc': f'{W}/scratch/hdm-src/drive/cb/centers_from_doc.csv',
  'dnahd': f'{W}/scratch/hdm-src/drive/dnahd/dna-hd-compiler/src/hd/gates.ts',
}
sha = {k: hashlib.sha256(open(v, 'rb').read()).hexdigest() for k, v in P.items()}

CENTER_KEYS = [('G Center','G'),('Throat','Throat'),('The Base of the Spine','Root'),('Sacral','Sacral'),('Ajna','Ajna'),
               ('Solar Plexus','SolarPlexus'),('SOlar Plexus','SolarPlexus'),('Splenic','Spleen'),('Heart','Heart'),('The Head Center','Head')]
def center_id(s):
    for k, v in CENTER_KEYS:
        if s and s.startswith(k): return v
    return None

bb = json.load(open(P['blackbook']))
gates = {}
for g in bb:
    printed = g['centerAsPrinted']
    cid = center_id(printed) or center_id(g['center'])
    bio = g['biology']
    if bio:
        bio = re.sub(r'\s+', ' ', bio.replace('Li ver','Liver').replace('OVaries','Ovaries').replace('Bnd Perathyroio','and Parathyroid').replace('Kidney.','Kidney')).strip()
    gates[g['gate']] = {'gate': g['gate'], 'blackBook': {'center': cid, 'centerAsPrinted': printed or g['center'], 'centerFrom': 'printed' if printed else 'page-keyword', 'biologyAsPrinted': bio, 'pdfPage': g['pdfPage']}}

# Black Book center -> biology (majority of printed per-gate Biology fields)
from collections import Counter, defaultdict
cb = defaultdict(Counter)
for g in gates.values():
    b = g['blackBook']
    if b['biologyAsPrinted']: cb[b['center']][re.sub(r'^The ', '', b['biologyAsPrinted']).replace(' the ', ' ')] += 1
centerBiology = {c: {'biology': cnt.most_common(1)[0][0], 'votes': dict(cnt)} for c, cnt in cb.items()}

# Rave I'Ching (2001) "THE GATE OF ..." keynotes
L = open(P['rave']).read().split('\n')
gk_start = next(i for i, l in enumerate(L) if l.strip() == 'GENE KEYS')
for i, l in enumerate(L[:gk_start]):
    m = re.match(r'\s*THE GATE OF (.+?)\s*$', l)
    if m:
        n = re.search(r'(\d+)\s*v\s*(\d+)\s*w\s*(\d+)', L[i-1])
        gnum = int(n.group(2)) if n else None
        if gnum and 'raveKeynote' not in gates[gnum]:
            gates[gnum]['raveKeynote'] = 'The Gate of ' + m.group(1).title().replace(' Of ', ' of ').replace(' The ', ' the ')
# Rave I'Ching codon statements (raw lines, quoted verbatim, short)
raveCodonStatements = []
for i, l in enumerate(L[:gk_start]):
    if re.search(r'[Cc]odon', l):
        raveCodonStatements.append({'line': i + 1, 'text': re.sub(r'\s+', ' ', (L[i-1] + ' ' + l + ' ' + L[i+1])).strip()})

# Gene Keys (Rudd) per-key: codon ring, amino acid, physiology. Sequential 1..64 after the GENE KEYS title.
k = 0
for i in range(gk_start, len(L)):
    l = L[i]
    if 'CODON RING:' not in l: continue
    k += 1
    ctx = '\n'.join(L[i:i+4])
    ring = re.search(r'CODON RING:\s*(.*?)(\s{2,}|$)', l).group(1).strip()
    if ring in ('THE RING OF', 'THE RING OF LIFE AND'):
        nxt = L[i+1].strip().split('  ')[0].strip()
        ring = (ring + ' ' + nxt).strip() if not nxt.startswith('(') else ring + ' DEATH'
    aa = re.search(r'AMINO ACID:\s*([A-Z ]+?)(\s*\(|\s{2,}|$)', ctx, re.M)
    aa = aa.group(1).strip() if aa else None
    members = re.search(r'\((?:THE RING OF [A-Z]+ – )?([\d,\s]+)\)', ctx)
    members = [int(x) for x in re.findall(r'\d+', members.group(1))] if members else None
    # physiology is on the line(s) just above
    phys = None
    for j in (i-1, i-2):
        m = re.search(r'PHYSIOLOGY:\s*(.*)$', L[j])
        if m:
            phys = m.group(1).strip()
            if not phys:
                phys = L[j+1].strip().split('  ')[-1].strip()
            nxt = L[j+1].strip()
            if j == i-2 and nxt not in ('KEY',) and not nxt.startswith('CODON'):
                tail = nxt.split('  ')[-1].strip()
                if tail and tail != phys and tail != 'KEY': phys = (phys + ' ' + tail).strip()
            break
    gates[k]['geneKeys'] = {'codonRing': ring.title().replace(' Of ', ' of ').replace(' The ', ' the ').replace('Aslchemy','Alchemy').replace(' And ',' and ').replace('the Prosperity','Prosperity'),
                            'ringAsPrinted': ring, 'aminoAcidAsPrinted': aa, 'ringMembersAsPrinted': members, 'physiologyAsPrinted': phys, 'textLine': i + 1}

# Rutt Zhouyi header
rutt = {h['hex']: h for h in json.load(open(P['rutt']))}
for g in gates:
    h = rutt.get(g)
    if h: gates[g]['zhouyi'] = {'headerAsPrinted': h['header'], 'printedPage': h.get('printedPage')}

# King Wen names + Fu Xi decimal from her kingwen.js
src = open(P['kingwen']).read()
table = [int(x) for x in re.findall(r'\d+', re.search(r'const TABLE = Object.freeze\(\[(.*?)\]\)', src, re.S).group(1))]
names = re.findall(r"'([^']+)'", re.search(r'KING_WEN_HEXAGRAM_NAMES = Object.freeze\(\[(.*?)\]\)', src, re.S).group(1))
for g in gates:
    d = table[g-1]
    gates[g]['kingWen'] = {'name': names[g-1], 'fuXiDecimal': d, 'linesBottomFirst': ''.join(str((d >> b) & 1) for b in range(6))}

# her merged CENTERS table (centers-channels.js)
cc = open(P['centersChannels']).read()
mergedCenters = {m.group(1): [int(x) for x in re.findall(r'\d+', m.group(2))] for m in re.finditer(r'(\w+): Object.freeze\(\[([\d, ]+)\]\)', cc)}
channels = [[int(a), int(b)] for a, b in re.findall(r'\[(\d+), (\d+)\]', cc.split('CANONICAL_CHANNELS')[1].split('].map')[0])]

# donor human-design.js: GATE_CENTER + per-gate aminoAcid/organ
hd = open(P['hdjs']).read()
gc = re.search(r'export const GATE_CENTER = \{(.*?)\};', hd, re.S).group(1)
donorCenter = {int(a): b for a, b in re.findall(r'(\d+): "([^"]+)"', gc)}
for m in re.finditer(r'number: (\d+), name: "([^"]+)".*?aminoAcid: "([^"]+)".*?organ: "([^"]+)"', hd, re.S):
    n = int(m.group(1))
    if 1 <= n <= 64 and 'donorHdJs' not in gates[n]:
        gates[n]['donorHdJs'] = {'center': donorCenter[n], 'aminoAcid': m.group(3), 'organ': m.group(4)}

# Drive codex csv
for row in csv.DictReader(open(P['codex'])):
    if row['Gate'] and row['Gate'].isdigit():
        gates[int(row['Gate'])]['driveCodex'] = {'center': row['Center'], 'codonFamily': row['Codon Family'] or None}
centersDoc = list(csv.DictReader(open(P['centersDoc'])))

# dna-hd-compiler gates.ts (custom mapping)
ts = open(P['dnahd']).read()
for m in re.finditer(r'\n\s*(\d+): \{ number: \d+,.*?codon: "([ACGU]{3})", aminoAcid: "([^"]+)".*?organ: "([^"]+)"', ts):
    gates[int(m.group(1))]['dnaHdCompiler'] = {'codon': m.group(2), 'aminoAcid': m.group(3), 'organ': m.group(4)}

out = {'generatedFrom': {k: {'path': v, 'sha256': sha[k]} for k, v in P.items()},
       'gates': [gates[g] for g in range(1, 65)], 'centerBiologyBlackBook': centerBiology,
       'raveCodonStatements': raveCodonStatements, 'mergedCenters': mergedCenters, 'channels': channels,
       'centersFromDoc': centersDoc}
here = os.path.dirname(os.path.abspath(__file__))
json.dump(out, open(os.path.join(here, 'her-sources.raw.json'), 'w'), indent=1, ensure_ascii=False)
print('gates', len(out['gates']), 'keynotes', sum('raveKeynote' in g for g in out['gates']), 'genekeys', sum('geneKeys' in g for g in out['gates']),
      'zhouyi', sum('zhouyi' in g for g in out['gates']), 'donor', sum('donorHdJs' in g for g in out['gates']), 'dnahd', sum('dnaHdCompiler' in g for g in out['gates']),
      'codex', sum('driveCodex' in g for g in out['gates']))
