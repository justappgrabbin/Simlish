# Synthia Unified Swarm — additive implementation v0.1.2 (corrected binary 3–2–1)

**Direct target: `justappgrabbin/Simlish`, consuming `justappgrabbin/Generative-address-system` as the sole semantic genotype authority.** Do not merge all binary donor archives or create an unrelated state-space engine. This is not an APK, an OS, or a simulated alternative to those applications.

This implementation integrates **unmodified source files** from user-supplied SWARM-OS Pass 05, Synthia Consciousness Swarm v0.1.1, and SWARM-OS Neural Composition. The source manifest provides cryptographic checksums. The 16 examined archives are classified in `DONOR-AUDIT.md`; heavyweight researchers and model systems remain optional. Original Drive ZIPs were not changed.

## Structural correction — 3/2/1 binaries (v0.1.2)

The author explicitly corrected the earlier projection. In the living swarm, **3**, **2**, and **1** each describe a *layer containing that many binary positions*, **not** three 2-bit bigrams / two 3-bit trigrams / one 6-bit hexagram. **Each position is one binary 0/1.** Level 3 is the lowest Human Design address layer; level 1 is the highest.

At the lowest layer, preserve these author-mentioned groupings as line-like units:

- Color + Tone + Base together form one grouped line.
- Line is one grouped line.
- Axis/axon + House was mentioned as a line; the exact wording and whether it denotes one combined group remain explicitly provisional. No aliases are invented between `arcAxis` and other address representations.

The 2 and 1 levels are binary slots whose detailed semantic mappings have **not** been supplied. Thus **all six binary values must come from a source or caller**; no gate parity, line parity, hash, hexagram, zodiac, numeric threshold or inferred address defaults can assign them. A grouping alone does not assign a binary state. Binary transitions update one explicit position and do not silently recompute others. The previous bigram/trigram/hexagram projection was a **Zhouyi six-line view**, not the swarm's 3/2/1. It remains separately available in `src/zhouyi-six-line.mjs`, and the exact old implementation remains copied in `src/archive/three-two-one-v0.1.1.mjs` for provenance.

Example (illustrative **test inputs**, not claimed Human Design meanings):

```js
const formation = swarm.mount321({
  provenance:'owner-provided-binary-observations',
  levels:{
    level3:[{bit:0},{bit:1},{bit:0}],
    level2:[{bit:1},{bit:0}],
    level1:[{bit:1}],
  },
  human:{color:4,tone:2,base:5,line:6,arcAxis:{axis:'ecliptic'},house:9},
});
```

The 0/1 values above are arbitrary example inputs, not derivations from the Human Design coordinates. The next semantic requirement is an **approved mapping for each binary opposition and its relation to the grouped address fields**.

## Working code

- **Formation:** `ConsciousnessSwarm.createPrimitive`, `connect`, `addFormationParticle`, `linkTransport`, `stepFormation`, `snapshot/restore`. Relationships require a caller-supplied predicate **and provenance**; transport recovery cannot manufacture semantic edges. Emitted cloud layouts show **positions** but do not turn coordinates into symbolic meaning.
- **Primitive computation:** `RecursivePrimitiveKernel.form/execute` on the original `PrimitiveExecutionVM`, supporting both nested compositions and 2D matrices. Its internal hash is a **computer identifier**, never the authoritative gate/line/color/tone/base address.
- **Neural composition:** the archived `composeDenseFormation`/`executeNeuralFormation` actual CPU forward pass. No learned weights, training or intelligent content claims are implied by this operation.
- **Address and expression:** existing King Wen↔Fu Xi `sixForGate`, source `PositionedStateSpace`, and explicit experimental shape/color/sound projector. Caller must supply evidence/provenance. `PositionedStateSpace`'s `planet/axis` convention is **not** silently mapped to Synthia's `planetary/arcAxis` fields.
- **Semantic cloud:** SVG creation from positioned, named nodes and source-attested relationship edges with text escaping; no invented language embeddings.
- **Sprites:** `packSpriteSheet` packs caller-supplied `RGBA8` frames into a deterministic atlas, preserving alpha and per-frame timing, with metadata and extraction. Full automatic sprite generation, seamless character morphing, occlusion, texture inpainting and 3D skins are **not** provided.
- **Simlish:** `attachUnifiedSwarmToSimlish({assembly,simlishPlug})` binds the **existing** `connectGenerativeStateSpace()` plug and delegates phenotype changes through its Klein/owner-authorized validation. No duplicate world simulation or forged execution result. The Simlish-specific `unified-swarm-entry.mjs` is the direct host entrypoint.

## Minimal use

```js
import {createUnifiedSwarm} from './src/index.mjs';
const swarm=createUnifiedSwarm({source:existingSynthiaAssembly,simlishPlug:existingSimlishPlug,authorize:ownerPolicy});
swarm.add({id:'form',label:'Shape',x:0,y:.25});
swarm.add({id:'world',label:'World',x:.3,y:-.25});
swarm.relate({from:'form',to:'world',predicate:'expresses',provenance:'verified-project-rule-42'});
const svg=swarm.cloudSVG();
// Actual Simlish morph: supply a complete authoritative genotype address.
await swarm.morphSimlish({intent:'Morph a world object',genotypeAddress:completeCanonicalAddress,
  genotypeSource:'owner-explicit',objectId:'chair',operation:'transform'});
```

## Test

```bash
npm test
npm run demo
```

The local unit suite runs directly against original donor code. An additional GitHub workflow runs the real Simlish/Generative State Space integration on both checked-out projects. It must be green before merging to `main`.

## Next correctness gates

1. Resolve the **dimensional WHERE/WHEN conflict** across donor versions against the user's approved leading-vector interpretation. Do not silently pick a donor table.
2. Implement evidence-driven formation selection, an observed-outcome learning policy, and durable storage in the existing computer/Simlish backend. The current `choices()` is constraint matching, **not autonomous semantic understanding**.
3. Build high-quality reusable skins with alpha/normal/depth channels, atlas metadata, animation frames, hitbox tracking and real device acceptance tests.
4. Optional chart-based birth initialization must use actual ephemeris/time-zone data with explicit permission; astrology/Human Design interpretations are user-configurable hypotheses, not established personality measurement.
5. Heavy model donors (HY-WorldPlay, VideoMamba, VFIMamba, DreamGaussian, etc.) are not automatically installed or executed. Their separate CPU/GPU/model/runtime and licensing requirements require a deliberate later integration.

**Safety:** two `ai-face-swap-2026` archives were blocked by Drive malware/spam scanning, and have been excluded. Do not attempt to work around that flag.
