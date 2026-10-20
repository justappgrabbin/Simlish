# MESSY / Synthia Mesh Studio

A **running, phone-first, dependency-free JavaScript subsystem** based on the evolution described in the supplied Google Doc (Klein/MESSY discussion, mesh, generative media, inversion, and grammar social learning). It is a **standalone donor/build** for integration into the existing Synthia Computer. It does **not** overwrite or claim to contain the existing Synthia OS or Synthia server.

## Launch

1. **Phone / Netlify:** upload `messy-mesh-studio-netlify.zip` to Netlify's manual deploy. Its `index.html` is at the ZIP root. You can also open `index.html` directly from files on most modern browsers.
2. **Desktop:** open `index.html`. There are no external dependencies, servers, iframes, CDN calls, or required internet connection.
3. **Development:** `npm test`, `npm run build`. `src/engine.js` is UMD, usable with `require()` from Node or `window.MessyCore` in a browser. Edit sources and regenerate `index.html` via `node build.js`.

## What actually works

- **Mesh**: assert time-aware semantic triples with numeric/reality/hypotheticality values; maintain class membership; exclude contradictory relation pairs; attach secondary triples; inspect graph; preserve superseded assertions as inactive history; export checkpoint and non-destructively merge foreign history; optional typed canonical addresses.
- **Rule program**: use the Mesh view’s Rule Studio to install JSON-shaped declarations and groups (no card parser), deterministic probabilistic conditions, recurring schedules and `SET / ASSERT / NEG / SEC / ADD / MOVE / REMOVE / ENABLE / DISABLE / END` actions. Simulated time advances by user action.
- **Neural**: a real two-layer 8×12×4 trainable numeric neural classifier, with softmax, backpropagation and persisted weights; learns which of the four output types a user favors in a particular mesh context. **It is not yet a neural code-writing model.**
- **Inversion**: select sample character Alice / Marcus / Nova, generate a series of scenes directly from the active mesh's weighted relations, choose approach/observe/reframe, integrate **the visitor's** finished traces into the semantic mesh. Sample characters are fictional.
- **Foundry**: create immediately working, downloadable standalone HTML games, relationship editor apps, narrative browsers, and animated storyboard programs; each build has a source graph snapshot, manifest and an IndexedDB artifact record; the original HTML generation code is accessible and downloadable. Video artifact can record `.webm` through `canvas.captureStream + MediaRecorder` where the browser supports it.
- **Language Contact**: three speakers with probabilistic grammar weights, Monte Carlo encounters, listener learning and provenance logs. Deterministic for a given seed.
- **Voice/chat**: grounded local Synthia operator, conversational text field, browser text-to-speech, browser speech recognition if supported. Speaks on significant runtime events after browser interaction with speech toggle. The full original Synthia neural/back-end conversational systems are **not** embedded or impersonated.
- **Persistence**: mesh, rule, neural, language and chat state to localStorage; generated HTML and source snapshots to IndexedDB; each registered artifact remains separately named. Downloads are user actions; imported mesh is **merged**, not substituted.

## Known limitations and truth in labeling

1. The historical Google document contains proposals, partial code samples, and several HTML experiments, but **not a complete unified runnable repository**. This implementation integrates its concepts in an original, clean runtime. It does not claim binary fidelity with 1970s MESSY card-format semantics, the FORTRAN code, or arbitrary open-ended NLG.
2. Code/media generation is constrained/algorithmic rather than trained free-form text-to-code or video generation. Neural ranking is a genuine working model. Rich generative neural coding, computer vision, custom physical simulations, music composition, and high-fidelity video synthesis would require additional model training and integrations.
3. Canonical typed addresses are **validated and stored**, but full causal matrix / morph engine semantics cannot be inferred from the incomplete specification in this document. No missing rules were made up. SP, DIM, GATE/Line/Color/Tone/Base/precision fields remain available to an eventual adapter.
4. File and graph storage are browser/device-local; private browsing, browser storage clearing, or quota limits can remove data. Export JSON checkpoints and download HTML artifacts. App reports persistence failures and does not silently clear saved content.
5. This ZIP intentionally does **not** modify existing GitHub repositories, the Synthia server on Render, Supabase project, or the user's original Drive file. The integration seams are documented below.
6. The Browser speech APIs may require a supported browser, enabled sound, and user interaction. Recognition may use the browser provider's network service; this code itself makes no direct API calls.

## Integration contract

```
Synthia Computer (existing UI + voice + worlds)
  └── MESSY subsystem (new, independently tested)
      ├── SemanticMesh       ← existing typed addresses / evidence registry
      ├── Scheduler          ← AutoDev / Planner task facts and events
      ├── TinyNeuralNet      ← output preference learning (optional)
      ├── Inhabitation       ← Synth World / game engine adapter
      ├── LanguageContact    ← teaching/linguistics simulation app
      ├── generatedDocument  ← artifact lifecycle / installation service
      └── Synthia adapter   ← existing Synthia speech/brain backend (not replaced)
```

The existing system owns identity, true world state, model/agent execution, secure server actions, artifact publication and long-term storage. The new mesh module takes explicit triples, stores provenance, and returns deterministic semantic events. Its `get`/`tick`/`assert`/`snapshot` behaviors are testable without changing the existing server. The HTML generator's output must be attached to the existing AutoDev installer and app registry before it becomes a production-installed app; the local Foundry's IndexedDB registry is **not** a substitute for server publication.

## Output package contents

- `index.html`: fully embedded production preview (single-file, phone-friendly, no iframes).
- `src/engine.js`: runtime, learning, language, generative output and inversion code.
- `src/ui.js`, `src/styles.css`, `src/shell.html`: source of the interface and event bridges.
- `build.js`, `package.json`: reproducible build with Node.
- `tests/engine.test.cjs`: automated behavior tests.
- `docs/SOURCE_MAP.md`: feature provenance and integration boundaries.

MIT license applies to this new implementation only. Third-party historical research/source materials retain their own attribution and licensing. 
