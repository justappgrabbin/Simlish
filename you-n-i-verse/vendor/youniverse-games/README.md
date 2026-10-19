# YOU-N-I-VERSE · Cynthia Morph Game

A local photo-morph studio with three persistent game worlds and the supplied Synthia automata tools. Upload a photo or artwork, direct a transformation with words and shape controls, compare images, and save branches of your character's history.

## Run

Requires Node.js 22+ and Python 3.10+ with Pillow and NumPy. No npm dependencies, API key or pretrained model is required for the local engine.

```sh
cd youniverse
python3 -m pip install 'Pillow>=10,<13' 'numpy>=1.24,<3'
npm run doctor
npm start
```

Open http://localhost:3000. Set `PYTHON_BIN` if your Python executable has a different name. Uploaded images, transformations, reviews and game saves stay in the local `data/` directory, which is excluded from version control.

The local renderer transforms actual source pixels. A small controller trained from authored examples and accepted reviews predicts deformation controls; it does not invent new anatomy or train an image diffusion model. Optional diffusion adapters require a separately configured server. The worlds currently use an included sample birth chart.

## Cynthia agent interface

`integration/cynthiaTools.js` exposes identity creation, state-driven morphing, job status, identity review and controller training. It also connects the supplied Pure Synthia Automata engine through `listReasoningTools`, `reason` and `reasonThenMorph`. Reasoning call provenance is saved with the resulting morph job. Visual directives still come from the caller; successful grammar execution does not establish image quality or goal optimality.

Use one engine owner per data directory. No HTTP server is needed for the in-process Node interface. The Python bridge is distributed in the broader repair package and is not part of this game upload.

## Checks

```sh
npm run check
npm test
node vendor/pure-synthia-automata/tests/run.mjs
node vendor/pure-synthia-automata/test/merged.smoke.mjs
```

Six original game suites and seven studio tests passed. Studio tests cover actual pixel transformation, history, restart persistence, worlds, cancellation, provider contracts and supplied-tool provenance. The supplied automata checks also passed. HTTP routes were tested in-process; live browser interaction and real diffusion providers remain unverified in the repair sandbox.

The vendored automata source is preserved from the supplied archive. Its scientific and historical claims are not established by passing software tests. Reference corpora and unrelated archived projects are outside this game upload.

## Play the journey

Create your identity, then enter the three worlds. Make three choices in each world and save one image transformation in the studio. The progress panel shows the next objective. Completing all ten steps reveals the return scene; the worlds remain available for continued exploration.

World scenes, choices and chapter progress save automatically. Returning to a world restores its last scene without creating another entry transformation. The Living Map exposes both abilities and travel pathways. The interface locks world-changing actions while a choice is saving.

This release uses the existing procedural story rules and source-photo deformation. Live browser interactions still need verification on a machine that permits a listening server. Run `npm run doctor` to check the actual local image pipeline before starting the game.
