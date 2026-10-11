# Character pipeline adjustment

Keep the working identity/project storage, source-backed lexical slots, changing-line paths, CPU face landmarks, and animation/sprite caches. Replace the presentation and body assembly that made intermediate experiments look like final characters.

## Three stages

1. **Identity and character image.** Preserve one original photo. Produce the chosen image/form with pose alignment and coherent skin/lighting. The current local warp is an image preview. It is not a completed 3D body, and its numerical fit does not certify identity fidelity or artistic quality.
2. **Integrated body.** Reuse or prepare an anatomically coherent body with the head, neck, hair, wardrobe, texture and rig integrated. MHR is the identified body-model candidate, not an installed runtime in this build. Its Python implementation and assets require separate integration/preparation. The segmented Unum donor body and frontal photo-on-robot test are not acceptable substitutes for the finished character. A face-only input cannot measure the user's unseen body; those choices must come from a chosen body/template or an explicit prior.
3. **Performance.** Apply reusable clips to the prepared character, interpolate the rig and bake/cache sprite frames. Do not rerun identity reconstruction per playback frame. The existing Three.js player and sprite renderer remain useful at this stage.

The main 3D page no longer automatically loads the demonstration robot or restores an experimental photo surface. Existing user files are retained. Diagnostic rig and face-surface tests are available only with `rig.html?test=1`; they are separated from the character flow. Prepared custom GLB imports and their existing playback tools remain available. The better integrated-body stage is still unfinished; no automatic replacement model has been claimed.

## CPU swarm fit implemented

`swarm-fit.mjs` performs bounded particle-swarm optimization in JavaScript on the main thread, yielding periodically to the event loop. It creates no Web Worker, backend request or GPU operation. Face fitting uses 24 parameter particles, 40 iterations, and at most a few hundred low-frequency image samples. The values are three color offsets and horizontal/vertical lighting gradients. Eyes and mouth are excluded from fit samples; high-frequency source texture is retained while these offsets are applied. Held-out samples decide whether to accept the fitted values or retain the bounded baseline. The original photo remains unchanged, and the numerical receipt is stored with the result.

This is **lighting fitting**, not a general neural body reconstruction model. Its loss compares low-frequency appearance samples and is not a measurement of human attractiveness, identity accuracy, topology validity, or overall character suitability. A small reduction in that loss is only evidence for that limited objective. Source pose warping, feathering and the CPU MediaPipe/WASM detector remain separate components; JavaScript PSO does not make the entire detector implementation pure JavaScript.

## State space and future reconstruction

The supplied meanings stay authoritative. Lighting belongs to Evolution. Visual optimizers act within explicitly supplied geometric/material constraints; they do not invent gate meanings, choose semantic coordinates from a random seed, or use degree as the generator of context. The optimizer seed only initializes numerical search. Existing semantic inference memory and canonical line rules are unchanged.

A future reconstruction swarm requires real primitives (geometry, connectivity, skin weights, appearance parameters), observed-image constraints, a projection/rendering objective, and a prior/template for unseen surfaces. It can optimize those values on CPU. Stable fitted representations can be cached. State-space organization provides consistent meaning and permitted transformations, but it does not supply missing photographic evidence or automatically guarantee photorealism.

## Verification

Controlled tests recover a known spatial illumination gradient and exercise bounded search/cancellation without workers or GPU. The browser test uses an actual test portrait, confirms held-out lighting loss does not worsen, checks source-photo bytes are unchanged, verifies the main page does not load a saved demonstration robot, and records zero off-origin requests. The observed improvement on the real fixture is modest and should not be generalized to full reconstruction quality. Physical phone behavior and the integrated body stage still need validation.
