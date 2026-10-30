# Synthia character morph — headless provider v1

This is a reusable service layer, **not a player-facing studio**. It preserves every existing Simlish module and source image. No app should require a player to enter dimensional coordinates, choose debug controls, or operate a morph editor.

## Input and result
- Call `createSimlishMorphProvider().prepare({ intent, reuseSavedPortrait })` on the user's device, or call `registerWithComputer(computer)` once when the game is embedded with the SynthAI Computer runtime. All calls are local; there is no network model.
- `intent`: `auto`, `portrait`, or `animated`. `rigged` is an explicit **not yet available** 3D stage. It must not silently fall back to pasting a face on a demo robot.
- `reuseSavedPortrait: true` must only be passed after the player has agreed to reuse the saved portrait; otherwise the provider returns `needs_consent` before reading IndexedDB. A caller may also pass a locally supplied `project`.
- Status values: `ready`, `needs_input`, `needs_asset`, `needs_consent`, and `unavailable`. Questions are normal-language, not technical setup forms. `ready` returns `output.dataUrl` in memory and a non-image receipt. Neither the photo nor the morph image is sent to the Computer state/event log.

## Execution
1. Reuse a saved morphed portrait or 2D animation where it exists.
2. Otherwise call the original `photo-morph.mjs` triangle-warp and lighting fit for a portrait, or `animation.mjs` and `frame-morph.mjs` for a requested 2D animation.
3. Do not manufacture an absent photo, body, wardrobe or rig. The planner reports which input or prepared asset is actually missing.
4. The existing `rig.mjs` still handles custom GLB and real animation clips independently. The 2D provider does **not** claim to be a 3D character generator.
5. This provider leaves the original project, image and cached sources unchanged. It does not write to the studio database; the caller decides how to display the result.

## Cross-system boundary
The SynthAI Computer is the registry/router, not the image generator. Its new `character-morph` capability initially reports **unavailable**, until a real local renderer registers. `registerWithComputer` connects the actual Simlish provider when the two runtimes share a JavaScript execution context. Separate Android apps/processes do not share JavaScript memory or IndexedDB automatically; a future secure device-local bridge must explicitly expose the capability to them. Do not publish a network endpoint for private photo bytes.

## Current verification and outstanding stages
Run `node --test you-n-i-verse/morph-provider.test.mjs`. Existing 2D source-engine tests also remain in place. A real Android phone, WebGL/GLB, and visual-quality pass still need physical testing. The draft Android launcher now enters the existing player world, not the manual studio. The studio remains intact behind an explicit Android developer intent; no original character data is replaced. The world attempts to show an already cached portrait after a player interaction, never forcing a morph editor. Without a saved morph it keeps the existing world artwork. This is still a player-world prototype, not completed 3D gameplay or the Human Design application. The body generator (coherent head, neck, hair, clothes, skinned 3D topology), 3D asset/rig selection, and truly autonomous end-to-end Synthia decision loop are not complete merely by adding a provider.
