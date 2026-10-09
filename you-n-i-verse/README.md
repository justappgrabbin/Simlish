# Klein neural laboratory

First runnable integration slice, not the finished Morphic Drama app.

Run `node demo.mjs` to train the local model, execute the supplied world capability and save `output/demo.json`. Run `node --test test.mjs` for checks. For the browser surface, serve this directory with any static host, e.g. `python3 -m http.server 8080`, then open localhost:8080. The static host only delivers files; training and execution run in JavaScript in the browser. No API key, LLM or application backend is required.

The original Klein source and Agentomagotchi execution modules are copied unchanged under vendor. AutoLing recognizes an authored three-token scene rule; AutoNovel composes actor/effect lineage. The world learns a transformation capability from six demonstrated transformations before the scored action executes. The world bridge enforces target compatibility. The saved event includes all five complete dimensional address projections and the world execution receipt.

The first learner has three fixed graph aggregation rounds over the supplied 36 channels and a trainable softmax neural action head. It is NOT the full supplied QuantumHDNet port. Its 24 synthetic examples and three held-out fixtures exercise the training/persistence connection, not broad generalization. Graph input for this demo is authored, not automatically inferred from arbitrary words. Grammar output gates the candidate action. Full semantic graph feature binding, jointly trainable message passing, user-example training, negation-aware language profiles, visual rendering and APK delivery remain next work.

The demo's addresses use the donor's coordinate generation for bookkeeping; this does not establish design-derived dimensional semantics. The user's Movement-upward rule remains to be explicitly implemented once its coordinate semantics are resolved.

Browser visual interaction has not been tested in this workspace. Browser storage can fail; export retains the model and episode. Node output is saved locally, not published to GitHub.

## Photo embodiment studio

The home page now loads a local portrait (including a separate mobile camera input), clips and scales it onto the body artwork supplied by the user, and provides draggable face placement, zoom, rotation, opacity and body-anchor controls. A custom body image can replace the reference. Cosmic/forest/ocean/desert presets alter surface tint and surrounding lighting. PNG export records the actual Canvas composition. Recipe export includes controls and source name, not private photo bytes or a fabricated canonical address.

Optional sprite-sheet upload supports configurable rows/columns, playback and adapted-sheet export. Face anchors are shared across frames in this first version; varying head positions require later per-frame landmarks. The four-world sheet is explicitly a variant atlas, not a twelve-action animation. The body artwork is cropped from the supplied reference board; it is not a segmented or articulated 3D body. This first compositor is not automatic face reconstruction, identity verification, anatomy generation, wardrobe generation, or the complete deep surface morph pipeline. Neural/world execution remains on world.html and laboratory.html; it is not yet bound to studio world presets. There is no cloud image upload or API call in the studio.

JavaScript syntax and existing three engine checks pass. Browser upload/camera/Canvas/export QA still needs a browser executable. The recipe alone cannot reconstruct a character without the original image assets.

### Face alignment and scene events

The studio now calls the browser's local FaceDetector when available, cropping one detected face into the face window. Detection is optional and unavailable on many browsers; there is no bundled face landmark model. Unsupported, failed, empty or multiple detections retain manual alignment. It detects a bounding box, not identity or eye/mouth landmarks.

The scene selector can execute an authored AutoLing entry rule and AutoNovel composition; this changes the visible world treatment and records lineage, all five full address projections and a rendering receipt. Direct preset clicks remain separate and clear the current event binding. Coordinates are explicitly authored demonstration coordinates, not calculated from user identity or astrology. Export includes the latest event; local history excludes photo bytes. Negation and unsupported commands are rejected before any scene effect. Run `node studio.test.mjs` for scene-address and face-fallback checks. Browser/camera QA remains unperformed due to missing browser executable.

### Saving a complete project

IndexedDB now stores the portrait, body, sprite image, alignment, world, controls and latest event together. The studio restores this project on opening and autosaves after edits. Save project on device forces a save; Download complete project produces a portable JSON containing the images, and Open saved project restores it. These project files contain personal photos: keep them private. Storage errors are displayed and a complete download remains available. PNG and settings-only exports are separate from project saves. Actual browser IndexedDB/camera/download behavior still requires device QA.

### Direct touch editing

Photo and face-circle modes are beside the preview. One finger moves the selected layer; two fingers scale it. Photo mode also supports two-finger rotation. The face window has a visible draggable size handle. Close-up view magnifies editing without changing the exported body framing. Fine sliders remain in the same panel. Guides are preview-only, and gestures autosave the existing project fields. Run node touch.test.mjs for focal-point zoom, rotated drag and circle independence checks. Physical-device multitouch still requires confirmation.
