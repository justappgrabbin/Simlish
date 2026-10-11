# Optional universal model host

The phone's local face morph and 3D face surface remain local. This separate service
accepts explicit model requests from the studio, queues one model job at a time,
preserves identity hashes and returns generated images, avatars, video and logs.
Add generated pictures to the morph library; GLB outputs can open in the 3D viewer.
Prompts are unrestricted descriptions, not a fixed form/category catalog.

## Start

Install PyTorch appropriate for your CPU/GPU and `pip install -r requirements.txt`.
Install upstream engines and their checkpoints into operator-controlled directories.
Configure only fully provisioned engines as ready. The API intentionally rejects
requests to unavailable engines before accepting identity photos.

```
export MORPH_API_TOKEN='choose-a-private-token'
export MORPH_ORIGINS='http://localhost:8765,https://appassets.androidplatform.net'
export SDXL_MODEL='/absolute/path/to/local/diffusers/model'
export MORPH_READY_ENGINES='text-image'
export MODEL_PYTHON='/absolute/path/to/model/python'
export MORPH_DEVICE='cpu'  # cuda for a GPU installation
uvicorn server:app --host 127.0.0.1 --port 8877
```

For phone access, serve this behind an HTTPS reverse proxy and configure its origin
in MORPH_ORIGINS. Enter that URL and token in the app's Generation and full-avatar
models panel. The token is held in the form, not saved in projects/localStorage.
No hosted public service is deployed by this source change. Android now has INTERNET
permission for these opt-in requests; the model service is never contacted by local
photo editing, and no default model host is configured.

## Engines

| Engine | Configuration | Inputs / outputs |
|---|---|---|
| text-image | SDXL_MODEL: any local AutoPipelineForText2Image-compatible model | Prompt + seed → PNG |
| instantid | INSTANTID_ROOT, SDXL_MODEL, InsightFace antelopev2, InstantID ControlNet and ip-adapter.bin | Photo + prompt + seed → identity-conditioned PNG |
| deca | DECA_ROOT with DECA/FLAME dependencies and checkpoints | Photo → OBJ, textures, landmarks; GLB conversion when Blender is installed |
| unum | UNUM_ROOT, BLENDER_BIN, UNUM_FACE_MODEL; Blender Python dependencies incl. MediaPipe, InsightFace buffalo_l, SciPy | Photo-derived head + inferred humanoid body → rigged GLB |
| liveportrait | LIVEPORTRAIT_ROOT with human checkpoints | Photo + driving MP4 → animated video |
| liveportrait-animal | same root with animal checkpoints | Animal photo + driving MP4 → animated video |
| facefusion | FACEFUSION_ROOT with configured models | Source photo + target picture → PNG |

Comma-separate provisioned IDs in MORPH_READY_ENGINES. Separate engines may need
separate environments: override MODEL_PYTHON_DECA, MODEL_PYTHON_INSTANTID,
MODEL_PYTHON_LIVEPORTRAIT, MODEL_PYTHON_LIVEPORTRAIT_ANIMAL or
MODEL_PYTHON_FACEFUSION where dependencies conflict. Unum uses Blender's Python.
Upstream model licenses still apply; the adapters do not redistribute their weights.
Boneyard skeleton/view authoring has not been wired in this change.

Readiness means the operator has marked an engine provisioned, not that every
checkpoint has been automatically tested. A job failure is explicit and its log can
be downloaded. Job output files/receipt persist, but status polling currently lasts
for the service session; restarting does not restore the in-memory queue. Inputs and
outputs are retained in MORPH_JOBS until the operator removes completed jobs.

## Verified scope

API authentication, unavailable-engine rejection, output-path isolation, and argument
construction have regression tests (`python test_service.py`). Source syntax checked.
Text-image tested with the actual downloaded segmind/tiny-sd model, revision
cad0bd7495fa6c4bcca01b19a723dc91627fe84f (CreativeML Open RAIL-M), on CPU.
Unum was tested with its actual MediaPipe and InsightFace buffalo_l models in Blender 4.3.2. DECA, InstantID, LivePortrait and FaceFusion have not run inference here.

End-to-end proof: a mobile-size Chromium studio submitted a real cartoon prompt
through the API, polled the worker, downloaded the generated PNG and added it to
the target library without page errors. An invalid token shows the authentication
error in the app. This is actual generation on CPU, not a mock engine.

The generated cartoon also passed the actual landmark morph step with the original
photo. External GLB import/preview/export controls passed a browser check with a
real exported face GLB. A separate 30-step photograph-style dog-on-log generation
completed on the same model.

## Full-body prototype

Unum retains the supplied photo-derived head and its 12 facial morph targets.
The integration replaces the cropped shoulder base with a **generic inferred
humanoid**, including hands, feet, a 15-bone skin, body-width shape keys and a
looped movement demonstration. This segmented mannequin is a prototype, not
photorealistic anatomy or recovery of the person's body/clothing from a face photo.
Body reconstruction from full-body images and production character topology remain
future work. Facial expressions and body shape controls are available in the viewer.

Use UNUM_INSIGHTFACE_ROOT for the directory containing models/buffalo_l.
Set UNUM_PYTHONPATH to dependencies built for Blender's own Python version,
and UNUM_LIBRARY_PATH if additional system shared libraries are required.
Blender is invoked with --python-use-system-env and --python-exit-code 1 so
Python dependency errors fail the job rather than reporting success. This runtime
used Blender Python 3.13, MediaPipe, InsightFace, ONNX Runtime, SciPy and libGLESv2.
The model host must be deployed/configured separately for access from a phone;
the workspace test host is loopback only.
