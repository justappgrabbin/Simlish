# One photo, multiple forms

Upload one identity photo, add multiple target artwork files, then select a form
or prepare every form. Each target invokes the existing local MediaPipe landmark,
triangle warp, color correction and feathered blending pipeline. Results are
cached for switching and invalidated when the identity changes. Failed face
detection on one target does not stop the others.

Target names are unrestricted. This implementation requires one detectable face
per target; it does not reconstruct anatomy or invent a new target from text.
DECA, InstantID, LivePortrait, FaceFusion, UnumAvatar and Boneyard adapters remain
future work. The form library is held in memory; the selected embodiment is saved
by the existing project save, but the entire library is not persisted yet.

Verified: source-sharing, cache invalidation, stale-result rejection, per-target
failure isolation, and JavaScript syntax. Real-image/browser and Android testing
remain outstanding.
