# One photo, multiple forms

Upload one identity photo, add multiple target artwork files, then select a form
or prepare every form. Each target invokes the existing local MediaPipe landmark,
triangle warp, color correction and feathered blending pipeline. Results are
cached for switching and invalidated when the identity changes. Failed face
detection on one target does not stop the others.

Target names are unrestricted. This implementation requires one detectable face
per target; it does not reconstruct anatomy or invent a new target from text.
The optional model service now connects DECA, InstantID, LivePortrait, FaceFusion
and UnumAvatar adapters; their upstream inference requires installed checkpoints.
Boneyard rig authoring remains pending. The form library, selected form, shape/sculpt settings and prepared results are
retained in device autosave and complete-project exports. Old projects still open.

Verified: source-sharing, cache invalidation, stale-result rejection, per-target
failure isolation, project validation and JavaScript syntax. Real-image morphing,
generation-to-morph and save/reload are browser-tested. Physical Android testing
remains outstanding.
