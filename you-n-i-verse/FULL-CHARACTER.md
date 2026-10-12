# Automatic one-photo human character

Choose one clear portrait. The app now creates a complete character automatically
using bundled MHR body geometry and locally detected photo landmarks. A body-photo
upload, model host, account login and manual face alignment are not required for
this path. The same retained source photo remains in the saved project.

Describe a form such as “make my agent curvy” or “a tall athletic female agent” and
press Apply my request. Local descriptive rules choose coordinated proportions.
Unsupported/contradictory requests return a clear message instead of silently
changing the character. This is a bounded human-proportion interpreter, not a
trained universal language or reconstruction model. The former Human Agent
height/shoulder/chest/waist/hip/limb control concepts drive surface targets and rig
scales; manual controls are optional, collapsed tools.

The photo supplies a textured visible face surface with estimated depth. The
MHR asset supplies the back of the skull, body and rig. Body dimensions, simple
vertex-colored clothing and unseen anatomy are generated defaults, not measured
from the portrait. Photo likeness and seams can be imperfect, especially in
profile. Hair reconstruction and arbitrary nonhuman body topology are not part
of this local human path. Native MHR identity/expression targets remain available.

The character has a small looping movement/expression demonstration, not
speech-driven lip synchronization or a full locomotion library. PNG and animated
GLB exports are available. Complete project exports/autosaves now include the
avatar binary, source identity, request, proportions and animation. Reopening
restores these together. Replacing the photo clears its old derived avatar.

Validation includes the real mobile-size Chromium flow: one portrait + curvy
request → automatic face/body build → animation → GLB export → device save →
reload with source identity, request, proportions and animation intact. Exported
GLB has zero Khronos validation errors; one hierarchy warning concerns the
non-root skinned mesh. The tested reimport preserves the rig and morph targets.
Use `node scripts/check-full-avatar.cjs /absolute/portrait.jpg` while serving this
folder at localhost:8765 to reproduce. Playwright and Chromium are test-only
runtime requirements. Physical Android device testing is still outstanding.

SAM 3D Body, MakeHuman-JS, InstantID and other optional models can be added behind
these identity/form/output boundaries. This path does not depend on gated SAM
checkpoints. A later particle swarm can supply reconstructed geometry/appearance
with the same source identity and rig/output contracts; no swarm model is trained
or implemented by this change.

## State and game handoff

The saved character opens automatically in the animation workspace. New game paths retain the source photo identity through all five observation views; choosing a new source photo starts a new character path. This does not resolve missing ontological address fields. World appearance presets change the 3D background and generated outfit vertex colors while preserving face pixels, facial geometry and source identity. These color regions are not interchangeable clothing meshes. MakeHuman hair/clothing proxies and UV skin integration remain unconnected.
