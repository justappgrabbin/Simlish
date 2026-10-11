# One-photo 3D face surface

Choose an identity photo, then **Build 3D face from my photo**. Drag the 3D preview
to turn it, adjust Stylize shape, play the mouth-expression demonstration and
export an animated GLB. The Android export uses the existing native file saver.

The bundled MediaPipe face model supplies 468 visible face landmarks with estimated
depth. Its canonical face tessellation becomes a photo-textured surface; eye
openings are filled with visible photo pixels. Normals are accumulated for smooth
shading. Two morph targets (Stylize and MouthOpen) and a looping expression clip
are embedded in GLB. Export retains the chosen stylized shape. Three.js 0.160.1 is
bundled locally (MIT notice retained); no CDN is contacted during use.

This is a visible-face surface, not a complete reconstructed head or body. It does
not infer hair, ears, clothing, unseen anatomy or a full cartoon appearance.
Stylize currently changes geometry; photo texture remains photographic. Strong
expressions or profile views can expose approximations. The expression is a small
mouth-opening demonstration, not speech-driven animation.

DECA/UnumAvatar full reconstruction, InstantID generation, LivePortrait animation
and alternate character rigs still need integration and model/runtime provisioning.

Validation: two GLB/geometry regression tests, plus existing morph checks. Chromium
with the real bundled model successfully built, rendered, styled, animated and
exported GLB from a public OpenCV portrait. Visual inspection corrected texture
coordinates and replaced projected triangulation with canonical face topology.
Export was parsed to verify selected weight, binary lengths and animation metadata.
Physical Android device testing remains outstanding.
