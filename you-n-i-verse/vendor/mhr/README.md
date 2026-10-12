# MHR body asset

`body-lod4.glb` is a conversion of Meta's Momentum Human Rig LOD4 FBX from
https://github.com/facebookresearch/MHR/releases/tag/v1.0.1 .

The official release includes assets/LICENSE.txt (Apache License 2.0), retained
as LICENSE here. Code checkout inspected at d96fafa33bbf018647c70c3525e91f53e79d2a14.
Conversion used Blender 4.3.2's FBX importer and GLB exporter, preserving the
126 exported rig joints, mesh skin and 117 identity/expression targets. MHR's
Python runtime reports 127 joints; FBX/export hierarchy omits one runtime joint.
This LOD4 mesh has 2,461 vertices. No original photo or generated user identity is
included in the bundled asset.

Runtime integration adds a photo-derived face, descriptive proportion controls
and a procedural movement clip. It does not run the original pose-corrective
neural network or fit the person's unseen body from a face photograph.
