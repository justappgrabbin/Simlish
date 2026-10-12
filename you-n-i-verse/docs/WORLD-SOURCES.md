# Supplied world sources and character boundary

Reviewed the actual implementations in the supplied Drive archives:

- [WorldGen-main.zip](https://drive.google.com/file/d/1Lop7ONxHDnd8AYRSfA9GyzXjV-mtpuI0/view): `src/worldgen/worldgen.py` implements `WorldGen.generate_pano(prompt, image)` and `generate_world(prompt, image, return_mesh)`. The result is an Open3D triangle mesh in mesh mode or `SplatFile` in default mode. The panorama is a PIL image. Mesh/splat PLY exports need different renderers; Gaussian splat data must not be mistaken for ordinary triangle geometry.
- [WorldCoder-main-1.zip](https://drive.google.com/file/d/1oKCQRcLcprbaf1jvOe7mIalNVqvZhJK5/view): `worldcoder/agent/world_model/main.py` predicts next state, reward and completion using synthesized transition/reward code; agent/planner modules use interactions to refine and plan. It is a behavioral world-model component, not a visual scene renderer.

The downloaded WorldGen archive contains empty submodule directories. Its actual builders load UniK3D, FLUX.1-dev / FLUX.1-Fill-dev and WorldGen LoRA weights; the low-VRAM route imports Nunchaku and the advertised low-VRAM configuration is approximately 10 GB GPU memory. These dependencies and checkpoints have not been provisioned or run here. Reading its source does not establish successful inference or mobile execution.

The integration boundary must keep the saved face invariant: world generation receives the place description and optional **environment reference**, not the user's identity photo as a scene reference. Generated environments remain separate scene objects; body/form, outfit, hair, animation and world appearance can vary by place while the photo face and source identity stay fixed. The existing state-space/address system remains authoritative. WorldCoder predictions do not supply missing ontological coordinates.

Current runtime: colored 3D background and generated outfit vertex-color themes, plus the existing gradient/ring 2D preview. Neither supplied world engine is wired into that runtime yet. The existing choice-based game retains character identity and five source-backed observation views; generated geometry has no verified collision/navigation behavior.
