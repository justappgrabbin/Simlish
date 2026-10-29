# One persistent source photo

The source photo is stored once in the complete project and local IndexedDB save.
It is cached in its original project encoding instead of being replaced with a
morphed output or re-encoded on every save. A SHA-256-based `photo:` reference
identifies that encoded source across project export, import and app reload.
Projects made before this field existed receive the reference on restore. An
import with a source photo that does not match its saved identity is rejected.

Every identity-conditioned model request includes this reference. The host receipt
keeps `source_identity_id` separate from `identity_sha256`, which hashes the resized
PNG actually sent for inference. The reference is client-supplied provenance, not
server proof that a generated image preserves someone's likeness. Generated
picture targets retain their engine, job ID and source identity reference in the
saved morph library. Prompt-only text-image outputs carry a null source identity:
that engine does not receive the source photo and is not identity-conditioned.

Changing or generating a body/form never replaces the source photo. Choosing a
replacement photo creates a different source reference and invalidates local
morph caches. Credentials remain outside project exports.

The intended automatic workflow remains one source photo → generated form/body →
3D reconstruction → rigging/animation, with the same source reference throughout.
The bundled MHR human-body handoff is implemented and browser-verified. SAM 3D
Body fitting and arbitrary generated/nonhuman-body handoffs remain unconnected.
Full-character GLB binaries, request metadata, proportions and animation are now
embedded in project autosaves/exports alongside their source identity. Device storage can be cleared by the user/OS; complete project
exports retain the source photo for restoring on another device.

Validation: unit checks cover shared/concurrent source capture, export/reload
identity stability, replacement photos and mismatched source rejection. Browser
verification checks save → page reload → complete project export without changing
the source encoding or its reference.
