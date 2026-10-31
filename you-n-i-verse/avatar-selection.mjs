// Choose a visible character using actual saved user output. Never silently replace
// a prepared Morph World character with the sample robot or procedural mannequin.
export function selectCharacterSource(project, rig) {
  if (project?.photo && project?.gameAvatar && /^data:image\/(?:png|jpeg|webp);base64,/.test(project.gameAvatar)) {
    return {kind:'morph-image', method:project.gameAvatarMethod??'studio-image',
      label:'Your Morph World character · 2D artwork inside this 3D world'};
  }
  if (rig?.buffer && rig.label && rig.label !== 'Demonstration robot') {
    return {kind:'custom-3d', label:'Your imported animated 3D character'};
  }
  return {kind:'reference-art', label:'Reference character · customize in Morph World'};
}
