/**
 * Direct, same-origin bridge between the existing YOU-N-I-VERSE Studio project
 * and the existing SynthAi Computer ATO fitness renderer.
 *
 * Imports the ORIGINAL Studio IndexedDB project-store; never moves photos into
 * Computer's event history. Accepted morphs become append-only visual layers
 * inside the original Studio project, leaving photo/body/morphedImage intact.
 */
import { readProject, saveProject, validateProject } from './project-store.mjs';

const MAX_PIXELS = 3_000_000;
const MAX_LAYERS = 64;
const DATA_URL = /^data:image\/(png|jpeg|webp);base64,/;

function selectedImage(project) {
  const active = project.activeFitnessMorphId;
  const layer = (project.fitnessMorphLayers ?? []).find(x => x.id === active);
  return {
    image: layer?.image ?? project.morphedImage ?? project.body,
    sourceLayerId: layer?.id ?? null,
  };
}
async function decode(dataUrl) {
  if (!DATA_URL.test(dataUrl || '')) throw new Error('Studio has no compatible character image');
  const [prefix, encoded] = dataUrl.split(',');
  const mime = prefix.slice(5, -7);
  const binary = atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < bytes.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}
function validateCoefficients(coefficients) {
  if (!Array.isArray(coefficients) || coefficients.length !== 9 ||
      !coefficients.every(v => Number.isFinite(v) && Math.abs(v) <= 60))
    throw new TypeError('A bounded nine-value accepted Simlish lighting expression is required');
}

export async function readStudioCharacter() {
  const project = await readProject();
  if (!project) throw new Error('Save a character in YOU-N-I-VERSE Studio first');
  if (project.sprites) throw new Error('The current character uses a sprite sheet; static image expression is not the active render path');
  const selected = selectedImage(project);
  const source = await decode(selected.image);
  return {
    file: new File([source], 'studio-current-character.' + source.type.split('/')[1], {
      type: source.type,
    }),
    savedAt: project.savedAt,
    sourceLayerId: selected.sourceLayerId,
    activeLayerCount: (project.fitnessMorphLayers ?? []).length,
    world: project.world,
  };
}

export async function applyAcceptedStudioLighting({
  coefficients, evidence, expectedSavedAt, expectedLayerId = null,
} = {}) {
  validateCoefficients(coefficients);
  if (!evidence?.accepted || !Number.isFinite(evidence.gain) || evidence.gain <= 0 ||
      typeof evidence.revision !== 'string' || !evidence.revision ||
      !['constant', 'gradient'].includes(evidence.selected))
    throw new Error('Only verified accepted ATO evolution may change Studio appearance');
  const project = await readProject();
  if (!project) throw new Error('No saved Studio character found');
  validateProject(project);
  if (project.savedAt !== expectedSavedAt || (project.activeFitnessMorphId ?? null) !== expectedLayerId)
    throw new Error('The Studio project changed since fitness measurement; reopen the Studio character and measure again');
  if (project.sprites) throw new Error('Static image fitness cannot replace a sprite-sheet character');
  const layers = project.fitnessMorphLayers ?? [];
  if (layers.length >= MAX_LAYERS)
    throw new Error('Studio fitness history capacity reached; export your original project before new layers');
  const selected = selectedImage(project);
  const blob = await decode(selected.image);
  const bitmap = await createImageBitmap(blob);
  let canvas;
  try {
    if (bitmap.width * bitmap.height > MAX_PIXELS)
      throw new Error('Character image exceeds local 3-million-pixel safety budget');
    canvas = document.createElement('canvas');
    canvas.width = bitmap.width; canvas.height = bitmap.height;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0);
    const img = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
    const { data } = img;
    for (let y = 0; y < bitmap.height; y++) for (let x = 0; x < bitmap.width; x++) {
      const fx = 2 * x / Math.max(1, bitmap.width - 1) - 1;
      const fy = 2 * y / Math.max(1, bitmap.height - 1) - 1;
      const at = (y * bitmap.width + x) * 4;
      for (let channel = 0; channel < 3; channel++) {
        const delta = coefficients[channel] +
          coefficients[channel + 3] * fx + coefficients[channel + 6] * fy;
        data[at + channel] = Math.max(0, Math.min(255, Math.round(data[at + channel] + delta)));
      }
    }
    ctx.putImageData(img, 0, 0);
  } finally {
    bitmap.close();
  }
  const id = 'studio-fitness-' + crypto.randomUUID();
  const layer = Object.freeze({
    id, sourceLayerId: selected.sourceLayerId,
    createdAt: new Date().toISOString(),
    source: 'simlish:formation-fitness-evolution.v1',
    image: canvas.toDataURL('image/png'),
    receipt: {
      strategy: evidence.selected, gain: evidence.gain,
      activeRevision: evidence.revision,
      candidates: structuredClone(evidence.candidates),
      coefficients: [...coefficients],
    },
  });
  // Never overwrite photo, body, native morphedImage, or previous layers.
  // Only append a newly verified expression and move the visible pointer.
  const updated = validateProject({
    ...project,
    fitnessMorphLayers: [...layers, layer],
    activeFitnessMorphId: id,
    savedAt: new Date().toISOString(),
  });
  // Validation and rendering finish BEFORE writing one IndexedDB transaction.
  await saveProject(updated);
  return Object.freeze({
    applied: true, layerId: id, previousLayerId: selected.sourceLayerId,
    layerCount: updated.fitnessMorphLayers.length, width: canvas.width,
    height: canvas.height, originalAssetsPreserved: true,
  });
}
