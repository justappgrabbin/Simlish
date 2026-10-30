/**
 * Synthia local character morph provider (portable across player/world/studio).
 * No UI, network dependency, automatic photo collection, or remote model.
 * Uses the existing Simlish photo and pose engines; leaves source photos intact.
 */
import { readProject } from './project-store.mjs';
import { morphPhoto } from './photo-morph.mjs';
import { sheetGrid, personalizeFrames } from './animation.mjs';
import { expandMotion } from './frame-morph.mjs';

const MODES = new Set(['auto', 'portrait', 'animated', 'rigged']);

/** Pure plan for assistants and world callers. Never expose developer controls. */
export function planCharacterMorph(project, { intent = 'auto' } = {}) {
  if (!MODES.has(intent)) throw new TypeError(`Unknown morph intent: ${intent}`);
  if (intent === 'rigged') return { status: 'needs_asset', reason: 'prepared-rig-required', question: 'What kind of character body should I use?' };
  if (!project?.photo) return { status: 'needs_input', reason: 'portrait-missing', question: 'Would you like to use a photo for your character, or describe their appearance?' };
  if (intent === 'animated') {
    if (project.morphedSprites) return { status: 'ready', kind: 'animated-2d', origin: 'saved-poses' };
    if (project.sprites) return { status: 'prepare', kind: 'animated-2d', origin: 'existing-pose-sheet' };
    return { status: 'needs_asset', reason: 'motion-missing', question: 'Which movement should your character perform?' };
  }
  if (intent === 'auto' && project.morphedSprites) return { status: 'ready', kind: 'animated-2d', origin: 'saved-poses' };
  if (project.morphedImage) return { status: 'ready', kind: 'portrait-2d', origin: 'saved-morph' };
  if (project.body) return { status: 'prepare', kind: 'portrait-2d', origin: 'existing-body-art' };
  return { status: 'needs_asset', reason: 'body-missing', question: 'What appearance or body would you like your character to have?' };
}

function loadImage(dataUrl) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('Could not load the saved character image'));
    image.src = dataUrl;
  });
}

/**
 * prepare() is an execution capability, not a screen. Accepts an injected
 * project (for tests or Computer bridge) or reads the user's local project.
 * Result images stay only in the caller's local process. Receipts contain no
 * images; the Computer adapter must persist only the receipt, not output.
 */
export function createSimlishMorphProvider({
  loadProject = readProject,
  decodeImage = loadImage,
  warp = morphPhoto,
  personalize = personalizeFrames,
  interpolate = expandMotion,
} = {}) {
  return {
    id: 'simlish-local-morph-v1',
    formats: ['portrait', 'animated'],
    supports(request = {}) { return request.intent !== 'rigged'; },
    async prepare(request = {}) {
      const intent = request.intent ?? 'auto';
      // Privacy: do not read a saved portrait without permission to reuse it.
      if (request.reuseSavedPortrait !== true && request.project == null) {
        return { status: 'needs_consent', reason: 'saved-portrait-consent', question: 'May I use your saved character photo for this experience?' };
      }
      const project = request.project ?? await loadProject();
      const plan = planCharacterMorph(project, { intent });
      if (plan.status !== 'prepare' && plan.status !== 'ready') return plan;
      if (plan.status === 'ready') {
        const dataUrl = plan.kind === 'animated-2d' ? project.morphedSprites : project.morphedImage;
        if (!dataUrl) throw new Error('Saved character image missing');
        return { status: 'ready', kind: plan.kind, output: { mime: 'image/png', dataUrl, frameGrid: plan.kind === 'animated-2d' ? (project.animationReceipt?.grid ?? null) : null }, receipt: { source: plan.origin, mode: plan.kind, reused: true, offline: true } };
      }
      const photo = await decodeImage(project.photo);
      if (plan.kind === 'portrait-2d') {
        const body = await decodeImage(project.body);
        const result = await warp(photo, body);
        return { status: 'ready', kind: plan.kind, output: { mime: 'image/png', dataUrl: result.canvas.toDataURL('image/png') }, receipt: { source: plan.origin, mode: plan.kind, reused: false, offline: true, engine: result.receipt?.method ?? 'simlish-photo-warp' } };
      }
      const sprite = await decodeImage(project.sprites);
      const size = project.sheet ?? { cols: 4, rows: 1 };
      const grid = sheetGrid(sprite.width, sprite.height, size.cols, size.rows);
      const frames = await personalize(photo, sprite, grid, { morph: warp });
      const animation = await interpolate(frames.canvas, grid);
      return { status: 'ready', kind: plan.kind, output: { mime: 'image/png', dataUrl: animation.canvas.toDataURL('image/png'), frameGrid: animation.grid }, receipt: { source: plan.origin, mode: plan.kind, reused: false, offline: true, keyframes: grid.count, frames: animation.grid.count, interpolation: animation.method } };
    },
  };
}

/** Register the actual local provider with an optional embedded Computer. */
export function registerWithComputer(computer, provider = createSimlishMorphProvider()) {
  if (typeof computer?.morph?.registerProvider !== 'function') throw new Error('Computer morph capability unavailable');
  return computer.morph.registerProvider(provider.id, provider, { formats: provider.formats, priority: 100 });
}
