// Video prompt synthesizer (spec §5.4) — extract setting/elements/lighting/
// mood/camera/motion from a rewritten SITUATION + effect parameters, and
// produce the templated cinematic prompt string.

const CAMERA_BY_BAND = {
  'severe-distortion': 'handheld shaky close-up, dutch angle',
  distortion: 'slow push-in, slight handheld drift',
  resonance: 'locked-off medium shot, gentle parallax',
  harmonic: 'smooth dolly arc, rising crane finish',
  'perfect-harmony': 'sweeping aerial orbit, golden-hour glide',
};

const MOTION_BY_ELEMENT = {
  water: 'flowing ripples and drifting mist',
  fire: 'flickering flames and rising embers',
  earth: 'settling dust and slow stone-shadow creep',
  metal: 'glinting reflections and ringing vibration',
  wood: 'swaying branches and unfurling leaves',
};

/**
 * @param {object} input - {situation, band, effectParams, elementAnalysis}
 * @returns {{setting, elements, lighting, mood, camera, motion, prompt}}
 */
export function synthesizeVideoPrompt({ situation, band = 'resonance', effectParams = {}, elementAnalysis }) {
  if (!situation?.setting) throw new Error('videoPromptSynthesizer: situation required');
  const setting = `${situation.setting.elementScene.name.replaceAll('-', ' ')} at ${situation.setting.temporalMarker}`;
  const elements = situation.entities.items.map((e) => e.name.replaceAll('-', ' '));
  const lighting = situation.atmosphere.lighting;
  const mood = `${situation.atmosphere.emotionalTone}, ${situation.atmosphere.emotionalState.name}`;
  const camera = CAMERA_BY_BAND[band] ?? CAMERA_BY_BAND.resonance;
  const leadElement = situation.setting.elementScene.element ?? elementAnalysis?.dominant ?? 'earth';
  const motion = MOTION_BY_ELEMENT[leadElement] ?? MOTION_BY_ELEMENT.earth;

  const fx = [];
  if (effectParams.chromaticAberrationPx > 0) fx.push(`chromatic aberration ${effectParams.chromaticAberrationPx}px`);
  if (effectParams.scanLines?.enabled) fx.push(`${effectParams.scanLines.count} scan lines at ${effectParams.scanLines.opacity} opacity`);
  if (effectParams.noiseOpacity > 0.05) fx.push(`film noise at ${effectParams.noiseOpacity} opacity`);
  if (effectParams.bloom > 0) fx.push(`bloom ${effectParams.bloom}`);
  if (effectParams.saturation && effectParams.saturation !== 1) fx.push(`saturation x${effectParams.saturation}`);
  if (effectParams.timeDilation && effectParams.timeDilation !== 1) fx.push(`time dilation x${effectParams.timeDilation}`);

  const prompt = [
    `Cinematic shot: ${setting}.`,
    `Scene elements: ${elements.join(', ')}.`,
    `Lighting: ${lighting}. Mood: ${mood}.`,
    `Camera: ${camera}. Motion: ${motion}.`,
    fx.length ? `Effects: ${fx.join('; ')}.` : 'Effects: clean grade.',
    'Style: painterly mythorealism, 35mm, shallow depth of field.',
  ].join(' ');

  return { setting, elements, lighting, mood, camera, motion, prompt };
}
