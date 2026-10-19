// LAYER 3 — UNSOURCED / FILLED-IN. Things this module needed that no cited source states.
// Each item says why it exists. Nothing here may be shown as science or as an HD claim.
const u = (id, subject, scale, text, reason, data) => Object.freeze({ id, layer: 'unsourced', subject, scale, text, cites: Object.freeze([]), reason, ...(data ? { data: Object.freeze(data) } : {}) });

// Where each HD center sits on the body, translated into the same coarse region words used for organs.
// The HD sources name some centers anatomically ("Head", "Throat", "Base of the Spine", "Sacral",
// "Solar Plexus", "Heart") but none gives a body region in these terms; G/Ajna/Spleen are placed by
// the usual bodygraph drawing only. These are interpretations, so mismatches that use them say so.
export const CENTER_REGION = Object.freeze({
  Head: 'head', Ajna: 'head', Throat: 'neck', G: 'thorax', Heart: 'thorax',
  SolarPlexus: 'abdomen', Spleen: 'abdomen', Sacral: 'pelvis', Root: 'pelvis',
});
const REASON_REGION = {
  Head: 'name "The Head Center" (Black Book) read as head',
  Ajna: 'not named anatomically; bodygraph drawing places it in the head',
  Throat: 'name "Throat" read as neck',
  G: 'not named anatomically; bodygraph drawing places it mid-chest',
  Heart: 'name "Heart" read as thorax',
  SolarPlexus: 'name "Solar Plexus" read as upper abdomen',
  Spleen: 'name "Splenic" read as abdomen',
  Sacral: 'name "Sacral" read as pelvis (sacrum)',
  Root: 'name "The Base of the Spine" read as pelvis',
};
export const UNSOURCED_FACTS = Object.freeze([
  ...Object.entries(CENTER_REGION).map(([c, r]) => u(`uns.center.${c}.region`, `center:${c}`, 'body', `Placed in the ${r} region for comparison.`, REASON_REGION[c], { region: r })),
  u('uns.layout.svg', 'scale:body', 'body', 'Body silhouette, center shapes and their x/y positions are drawn for navigation only; not anatomical, not to scale.', 'no source supplies drawing coordinates'),
  u('uns.layout.schematics', 'scale:cell', 'cell', 'The cell, DNA, codon and amino-acid pictures are simplified schematics, not to scale.', 'drawn for this module'),
  u('uns.region.vocab', 'scale:body', 'body', 'Region words (head, neck, thorax, abdomen, pelvis, scrotum, whole-body) are this module\u2019s own coarse vocabulary.', 'needed to compare organ location with center placement'),
]);

// SVG layout (unsourced: see uns.layout.svg). viewBox 0 0 300 640.
export const LAYOUT = Object.freeze({
  centers: {
    Head: { x: 150, y: 52, shape: 'tri-up' }, Ajna: { x: 150, y: 108, shape: 'tri-down' }, Throat: { x: 150, y: 170, shape: 'square' },
    G: { x: 150, y: 250, shape: 'diamond' }, Heart: { x: 196, y: 280, shape: 'tri-small' }, SolarPlexus: { x: 214, y: 380, shape: 'tri-right' },
    Spleen: { x: 86, y: 380, shape: 'tri-left' }, Sacral: { x: 150, y: 410, shape: 'square' }, Root: { x: 150, y: 480, shape: 'square' },
  },
  organs: {
    pineal: { x: 150, y: 70 }, pituitary: { x: 150, y: 90 }, thyroid: { x: 150, y: 152 }, parathyroid: { x: 166, y: 156 }, thymus: { x: 150, y: 210 },
    liver: { x: 122, y: 300 }, spleen: { x: 186, y: 312 }, pancreas: { x: 158, y: 322 }, adrenal: { x: 128, y: 336 }, kidney: { x: 122, y: 350 },
    lymphatic: { x: 228, y: 230 }, ovary: { x: 126, y: 440 }, testis: { x: 150, y: 522 },
  },
});
