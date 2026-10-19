// LAYER 1 — ESTABLISHED SCIENCE. Every fact cites >=1 'science' source id and a locator
// (section of the cited page). Text is a close paraphrase of the cited sentence; nothing here
// mentions Human Design. Scales: body | organ | cell | dna | codon | amino-acid.
// `region` is a coarse anatomical region used only for comparison with HD placement; each
// organ's region is backed by its body-scale fact.
const f = (id, subject, scale, text, cites, locator) => Object.freeze({ id, layer: 'science', subject, scale, text, cites: Object.freeze(cites), locator });

export const ORGANS = Object.freeze({
  pineal:      { name: 'Pineal gland', region: 'head', regionFact: 'sci.pineal.body' },
  pituitary:   { name: 'Pituitary gland (hypophysis)', region: 'head', regionFact: 'sci.pituitary.body' },
  thyroid:     { name: 'Thyroid gland', region: 'neck', regionFact: 'sci.thyroid.body' },
  parathyroid: { name: 'Parathyroid glands', region: 'neck', regionFact: 'sci.parathyroid.body' },
  thymus:      { name: 'Thymus', region: 'thorax', regionFact: 'sci.thymus.body' },
  liver:       { name: 'Liver', region: 'abdomen', regionFact: 'sci.liver.body' },
  kidney:      { name: 'Kidneys', region: 'abdomen', regionFact: 'sci.kidney.body' },
  pancreas:    { name: 'Pancreas', region: 'abdomen', regionFact: 'sci.pancreas.body' },
  spleen:      { name: 'Spleen', region: 'abdomen', regionFact: 'sci.spleen.body' },
  lymphatic:   { name: 'Lymphatic system', region: 'whole-body', regionFact: 'sci.lymphatic.body' },
  adrenal:     { name: 'Adrenal glands', region: 'abdomen', regionFact: 'sci.adrenal.body' },
  testis:      { name: 'Testes', region: 'scrotum', regionFact: 'sci.testis.body' },
  ovary:       { name: 'Ovaries', region: 'pelvis', regionFact: 'sci.ovary.body' },
});

export const SCIENCE_FACTS = Object.freeze([
  // ---- pineal
  f('sci.pineal.body', 'organ:pineal', 'body', 'A tiny endocrine gland in the brain, inferior and somewhat posterior to the thalamus.', ['osx-17.7'], '§17.7'),
  f('sci.pineal.organ', 'organ:pineal', 'organ', 'Its functions are not entirely clear; melatonin secretion varies with the level of light received from the environment.', ['osx-17.7'], '§17.7'),
  f('sci.pineal.cell', 'organ:pineal', 'cell', 'Pinealocytes produce and secrete the amine hormone melatonin, which is derived from serotonin.', ['osx-17.7'], '§17.7'),
  f('sci.pineal.aa', 'organ:pineal', 'amino-acid', 'Melatonin is an amine hormone derived from the amino acid tryptophan; it helps regulate circadian rhythm.', ['osx-17.2'], '§17.2 Amine hormones'),
  // ---- pituitary
  f('sci.pituitary.body', 'organ:pituitary', 'body', 'A bean-sized organ suspended from the hypothalamus by the infundibulum, cradled in the sella turcica of the sphenoid bone of the skull.', ['osx-17.3'], '§17.3'),
  f('sci.pituitary.organ', 'organ:pituitary', 'organ', 'Two lobes of different embryonic origin: the posterior pituitary (neurohypophysis) is neural tissue; the anterior pituitary (adenohypophysis) is glandular tissue from the primitive digestive tract.', ['osx-17.3'], '§17.3'),
  f('sci.pituitary.function', 'organ:pituitary', 'organ', 'The posterior lobe releases two hormones; the anterior lobe produces six, whose release the hypothalamus regulates.', ['osx-17.3'], '§17.3 Learning Objectives'),
  f('sci.pituitary.aa', 'organ:pituitary', 'amino-acid', 'Antidiuretic hormone (ADH), a pituitary hormone important in fluid balance, is a peptide hormone: a chain of amino acids translated from mRNA.', ['osx-17.2'], '§17.2 Peptide hormones'),
  // ---- thyroid
  f('sci.thyroid.body', 'organ:thyroid', 'body', 'A butterfly-shaped organ located anterior to the trachea, just inferior to the larynx.', ['osx-17.4'], '§17.4'),
  f('sci.thyroid.organ', 'organ:thyroid', 'organ', 'A medial isthmus flanked by left and right lobes; the tissue is mostly thyroid follicles, each a cavity of colloid.', ['osx-17.4'], '§17.4'),
  f('sci.thyroid.cell', 'organ:thyroid', 'cell', 'Epithelial follicle cells surround the colloid, the centre of thyroid hormone production, which depends on iodine. Parafollicular (C) cells between follicles produce calcitonin.', ['osx-17.4'], '§17.4'),
  f('sci.thyroid.aa', 'organ:thyroid', 'amino-acid', 'Iodine is linked to tyrosine residues of thyroglobulin to make triiodothyronine (T3) and thyroxine (T4); thyroid hormones are tyrosine derivatives.', ['osx-17.4', 'osx-17.2'], '§17.4; §17.2'),
  // ---- parathyroid
  f('sci.parathyroid.body', 'organ:parathyroid', 'body', 'Small glands embedded in the posterior surface of the thyroid gland.', ['osx-17.5'], '§17.5 Fig. 17.15'),
  f('sci.parathyroid.cell', 'organ:parathyroid', 'cell', 'Chief cells produce parathyroid hormone (PTH), the major hormone regulating blood calcium; rising calcium inhibits further PTH release.', ['osx-17.5'], '§17.5'),
  // ---- thymus
  f('sci.thymus.body', 'organ:thymus', 'body', 'The thymus lies above the heart.', ['osx-21.1'], '§21.1 Fig. 21.7'),
  f('sci.thymus.organ', 'organ:thymus', 'organ', 'A primary lymphoid organ; larger and more active in infancy and early childhood, it atrophies with age.', ['osx-21.1', 'osx-17.10'], '§21.1; §17.10'),
  f('sci.thymus.cell', 'organ:thymus', 'cell', 'T cells migrate from bone marrow to the thymus, where they mature; thymosins contribute to T-lymphocyte development.', ['osx-21.1', 'osx-17.10'], '§21.1; §17.10'),
  // ---- liver
  f('sci.liver.body', 'organ:liver', 'body', 'Lies inferior to the diaphragm in the right upper quadrant of the abdominal cavity, protected by the ribs.', ['osx-23.6'], '§23.6'),
  f('sci.liver.organ', 'organ:liver', 'organ', 'The largest gland in the body (about three pounds in an adult); it produces bile and has many roles in metabolism and regulation.', ['osx-23.6'], '§23.6'),
  f('sci.liver.cell', 'organ:liver', 'cell', 'Hepatocytes, about 80% of liver volume, form plates (hepatic laminae) radiating from the central vein of each hepatic lobule; bile canaliculi run between them.', ['osx-23.6'], '§23.6 Histology'),
  // ---- kidney
  f('sci.kidney.body', 'organ:kidney', 'body', 'The kidneys lie on either side of the spine in the retroperitoneal space, protected by muscle, fat and ribs.', ['osx-25.3'], '§25.3'),
  f('sci.kidney.organ', 'organ:kidney', 'organ', 'About fist-sized and well vascularized, receiving about 25% of cardiac output at rest; renal pyramids and papillae lie in the medulla.', ['osx-25.3'], '§25.3'),
  f('sci.kidney.cell', 'organ:kidney', 'cell', 'Nephrons make urine, which collecting ducts carry through the papillae to the calyces.', ['osx-25.3'], '§25.3'),
  // ---- pancreas
  f('sci.pancreas.body', 'organ:pancreas', 'body', 'A long, slender organ, mostly posterior to the bottom half of the stomach.', ['osx-17.9'], '§17.9'),
  f('sci.pancreas.organ', 'organ:pancreas', 'organ', 'Primarily an exocrine gland secreting digestive enzymes, with an endocrine function in the pancreatic islets.', ['osx-17.9'], '§17.9'),
  f('sci.pancreas.cell', 'organ:pancreas', 'cell', 'Islet beta cells produce insulin and alpha cells produce glucagon, regulating glucose metabolism; acinar cells secrete digestive enzymes.', ['osx-17.9'], '§17.9'),
  // ---- spleen + lymphatic
  f('sci.spleen.body', 'organ:spleen', 'body', 'About 12 cm long and attached to the lateral border of the stomach by the gastrosplenic ligament.', ['osx-21.1'], '§21.1 Spleen'),
  f('sci.spleen.organ', 'organ:spleen', 'organ', 'A major secondary lymphoid organ, fragile and dark red from its vascularization, divided into red pulp and white pulp.', ['osx-21.1'], '§21.1 Spleen'),
  f('sci.lymphatic.body', 'organ:lymphatic', 'body', 'The system of vessels, cells and organs that carries excess fluid to the bloodstream and filters pathogens from the blood.', ['osx-21.1'], '§21.1'),
  f('sci.lymphatic.cell', 'organ:lymphatic', 'cell', 'B cells (maturing in red bone marrow) and T cells (maturing in the thymus) circulate in blood and lymph and reside in lymph nodes and spleen.', ['osx-21.1'], '§21.1'),
  // ---- adrenal
  f('sci.adrenal.body', 'organ:adrenal', 'body', 'Wedges of glandular and neuroendocrine tissue adhering to the top of the kidneys by a fibrous capsule.', ['osx-17.6'], '§17.6'),
  f('sci.adrenal.organ', 'organ:adrenal', 'organ', 'An outer cortex of glandular tissue (zona glomerulosa, fasciculata, reticularis) and an inner medulla of nervous tissue.', ['osx-17.6'], '§17.6'),
  f('sci.adrenal.cell', 'organ:adrenal', 'cell', 'The cortex secretes steroid hormones (aldosterone, cortisol); the medulla secretes epinephrine and norepinephrine for the fight-or-flight response.', ['osx-17.6'], '§17.6 Table 17.5'),
  f('sci.adrenal.aa', 'organ:adrenal', 'amino-acid', 'Epinephrine and norepinephrine are catecholamines derived from tyrosine; cortical steroid hormones derive from cholesterol, not from an amino acid.', ['osx-17.2'], '§17.2'),
  // ---- gonads
  f('sci.testis.body', 'organ:testis', 'body', 'Each testis descends through the abdominal musculature into the scrotal cavity during the seventh fetal month.', ['osx-27.1'], '§27.1'),
  f('sci.testis.organ', 'organ:testis', 'organ', 'Septa divide each testis into 300–400 lobules in which sperm develop inside seminiferous tubules.', ['osx-27.1'], '§27.1'),
  f('sci.testis.cell', 'organ:testis', 'cell', 'Leydig (interstitial) cells between the tubules produce testosterone; Sertoli cells surround the developing sperm.', ['osx-27.1'], '§27.1'),
  f('sci.ovary.body', 'organ:ovary', 'body', 'The ovaries are the female gonads; the ovarian reproductive system lies primarily inside the pelvic cavity.', ['osx-27.2'], '§27.2'),
  f('sci.ovary.cell', 'organ:ovary', 'cell', 'Oocytes sit in follicles; primordial follicles have a single flat layer of granulosa cells.', ['osx-27.2'], '§27.2'),
  f('sci.gonad.aa', 'organ:testis', 'amino-acid', 'Testosterone and the estrogens are steroid hormones derived from cholesterol, not from an amino acid.', ['osx-17.2', 'osx-17.8'], '§17.2; §17.8'),
  // ---- shared lower scales (every nucleated body cell)
  f('sci.cell.nucleus', 'scale:cell', 'cell', 'The nucleus is the largest organelle, enclosed by a double-membrane nuclear envelope with nuclear pores.', ['osx-3.3'], '§3.3'),
  f('sci.dna.chromatin', 'scale:dna', 'dna', 'Inside the nucleus, DNA is wrapped around histone proteins (nucleosomes) to form chromatin.', ['osx-3.3'], '§3.3'),
  f('sci.dna.helix', 'scale:dna', 'dna', 'DNA is an antiparallel double helix with sugar-phosphate backbones; its bases pair A–T and C–G.', ['osx-3.3'], '§3.3'),
  f('sci.codon.def', 'scale:codon', 'codon', 'A codon is a three-base sequence of mRNA that encodes an amino acid; DNA triplets are transcribed into mRNA codons.', ['osx-3.4'], '§3.4'),
  f('sci.codon.table', 'scale:codon', 'codon', 'The standard code has 64 codons: 61 encode the 20 amino acids and 3 (UAA, UAG, UGA) are stops. AUG encodes methionine and is the initiation codon; UUG and CUG can rarely initiate.', ['ncbi-gc1'], 'transl_table=1'),
  f('sci.aa.translation', 'scale:amino-acid', 'amino-acid', 'tRNA brings each amino acid to the ribosome, matching its anticodon to the mRNA codon, until a stop codon releases the finished polypeptide.', ['osx-3.4'], '§3.4 Translation'),
  f('sci.aa.names', 'scale:amino-acid', 'amino-acid', 'Names and one/three-letter symbols of the 20 genetically encoded amino acids follow IUPAC-IUB nomenclature.', ['iupac-aa'], '3AA-1 Table 1'),
]);

/** Science says which amino acid (if any) a gland's signature hormones are made from. Used by comparisons, never as HD evidence. */
export const HORMONE_PRECURSORS = Object.freeze({
  pineal:  { aminoAcids: ['W'], fact: 'sci.pineal.aa' },
  thyroid: { aminoAcids: ['Y'], fact: 'sci.thyroid.aa' },
  adrenal: { aminoAcids: ['Y'], fact: 'sci.adrenal.aa', note: 'medulla only; cortex steroids are cholesterol-derived' },
});
