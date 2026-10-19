// SOURCES layer: every fact in this module references one or more of these ids.
// kind: 'science'      = published, peer-reviewed / reference science (ESTABLISHED SCIENCE layer may cite only these)
//       'hd'           = Human Design (or HD-derived) literature from Alexis's own files (HD CORRESPONDENCE layer)
//       'her-project'  = Alexis's own Synthia/Simlish code or data tables (project models, not science, not HD canon)
//       'checked-not-used' = searched/read, contains no body/codon material usable here
// URLs were opened and their page titles verified on 2026-10-19 (America/Los_Angeles).
const OS = 'https://openstax.org/books/anatomy-and-physiology-2e/pages/';
const osx = (id, section, slug) => ({
  id, kind: 'science', title: `Anatomy and Physiology 2e, section ${section}`,
  author: 'Betts JG, Young KA, Wise JA, Johnson E, Poe B, Kruse DH, Korol O, Johnson JE, Womble M, DeSaix P',
  publisher: 'OpenStax, Rice University', year: 2022, url: OS + slug, accessed: '2026-10-19',
});
export const SOURCES = Object.freeze([
  { id: 'ncbi-gc1', kind: 'science', title: 'The Genetic Codes — 1. The Standard Code (transl_table=1)',
    author: 'Elzanowski A, Ostell J (NCBI)', publisher: 'National Center for Biotechnology Information', year: 2024,
    url: 'https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi?chapter=tgencodes#SG1', accessed: '2026-10-19',
    note: 'Last update of the Genetic Codes: Sep. 23, 2024. Table uses T for U.' },
  { id: 'iupac-aa', kind: 'science', title: 'Nomenclature and Symbolism for Amino Acids and Peptides, 3AA-1 Table 1',
    author: 'IUPAC-IUB Joint Commission on Biochemical Nomenclature (JCBN)', publisher: 'IUPAC (web version, Queen Mary University of London)', year: 1983,
    url: 'https://iupac.qmul.ac.uk/AminoAcid/AA1n2.html', accessed: '2026-10-19' },
  osx('osx-3.3', '3.3 The Nucleus and DNA Replication', '3-3-the-nucleus-and-dna-replication'),
  osx('osx-3.4', '3.4 Protein Synthesis', '3-4-protein-synthesis'),
  osx('osx-17.2', '17.2 Hormones', '17-2-hormones'),
  osx('osx-17.3', '17.3 The Pituitary Gland and Hypothalamus', '17-3-the-pituitary-gland-and-hypothalamus'),
  osx('osx-17.4', '17.4 The Thyroid Gland', '17-4-the-thyroid-gland'),
  osx('osx-17.5', '17.5 The Parathyroid Glands', '17-5-the-parathyroid-glands'),
  osx('osx-17.6', '17.6 The Adrenal Glands', '17-6-the-adrenal-glands'),
  osx('osx-17.7', '17.7 The Pineal Gland', '17-7-the-pineal-gland'),
  osx('osx-17.8', '17.8 Gonadal and Placental Hormones', '17-8-gonadal-and-placental-hormones'),
  osx('osx-17.9', '17.9 The Endocrine Pancreas', '17-9-the-endocrine-pancreas'),
  osx('osx-17.10', '17.10 Organs with Secondary Endocrine Functions', '17-10-organs-with-secondary-endocrine-functions'),
  osx('osx-21.1', '21.1 Anatomy of the Lymphatic and Immune Systems', '21-1-anatomy-of-the-lymphatic-and-immune-systems'),
  osx('osx-23.6', '23.6 Accessory Organs in Digestion: The Liver, Pancreas, and Gallbladder', '23-6-accessory-organs-in-digestion-the-liver-pancreas-and-gallbladder'),
  osx('osx-25.3', '25.3 Gross Anatomy of the Kidney', '25-3-gross-anatomy-of-the-kidney'),
  osx('osx-27.1', '27.1 Anatomy and Physiology of the Testicular Reproductive System', '27-1-anatomy-and-physiology-of-the-testicular-reproductive-system'),
  osx('osx-27.2', '27.2 Anatomy and Physiology of the Ovarian Reproductive System', '27-2-anatomy-and-physiology-of-the-ovarian-reproductive-system'),

  { id: 'hd-blackbook', kind: 'hd', title: 'The Human Design System ("The Black Book")', author: 'Ra Uru Hu', year: 1992,
    localPath: '/workspace/blackbook/blackbook.pdf (OCR: bb_gates.json)', drive: 'pdfcoffee.com-the-black-book.pdf (Drive id 1L7VD2PkU1yv-XDFKD2uNdPvcZYXUNr3D)',
    note: 'Per-gate "Center:" and "Biology:" fields. OCR text; values kept as printed.' },
  { id: 'hd-rave-iching', kind: 'hd', title: 'The Complete Rave I\u2019Ching', author: 'Ra Uru Hu', publisher: 'Jovian Archive Corporation', year: 2001,
    localPath: '/workspace/rave2001/combined.pdf (text: combined.txt, first part)', note: '"THE GATE OF …" keynotes and two codon statements (gates 41, 56).' },
  { id: 'hd-gene-keys', kind: 'hd', title: 'Gene Keys: Unlocking the Higher Purpose Hidden in Your DNA', author: 'Richard Rudd', year: null,
    localPath: '/workspace/rave2001/combined.pdf (text: combined.txt, from line "GENE KEYS")',
    note: 'HD-derived system. Per key: PHYSIOLOGY, CODON RING, AMINO ACID. Edition/year not printed in the local copy; first published 2009.' },
  { id: 'zhouyi-rutt', kind: 'hd', title: 'The Book of Changes (Zhouyi): A Bronze Age Document', author: 'Richard Rutt', publisher: 'Curzon', year: 1996,
    localPath: '/workspace/zhouyi/rutt_hex.json', drive: 'the-book-of-changes-zhouyi-a-bronze-age-document-0700704671_compress.pdf (Drive id 1OgSb-EljAqAhM5m-uNtcP-w_V5WhgZSy)',
    note: 'Hexagram headers only (pinyin + gloss). Not HD; the textual root of the 64 hexagrams HD uses as gates.' },

  { id: 'synthia-kingwen', kind: 'her-project', title: 'Synthia State-Space Foundation — src/merged/kingwen.js (King Wen ↔ Fu Xi table, hexagram names)',
    localPath: '/workspace/run-existing/foundation/Synthia-State-Space-Foundation/vendor/browser/src/merged/kingwen.js' },
  { id: 'synthia-centers', kind: 'her-project', title: 'Synthia State-Space Foundation — src/merged/centers-channels.js (36 channels, CENTERS)',
    localPath: '/workspace/run-existing/foundation/Synthia-State-Space-Foundation/vendor/browser/src/merged/centers-channels.js' },
  { id: 'synthia-hdjs', kind: 'her-project', title: 'Synthia State-Space Foundation — src/state-space/human-design.js (donor port: GATE_CENTER, GATES aminoAcid/organ)',
    localPath: '/workspace/run-existing/foundation/Synthia-State-Space-Foundation/vendor/browser/src/state-space/human-design.js',
    note: 'Its own header says the amino-acid/organ attributions are "donor attributions, not independently attested".' },
  { id: 'drive-codex', kind: 'her-project', title: 'HumanDesign_GateCodon_Codex_FULL.csv (in "codon and body.zip")',
    drive: 'codon and body.zip (Drive id 1Qz3XiWllSjMhJ31QvUd5rGswFkmaLxoK)' },
  { id: 'drive-centers-doc', kind: 'her-project', title: 'centers_from_doc.csv (in "codon and body.zip")',
    drive: 'codon and body.zip (Drive id 1Qz3XiWllSjMhJ31QvUd5rGswFkmaLxoK)' },
  { id: 'dna-hd-compiler', kind: 'her-project', title: 'dna-hd-compiler/src/hd/gates.ts ("custom 64-gate codon mapping")',
    drive: 'dna-hd-compiler.zip (Drive id 1Kzr_1fNTzDeZHkx3cmOjLozx2_zNiGLG)' },
  { id: 'synthai-digram', kind: 'her-project', title: 'SynthAi v0.5.3 biological-translation-runtime.mjs (gate bits → three digrams → codon)',
    repo: 'justappgrabbin/SynthAi', commit: '6d4fcd162a4d4c4feb3317e78ab7b5ca60893de6', path: 'computer/hover/src/state-space/biological-translation-runtime.mjs',
    note: 'Its own doc calls the mapping "the project model under test".' },

  { id: 'bantu-katushemererwe', kind: 'checked-not-used', title: 'Computational morphology and Bantu language learning: an implementation for Runyakitara',
    author: 'Fridah Katushemererwe', publisher: 'University of Groningen (PhD thesis)', year: 2013, localPath: '/workspace/bantu.txt',
    note: 'Linguistic morphology (word structure), not body morphology. No anatomy, codon or HD content.' },
  { id: 'bantu-creissels', kind: 'checked-not-used', title: 'Bantu languages: typology and variation (3rd draft)', author: 'Denis Creissels', year: 2019,
    localPath: '/workspace/bantu-extra/creissels.txt', note: 'Linguistic typology. No anatomy, codon or HD content.' },
  { id: 'bantu-bridge', kind: 'checked-not-used', title: 'B.A.N.T.U. × 3–2–1 Swarm Bridge v0.2 (Drive zip)', drive: 'BANTU-321-Swarm-Bridge-v0.2.zip (Drive id 1HBC0vEW0UR1b9BUUuJYdabcLuKZJZroJ)',
    note: 'A text-to-6-bit encoding demo; its README says it is not a hexagram-to-word semantic mapping. No body content.' },
]);
export const SOURCE_BY_ID = Object.freeze(Object.fromEntries(SOURCES.map((s) => [s.id, s])));
export default SOURCES;
