# HD morphology: body and science layer

This is a separate, self-contained module. It is pure JavaScript (ES modules) with no backend, no LLM, no build step and no network calls at runtime.
It sets Human Design's body correspondences next to published science. The two are kept apart, labelled, and cited.

## View

```
cd you-n-i-verse/modules/hd-morphology
python3 -m http.server 8731 --bind 127.0.0.1
# open http://127.0.0.1:8731/
```

To use the body map, click or Tab to one of the 9 centers or 13 organ markers and press Enter. The scale stepper then zooms through **body → organ/gland → cell → DNA → codon → amino acid**. Esc steps back up one scale. The 64-gate grid opens each gate's page. Each gate page shows the HD claims, the science for the amino acid HD names, your four project tables, and any flags. The page also has the generated mismatch list, the full NCBI codon table and the sources table.

## Layers (data and UI)

| Layer | File | Rule |
|---|---|---|
| ESTABLISHED SCIENCE | `data/science.mjs`, `data/genetic-code.mjs` | Every fact cites ≥1 `science` source plus a section locator. No HD wording. |
| HUMAN DESIGN CORRESPONDENCE | `data/hd.mjs` (built from `data/her-sources.mjs`) | `claim` = HD literature in your files. `variant` = your own project tables, shown beside the claims and never merged into them. |
| UNSOURCED / FILLED-IN | `data/unsourced.mjs` | No citation. Each item gives a `reason`. These are the region words for center placement and all drawings. |
| SOURCES | `data/sources.mjs` | The citation table that every fact refers to by id. |

Mismatches are produced by `src/mismatches.mjs` from the data. The prose in the list is generated, not hardcoded, and the tests change the data and check that the list changes. The API is `src/index.mjs`: `describeGate(n)`, `describeCenter(id)`, `mismatches()`, `translateCodon()` and more. It imports with no DOM. `src/view.mjs` exports `mount(document)` for wiring into another page.

## Your material used (read-only copies; originals untouched)
- **Black Book** (Ra Uru Hu 1992): the per-gate `Center:` and `Biology:` fields. These give gate→center and center→gland.
- **Complete Rave I'Ching** (2001): the "The Gate of …" keynote for each gate. It also says gate 41 is "the only start codon" and gate 56 is a "stop codon".
- **Gene Keys** (Rudd), which is in the same local PDF. For each key it gives PHYSIOLOGY, CODON RING and AMINO ACID. This is the only HD-family source found that assigns an amino acid to every gate.
- **Zhouyi** (Rutt 1996): the header of each hexagram.
- **Synthia project tables**: `kingwen.js` (line patterns), `centers-channels.js`, the `human-design.js` donor port, and from Drive `codon and body.zip` (GateCodon Codex CSV and centers_from_doc.csv) and `dna-hd-compiler`. Also the **SynthAi v0.5.3** gate→codon digram rule at commit 6d4fcd1.

You can regenerate `data/her-sources.mjs` on the Synthia box. Run `python3 tools/extract-her-sources.py`, then `node tools/extract-synthai-rule.mjs`, then `node tools/build-data.mjs`.

## Unresolved / missing
- **Gate → exact codon triplet:** none of the HD literature found states one, so it is marked UNRESOLVED. Two of your own project rules each give a triplet: dna-hd-compiler, and the SynthAi digram rule. They disagree with each other and with HD's start and stop gates. Both are shown as PROJECT variants.
- **Bantu:** the Bantu texts found are Katushemererwe (2013) and Creissels (2019), plus the B.A.N.T.U. bridge zip. They are about linguistic morphology or a text encoding. None of them has body, codon or HD content, so they are listed as "checked, not used".
- Cell-level facts are still missing for some organs: no ovarian hormone cell type, and no spleen cell type beyond the pulp.

## Tests
`node --test test/`
