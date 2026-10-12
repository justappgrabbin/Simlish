# Verification before application

Open `verification.html` from the studio's “Investigate candidate rules” link.

Candidates, evidence records and findings are separate. The investigation automatically rechecks on opening/import and when candidates or evidence are added. It checks every supported candidate independently, searches retained evidence for unresolved candidates, records check results and scope, and persists its records locally. Export retains the records outside browser storage. Candidate revisions and evidence changes invalidate prior findings; reloaded findings require rerunning reproducible checks. Imported status labels cannot establish verification.

Implemented verifiers:

- Formal arithmetic (sum/product/quotient) and binary-decimal calculations: independently reproduce a specified proposition and mark the check verified or refuted. This verifies only that mathematical calculation, not additional claims in its free-text description.
- Exact source comparison: supports or contradicts the transcription. It cannot establish causal truth or be applied as empirical verification.
- Empirical claims: retained as unverified. Local evidence discovery finds potentially relevant records; no empirical verifier, measurement instrument, automatic browser research or model-based science investigator is implemented yet.

The supplied causal-chain, gate-anchor, line-trajectory and resonance-weight statements remain candidate hypotheses. `traverseCausal` blocks runtime execution until the corresponding empirical findings exist. Explicit exploration returns a hypothesis trace with `emitted:false`; it does not certify the trace or admit a resident. The previous scene preset still changes its visual theme as an explicit app command; its causal action receipt now reports awaiting verification. No display color or example frequency is promoted to a causal finding.

All five screenshot chains and microcosmic lists remain transcriptions with source provenance. Their binary relationships, differing orders and emergent field interactions are not fully implemented by those lists. Testing code conformance does not verify the physical or causal claims.

Current checks cover correct/incorrect formal results, source-vs-causality separation, evidence lookup, revision invalidation, imported-finding recheck and causal execution blocking. Real mobile-size Chromium checks local persistence/reload, rechecking and an incorrect calculation being refuted, with no page errors. Physical Android testing remains outstanding.
