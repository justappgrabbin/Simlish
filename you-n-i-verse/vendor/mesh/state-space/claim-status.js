// Pure Synthia Automata — state-space: claim-type system as code
// (Synthia OS Canonical Integration Handoff, 05_STATE_SPACE_REPAIR_CONTRACT.md;
// reconciliation with docs/MAPPING_AUDIT.md recorded in docs/HANDOFF_INTEGRATION.md)

/**
 * The repair contract's claim classification (§"Classification required for
 * every mapping/formula") as an executable vocabulary. Every mapping/formula
 * value that could be mistaken for source-grounded dimensional truth should be
 * carried as a claim: `{ value, status, source, hypothesisId, confidence,
 * evidence }` — the shape docs/MAPPING_AUDIT.md §1 asks for and
 * primitive-dimensions.js `dimensionOf` already approximates with
 * `{basis, hypothesisId}`.
 *
 * Tag correspondence (handoff -> MAPPING_AUDIT):
 *   SOURCE_STATEMENT        -> SOURCE_STATEMENT
 *   STRUCTURAL_MATH         -> STRUCTURAL_MATH
 *   DERIVED                 -> DERIVED_RESULT
 *   PROJECT_HYPOTHESIS      -> PROJECT_HYPOTHESIS
 *   RENDERER_CONVENTION     -> RENDERER_CONVENTION
 *   LEARNED                 -> (audit folds this into EXPERIMENTAL_RESULT adjacency)
 *   EMPIRICALLY_SUPPORTED   -> EXPERIMENTAL_RESULT (pre-registered, controls beaten)
 *   CONFLICT                -> CONFLICT
 *   CONTROL_ONLY            -> (new in the handoff — audit §3a tags these
 *                              PROJECT_HYPOTHESIS; the handoff tag is stricter
 *                              and is adopted here)
 *
 * Pure JS, zero deps, deterministic (no wall-clock, no randomness).
 */

import { controlMappings } from '../experiments/scale/controls.js';

export const CLAIM_STATUS = Object.freeze({
  SOURCE_STATEMENT: 'SOURCE_STATEMENT',
  STRUCTURAL_MATH: 'STRUCTURAL_MATH',
  DERIVED: 'DERIVED',
  PROJECT_HYPOTHESIS: 'PROJECT_HYPOTHESIS',
  RENDERER_CONVENTION: 'RENDERER_CONVENTION',
  LEARNED: 'LEARNED',
  EMPIRICALLY_SUPPORTED: 'EMPIRICALLY_SUPPORTED',
  CONFLICT: 'CONFLICT',
  CONTROL_ONLY: 'CONTROL_ONLY',
});

const CLAIM_STATUS_VALUES = Object.freeze(Object.values(CLAIM_STATUS));

export function isClaimStatus(status) {
  return CLAIM_STATUS_VALUES.includes(status);
}

/**
 * Wrap a value with its epistemic provenance. The wrapper is frozen; `evidence`
 * (when an object/array) is deep-frozen. `status` must be one of CLAIM_STATUS.
 * SOURCE_STATEMENT claims should carry `source` (a citation string); the
 * factory enforces it because a source statement without a source is exactly
 * the failure mode the contract exists to prevent.
 */
export function claim(value, { status, source = null, hypothesisId = null, confidence = null, evidence = null } = {}) {
  if (!isClaimStatus(status)) {
    throw new RangeError(`claim: status must be one of ${CLAIM_STATUS_VALUES.join(', ')} (got ${JSON.stringify(status)})`);
  }
  if (status === CLAIM_STATUS.SOURCE_STATEMENT && (typeof source !== 'string' || !source)) {
    throw new TypeError('claim: SOURCE_STATEMENT requires a source citation string');
  }
  const record = { value, status, source, hypothesisId, confidence };
  if (evidence !== null && evidence !== undefined) record.evidence = evidence;
  return deepFreeze(record);
}

export function isClaim(x) {
  return x !== null && typeof x === 'object'
    && Object.prototype.hasOwnProperty.call(x, 'value')
    && isClaimStatus(x.status);
}

/** The status carried by a claim, or null for bare (unlabeled) values. */
export function resolveClaimStatus(claimObj) {
  return isClaim(claimObj) ? claimObj.status : null;
}

/* ------------------------- resolution order (contract §2) -------------------------
 *
 * Primitive -> dimension matching must not come from array position.
 * Resolution order (05_STATE_SPACE_REPAIR_CONTRACT.md §2):
 *   1. direct source anchor                        -> SOURCE_STATEMENT
 *   2. explicit mechanical derivation              -> STRUCTURAL_MATH, DERIVED
 *   3. context-conditioned candidate               -> RENDERER_CONVENTION
 *   4. project hypothesis                          -> PROJECT_HYPOTHESIS
 *   5. learned/tested mapping                      -> LEARNED, EMPIRICALLY_SUPPORTED
 *   6. null when unresolved
 *
 * CONFLICT and CONTROL_ONLY are NEVER resolution outcomes: CONFLICT entries
 * stay preserved (canon-registry.js keeps them first-class) and CONTROL_ONLY
 * conditions exist to be beaten by candidates, not to answer questions
 * (contract §1: modulo mappings are CONTROL_ONLY, never canonical ontology).
 */
export const RESOLUTION_ORDER = Object.freeze([
  Object.freeze([CLAIM_STATUS.SOURCE_STATEMENT]),
  Object.freeze([CLAIM_STATUS.STRUCTURAL_MATH, CLAIM_STATUS.DERIVED]),
  Object.freeze([CLAIM_STATUS.RENDERER_CONVENTION]),
  Object.freeze([CLAIM_STATUS.PROJECT_HYPOTHESIS]),
  Object.freeze([CLAIM_STATUS.LEARNED, CLAIM_STATUS.EMPIRICALLY_SUPPORTED]),
]);

// Statuses that can never resolve a canon question by themselves.
export const NON_RESOLVING = Object.freeze([CLAIM_STATUS.CONFLICT, CLAIM_STATUS.CONTROL_ONLY]);

/**
 * Honest resolver over candidate claims for one question (e.g. "which
 * dimension does primitive X belong to"). Returns the first candidate at the
 * highest tier that clears the bar, or null when nothing does (contract §7:
 * unknown is a valid state). Non-claim bare values never clear the bar.
 */
export function resolveDimension(candidates) {
  const claims = [...(candidates || [])].filter((c) => isClaim(c) && !NON_RESOLVING.includes(c.status));
  for (const tier of RESOLUTION_ORDER) {
    const hit = claims.find((c) => tier.includes(c.status));
    if (hit) return hit;
  }
  return null;
}

/* ------------------------- promotion gate (contract §10) -------------------------
 *
 * A candidate dimensional mapping becomes EMPIRICALLY_SUPPORTED only if it
 * beats pre-registered controls — random, shuffled, modulo/control, ablated —
 * on held-out reconstruction/generation/transfer metrics. The control battery
 * itself is controlMappings() from src/experiments/scale/controls.js: its
 * non-'source' conditions are exactly the controls named by the contract and
 * by 06_DIMENSION_CANON.json's promotion_rule.requires.
 */

// The required control conditions, derived from the actual control battery so
// this gate cannot drift from the experiment layer: every controlMappings
// condition except 'source' (the candidate itself) must be beaten.
const REQUIRED_CONTROLS = Object.freeze(
  Object.keys(controlMappings({ a: 1, b: 2 }).conditions).filter((k) => k !== 'source').sort(),
);

export const PROMOTION_GATE = Object.freeze({
  from: CLAIM_STATUS.PROJECT_HYPOTHESIS,
  to: CLAIM_STATUS.EMPIRICALLY_SUPPORTED,
  requiredControls: REQUIRED_CONTROLS, // ['ablated','modulo','random','shuffled'] — from controlMappings
  requires: Object.freeze([
    'pre-registered mapping',
    'held-out test',
    'random/shuffled/modulo controls',
    'ablation',
    'reproducible derivations',
    'non-regression on canonical runtime',
  ]),
});

/**
 * PROMOTION_GATE predicate: may `candidateClaim` be asserted
 * EMPIRICALLY_SUPPORTED on the basis of `evidence`?
 *
 * evidence: {
 *   preRegistered: boolean,          // mapping registered before the test ran
 *   heldOut: boolean,                // evaluated on held-out data
 *   beatenControls: string[],        // control ids beaten (must cover all of
 *                                    // PROMOTION_GATE.requiredControls)
 *   reproducible?: boolean,          // derivations reproduce
 *   nonRegressing?: boolean,         // canonical runtime suite still green
 * }
 *
 * Refuses (returns false) when the candidate is not a PROJECT_HYPOTHESIS /
 * LEARNED claim, when evidence is missing, or when any required pre-registered
 * control was not beaten. Unknown is a valid state; false is not a verdict of
 * falsity, only of insufficient warrant.
 */
export function passesPromotionGate(candidateClaim, evidence = null) {
  if (!isClaim(candidateClaim)) return false;
  if (![CLAIM_STATUS.PROJECT_HYPOTHESIS, CLAIM_STATUS.LEARNED].includes(candidateClaim.status)) return false;
  if (!evidence || typeof evidence !== 'object') return false;
  if (evidence.preRegistered !== true || evidence.heldOut !== true) return false;
  const beaten = new Set(Array.isArray(evidence.beatenControls) ? evidence.beatenControls : []);
  for (const control of PROMOTION_GATE.requiredControls) {
    if (!beaten.has(control)) return false;
  }
  if (evidence.reproducible === false || evidence.nonRegressing === false) return false;
  return true;
}

/**
 * Consistency check for claims already tagged EMPIRICALLY_SUPPORTED: the tag
 * is only honest if the gate is satisfied by the claim's own evidence field.
 */
export function isHonestlyEmpirical(claimObj) {
  if (!isClaim(claimObj) || claimObj.status !== CLAIM_STATUS.EMPIRICALLY_SUPPORTED) return false;
  const evidence = claimObj.evidence;
  if (!evidence || typeof evidence !== 'object') return false;
  return passesPromotionGate(
    claim(claimObj.value, {
      status: CLAIM_STATUS.PROJECT_HYPOTHESIS,
      source: claimObj.source ?? null,
      hypothesisId: claimObj.hypothesisId ?? null,
      confidence: claimObj.confidence ?? null,
    }),
    evidence,
  );
}

function deepFreeze(x) {
  if (x && typeof x === 'object' && !Object.isFrozen(x)) {
    for (const v of Object.values(x)) deepFreeze(v);
    Object.freeze(x);
  }
  return x;
}
