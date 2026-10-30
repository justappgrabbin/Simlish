/**
 * Computer Capability Registry — loads computer/registry/capability-registry.json
 * (capability-registry-v1 schema) and serves queryCapability(name).
 *
 * Statuses: NOT_FOUND | PRESENT | PARTIALLY_WIRED | WIRED | VERIFIED.
 * WIRED/VERIFIED require verification_evidence — markStatus() refuses to
 * promote without it. Evidence is append-only; demotion is not supported
 * (unknown stays unknown, and claimed statuses are never silently lowered).
 */

const SEED_URL = new URL('./capability-registry.json', import.meta.url);
const STATUSES = Object.freeze(['NOT_FOUND', 'PRESENT', 'PARTIALLY_WIRED', 'WIRED', 'VERIFIED']);

export class CapabilityRegistryService {
  constructor({ bus = null, seedPath = null } = {}) {
    this.bus = bus;
    this.seedPath = seedPath;
    this.capabilities = new Map();
  }

  async load() {
    const url = this.seedPath ?? SEED_URL;
    let doc;
    if (typeof window !== 'undefined') {
      const response = await fetch(url);
      if (!response.ok) throw new Error('Computer capability registry could not load: HTTP ' + response.status);
      doc = await response.json();
    } else {
      // Keep the established Node JSON seed and avoid static Node imports in browsers.
      const { readFile } = await import('node:fs/promises');
      const { fileURLToPath } = await import('node:url');
      doc = JSON.parse(await readFile(url instanceof URL ? fileURLToPath(url) : url, 'utf8'));
    }
    for (const entry of doc.capabilities ?? []) {
      this.capabilities.set(entry.capability, entry);
    }
    this.meta = { schema: doc.schema, stage: doc.stage, donor_pin: doc.donor_pin };
    this.bus?.emit('capability-registry:loaded', { capabilities: [...this.capabilities.keys()], seed: String(url) });
    return this;
  }

  /** Contract: queryCapability(name) -> capability entry with provider statuses. */
  queryCapability(name) {
    const entry = this.capabilities.get(name) ?? null;
    return {
      capability: name,
      found: Boolean(entry),
      providers: (entry?.providers ?? []).map((p) => ({
        provider_id: p.provider_id,
        status: p.status,
        repository: p.repository,
        source_artifact: p.source_artifact,
        last_verified: p.last_verified ?? null,
        verification_evidence: p.verification_evidence ?? [],
        known_limitations: p.known_limitations ?? [],
        lineage: p.lineage ?? { derived_from: [], supersedes_for_scope: [] },
      })),
      description: entry?.description ?? null,
    };
  }

  listCapabilities() {
    return [...this.capabilities.keys()].map((name) => this.queryCapability(name));
  }

  /**
   * Contract: route(capability, ctx) -> routing_decision record.
   * Prefers VERIFIED > WIRED > PARTIALLY_WIRED > PRESENT providers.
   */
  route(capability, ctx = {}) {
    const q = this.queryCapability(capability);
    const candidates = q.providers;
    const rank = (s) => STATUSES.indexOf(s);
    const selected = [...candidates].sort((a, b) => rank(b.status) - rank(a.status))[0] ?? null;
    const decision = {
      request_id: ctx.request_id ?? `req-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
      required_capability: capability,
      candidate_providers: candidates.map((c) => c.provider_id),
      selected_provider: selected?.provider_id ?? null,
      selection_rationale: selected
        ? `highest status among candidates (${selected.status})`
        : 'no provider found',
      timestamp: new Date().toISOString(),
      outcome_event_id: ctx.outcome_event_id ?? null,
    };
    this.bus?.emit('capability-registry:routing-decision', decision);
    return decision;
  }

  /**
   * Promote a provider status with REQUIRED evidence. Append-only:
   * promotions only move forward, and WIRED/VERIFIED without evidence throws.
   */
  markStatus(capability, providerId, status, evidence = null) {
    if (!STATUSES.includes(status)) throw new Error(`unknown status: ${status}`);
    const entry = this.capabilities.get(capability);
    if (!entry) throw new Error(`unknown capability: ${capability}`);
    const provider = entry.providers.find((p) => p.provider_id === providerId);
    if (!provider) throw new Error(`unknown provider ${providerId} for ${capability}`);
    if ((status === 'WIRED' || status === 'VERIFIED') && !(Array.isArray(evidence) && evidence.length)) {
      throw new Error(`status ${status} requires verification_evidence`);
    }
    if (STATUSES.indexOf(status) < STATUSES.indexOf(provider.status)) {
      throw new Error(`status demotion refused (${provider.status} -> ${status}); registry is append-only`);
    }
    provider.status = status;
    if (evidence?.length) {
      provider.verification_evidence = [...(provider.verification_evidence ?? []), ...evidence];
      provider.last_verified = new Date().toISOString();
    }
    this.bus?.emit('capability-registry:status', { capability, provider_id: providerId, status });
    return provider;
  }
}

export default CapabilityRegistryService;
