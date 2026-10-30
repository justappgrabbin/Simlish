import { EventBus, MemoryPersistence, StateStore, Registry } from './core/kernel.mjs';
import { VirtualFileSystem, ArtifactIntake } from './runtime/storage.mjs';
import { AutomataEngine, ProcessFabric, BackendBroker } from './runtime/execution.mjs';
import { CapabilityGraph, MorphEngine, CultivationEngine, PolicyEngine } from './runtime/experience.mjs';
import { ShellManager, MutationDispatcher } from './runtime/shells.mjs';
import { ProjectWorkspace } from './runtime/projects.mjs';
import { GitHubWorkspaceAdapter } from './adapters/GitHubWorkspaceAdapter.mjs';
import { registerCanonicalMicros } from './micros/MicroRegistry.mjs';
import { registerSystemAdapters } from './adapters/SystemAdapters.mjs';
import { CapabilityRegistryService } from './registry/capability-registry.mjs';
import { ComputerFormationService, FORMATION_PROVIDER } from './services/executable-formations.mjs';

export class ComputerRuntime {
  constructor({ persistence = new MemoryPersistence(), namespace = 'synthai-computer', github = null, eventLogPath = null, platform = typeof window === 'undefined' ? 'node' : 'browser' } = {}) {
    if (!['node', 'browser'].includes(platform)) throw new TypeError('Unknown Computer platform: ' + platform);
    this.platform = platform;
    this.namespace = namespace;
    this.eventLogPath = eventLogPath;
    this.bus = new EventBus();
    this.state = new StateStore({ bus: this.bus, persistence, namespace });
    const registry = kind => new Registry({ kind, bus: this.bus });

    this.tools = registry('tool');
    this.apps = registry('app');
    this.shells = registry('shell');
    this.agents = registry('agent');
    this.worlds = registry('world');
    this.automataRegistry = registry('automaton');
    this.processRegistry = registry('process');
    this.capabilityRegistry = registry('capability');
    this.backendsRegistry = registry('backend');
    this.artifacts = registry('artifact');
    this.micros = registry('micro');
    this.systems = registry('system');
    this.services = registry('service');

    this.vfs = new VirtualFileSystem({ bus: this.bus, state: this.state });
    this.shellManager = new ShellManager({ bus: this.bus, shells: this.shells, apps: this.apps, state: this.state });
    this.automata = new AutomataEngine({ bus: this.bus, automata: this.automataRegistry, tools: this.tools, state: this.state });
    this.processes = new ProcessFabric({ bus: this.bus, processes: this.processRegistry, state: this.state });
    this.backends = new BackendBroker({ bus: this.bus, backends: this.backendsRegistry });
    this.capabilities = new CapabilityGraph({ capabilities: this.capabilityRegistry });
    this.morph = new MorphEngine({ bus: this.bus, state: this.state, micros: this.micros });
    this.cultivation = new CultivationEngine({ bus: this.bus, state: this.state });
    this.policy = new PolicyEngine({ bus: this.bus });
    this.mutations = new MutationDispatcher({ bus: this.bus, state: this.state, shellManager: this.shellManager, automata: this.automata, tools: this.tools, agents: this.agents, morph: this.morph });
    this.intake = new ArtifactIntake({ bus: this.bus, vfs: this.vfs, artifacts: this.artifacts, mutations: this.mutations });
    this.projects = new ProjectWorkspace({ bus: this.bus, state: this.state, vfs: this.vfs, backends: this.backends });

    if (github) this.configureGitHub(github);
  }

  configureGitHub(config = {}) {
    const adapter = config instanceof GitHubWorkspaceAdapter ? config : new GitHubWorkspaceAdapter(config);
    this.backends.register('github', adapter, { replace: true });
    this.capabilityRegistry.register('github-project-publish', {
      providers: ['github'],
      requires: ['synthia-server'],
      description: 'Publish Computer project files through the authenticated Synthia Server GitHub bridge.'
    }, { replace: true });
    this.bus.emit('github:configured', { baseUrl: adapter.baseUrl, tokenConfigured: Boolean(adapter.token) });
    return adapter;
  }

  async boot() {
    await this.state.restore();
    if (!this.shells.has('workspace')) this.shells.register('workspace', { name: 'Workspace Shell', kind: 'generic' });
    if (!this.shells.has('hover-field')) this.shells.register('hover-field', { name: 'Hover Field Shell', kind: 'morph-field' });
    registerCanonicalMicros(this.micros);
    registerSystemAdapters(this);

    this.policy.register('xynthai:age-boundary', ({ ageConfirmed = false } = {}) =>
      ageConfirmed ? { allow: true } : { allow: false, reason: 'XynthAI requires an explicit 18+ age confirmation.' }
    );

    for (const id of ['artifact-intake', 'app-mounting', 'morph-expression', 'automata', 'cultivation', 'project-workspace']) {
      if (!this.capabilityRegistry.has(id)) this.capabilities.register(id, { providers: ['computer'] });
    }

    // One Computer, two platform-specific persistence implementations of the
    // existing emitEvent/readAll grammar-v1 contract. Browser never imports
    // Node-only providers (node:fs/child_process); Node keeps all donor mounts.
    if (this.platform === 'browser') {
      const { BrowserEventEmitter } = await import('./events/browser-event-emitter.mjs');
      this.eventEmitter = new BrowserEventEmitter({ bus: this.bus, namespace: this.namespace });
    } else {
      const { EventEmitter } = await import('./events/event-emitter.mjs');
      this.eventEmitter = new EventEmitter({
        bus: this.bus, ...(this.eventLogPath ? { logPath: this.eventLogPath } : {}),
      });
    }
    this.services.register('event-emitter', { provider: this.eventEmitter, contract: 'emitEvent/readAll/history' });

    // Existing recovered ATO executes formations; the Computer event ledger
    // remains the authority for identity, revisions and prior executions.
    this.formations = await new ComputerFormationService({ computer: this }).recover();
    this.services.register('formation-execution', {
      provider: this.formations, contract: 'create/transform/compose/activate/execute/history',
    });
    this.capabilities.register('formation_execution', { providers: [FORMATION_PROVIDER] });
    if (this.platform === 'browser') await this.enableNativeKlein();

    this.capabilityRegistryService = await new CapabilityRegistryService({ bus: this.bus }).load();
    this.services.register('capability-registry', { provider: this.capabilityRegistryService, contract: 'queryCapability' });

    if (this.platform === 'node') {
      const [
        { AddressService }, { StateResolver }, { AutomataEngineGateway },
        { WorldEngineGateway }, { PentaEphemerisService }, { ExperimentLoop },
      ] = await Promise.all([
        import('./services/address-service.mjs'),
        import('./services/state-resolver.mjs'),
        import('./services/automata-engine.mjs'),
        import('./services/world-engine.mjs'),
        import('./services/penta-ephemeris.mjs'),
        import('./services/experiment-loop.mjs'),
      ]);
      this.addressService = new AddressService({ bus: this.bus });
      this.stateResolver = new StateResolver({
        bus: this.bus, addressService: this.addressService,
        eventLog: () => this.eventEmitter.readAll(),
      });
      this.addressService.capabilityRegistry = this.capabilityRegistryService;
      this.services.register('address-service', { provider: this.addressService, contract: 'resolveAddress' });
      this.services.register('state-resolver', { provider: this.stateResolver, contract: 'resolveState' });
      this.automataGateway = new AutomataEngineGateway({ bus: this.bus, state: this.state });
      this.services.register('automata-engine', { provider: this.automataGateway, contract: 'execute' });
      this.worldGateway = new WorldEngineGateway({ bus: this.bus });
      this.services.register('world-engine', { provider: this.worldGateway, contract: 'worldEvent' });
      this.pentaEphemeris = new PentaEphemerisService({ bus: this.bus });
      this.services.register('penta-ephemeris', { provider: this.pentaEphemeris, contract: 'groupPenta' });
      this.experiments = new ExperimentLoop({ bus: this.bus });
      this.services.register('experiment-loop', { provider: this.experiments, contract: 'runExperiment' });
      for (const id of ['resolve_address', 'resolve_state', 'emit_event', 'query_capability', 'automata_activation'])
        if (!this.capabilityRegistry.has(id))
          this.capabilities.register(id, { providers: ['back-up-', 'computer'] });
    }

    await this.state.set('computer.boot', { status: 'ready', at: Date.now(), version: '0.2.0-project-workspace' }, { source: 'boot' });
    this.bus.emit('computer:ready', this.snapshot());
    return this;
  }

  // ── Stage-4a contract surface (delegates to mounted services, not direct donor imports) ──
  queryCapability(name) { return this.capabilityRegistryService.queryCapability(name); }
  route(capability, ctx = {}) { return this.capabilityRegistryService.route(capability, ctx); }
  resolveAddress(entityOrEvent, options) { return this.addressService.resolveAddress(entityOrEvent, options); }
  compareAddresses(a, b) { return this.addressService.compareAddresses(a, b); }
  resolveRelationship(a, b, context) { return this.addressService.resolveRelationship(a, b, context); }
  resolveState(entity, event, context) { return this.stateResolver.resolveState(entity, event, context); }
  emitEvent(event) { return this.eventEmitter.emitEvent(event); }
  executeOnSwarm(capability, input, ctx) { return this.automataGateway.execute(capability, input, ctx); }
  async enableNativeKlein() {
    if (this.nativeKlein) return this.nativeKlein;
    const { NativeKleinMemory } = await import('./services/native-klein-memory.mjs');
    this.nativeKlein = await new NativeKleinMemory({ computer: this }).boot();
    this.services.register('klein-native-memory', {
      provider: this.nativeKlein, contract: 'autoling.recognize/diseminer.ingest/replay',
    });
    return this.nativeKlein;
  }
  registerFormationOperation(id, implementation) { this.formations.registerOperation(id, implementation); return this; }
  createFormation(spec) { return this.formations.create(spec); }
  transformFormation(id, graph, options) { return this.formations.transform(id, graph, options); }
  composeFormations(spec) { return this.formations.compose(spec); }
  activateFormation(id, revisionId) { return this.formations.activate(id, revisionId); }
  executeFormation(id, input, options) { return this.formations.execute(id, input, options); }
  formationHistory(id) { return this.formations.history(id); }
  worldEvent(event) { return this.worldGateway.process(event); }
  observeWorld() { return this.worldGateway.snapshot(); }
  groupPenta(members) { return this.pentaEphemeris.groupPenta(members); }

  /**
   * Contract: mount(application, contract). Mounts a real artifact app through
   * the existing intake -> mutation -> shell-manager pipeline and returns the
   * mount contract: a RESTRICTED shared-services surface (the app consumes
   * Computer services; it does NOT become a Computer). State is namespaced.
   */
  async mountApplication(appId, { artifactId = null, shell = 'workspace' } = {}) {
    if (!this.artifacts.get(artifactId)) throw new Error(`mountApplication: unknown artifact ${artifactId} (ingest first)`);
    const proposal = this.intake.proposeMount(artifactId, { appId, shell });
    const applied = await this.mutations.apply(proposal.id);
    const mount = applied.results[0].result;
    const ns = `apps.${appId}`;
    const runtime = this;
    const contract = {
      mountId: mount.mountId,
      appId,
      artifactId,
      services: Object.freeze({
        emitEvent: (event) => runtime.emitEvent({ ...event, actor_id: event.actor_id ?? `app:${appId}`, actor_type: event.actor_type ?? 'application' }),
        resolveAddress: (input, options) => runtime.resolveAddress(input, options),
        resolveState: (entity, event, context) => runtime.resolveState(entity, event, context),
        getState: (path, fallback) => runtime.state.get(`${ns}.${path}`, fallback),
        setState: (path, value) => runtime.state.set(`${ns}.${path}`, value, { source: `app:${appId}` }),
        readArtifact: () => runtime.vfs.read(runtime.artifacts.get(artifactId).path),
      }),
    };
    this.bus.emit('app:contract-issued', { appId, mountId: mount.mountId, services: Object.keys(contract.services) });
    return contract;
  }

  snapshot() {
    return {
      version: '0.2.0-project-workspace',
      state: this.state.snapshot(),
      systems: this.systems.list(),
      micros: this.micros.list(),
      shells: this.shells.list(),
      apps: this.apps.list(),
      tools: this.tools.list(),
      agents: this.agents.list(),
      worlds: this.worlds.list(),
      artifacts: this.artifacts.list(),
      formations: this.formations?.list() ?? [],
      projects: this.projects.list(),
      backends: this.backendsRegistry.list().map(({ id, status }) => ({ id, status })),
      capabilities: this.capabilityRegistry.list(),
      mounted: this.shellManager.listMounted()
    };
  }
}
