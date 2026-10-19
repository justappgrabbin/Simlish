import { FactoryError } from './integrated-tool-factory.mjs';

/**
 * Converts factory tools into genuine instances of the Automaton class owned
 * by the host ATO Core. Constructor injection keeps this package portable
 * while preserving ATO's instanceof boundary and validation rules.
 */
export class ATONativeBridge {
  constructor({ Automaton, mesh, factory } = {}) {
    if (typeof Automaton !== 'function') throw new FactoryError('ATO_AUTOMATON_REQUIRED', 'ATO Core Automaton constructor is required');
    if (!mesh || typeof mesh.add !== 'function' || !(mesh.automatons instanceof Map)) throw new FactoryError('ATO_MESH_REQUIRED', 'A live ATO AutomataMesh is required');
    if (!factory || typeof factory.generate !== 'function') throw new FactoryError('FACTORY_REQUIRED', 'IntegratedToolFactory is required');
    this.Automaton = Automaton;
    this.mesh = mesh;
    this.factory = factory;
    this.nativeTools = new Map();
    this.events = [];
  }

  materialize(factoryTool) {
    if (!factoryTool?.manifest || typeof factoryTool.execute !== 'function') throw new FactoryError('INVALID_FACTORY_TOOL', 'A generated factory tool is required');
    if (this.nativeTools.has(factoryTool.id)) return this.nativeTools.get(factoryTool.id);
    const source = factoryTool.manifest();
    const structuralBand = source.structure.level <= 1
      ? { structure: 'bigram', activeLevels: [1] }
      : source.structure.level <= 4
        ? { structure: 'trigram', activeLevels: [1, 2] }
        : { structure: 'hexagram', activeLevels: [1, 2, 3, 4, 5] };
    // ATO Core names the Evolution binding "mind". Preserve Evolution in
    // metadata/address while crossing that native vocabulary boundary.
    const nativeLevel = source.functionalLevel === 'evolution' ? 'mind' : source.functionalLevel;
    const native = new this.Automaton({
      id: factoryTool.id,
      address: source.address,
      structure: structuralBand.structure,
      activeLevels: structuralBand.activeLevels,
      functionalLevel: nativeLevel,
      ports: source.ports.map(port => ({ ...port })),
      implementation: (input, context) => factoryTool.execute(input, { ...context, nativeATO: true }),
      state: factoryTool,
      metadata: {
        ...source.metadata,
        family: 'ato-native-generated-tool',
        factoryManifestVersion: source.manifestVersion,
        factoryToolId: factoryTool.id,
        factoryDimension: source.functionalLevel,
        factoryStructuralLevel: source.structure.level,
        generatedSource: factoryTool.source,
      },
    });
    this.nativeTools.set(native.id, native);
    this.events.push(Object.freeze({ type: 'materialize', toolId: native.id, addressKey: native.addressKey }));
    return native;
  }

  mount(factoryTool) {
    const native = this.materialize(factoryTool);
    if (this.mesh.automatons.has(native.id)) {
      if (this.mesh.automatons.get(native.id) !== native) throw new FactoryError('ATO_ID_COLLISION', native.id);
      return Object.freeze({ status: 'existing', automaton: native });
    }
    this.mesh.add(native);
    this.events.push(Object.freeze({ type: 'mount', toolId: native.id, addressKey: native.addressKey }));
    return Object.freeze({ status: 'mounted', automaton: native });
  }

  generateAndMount(request) {
    const generated = this.factory.generate(request);
    if (!generated.tool) return generated;
    const mounted = this.mount(generated.tool);
    return Object.freeze({ status: mounted.status, generationStatus: generated.status, tool: generated.tool, automaton: mounted.automaton });
  }

  connect(fromId, toId, options = {}) {
    const result = this.mesh.connect(fromId, toId, options);
    this.events.push(Object.freeze({ type: 'connect', fromId, toId, status: result.status, edgeId: result.edge?.id || null }));
    return result;
  }

  async run(toolId, input, options = {}) {
    if (!this.mesh.automatons.has(toolId)) throw new FactoryError('UNMOUNTED_TOOL', toolId);
    const result = await this.mesh.run(toolId, input, options);
    this.events.push(Object.freeze({ type: 'run', toolId, runId: result.runId, visited: result.visited }));
    return result;
  }

  dissolve(toolId, reason = 'dissolved') {
    const native = this.nativeTools.get(toolId);
    if (!native) return false;
    for (const [edgeId, edge] of this.mesh.edges) if (edge.fromId === toolId || edge.toId === toolId) this.mesh.edges.delete(edgeId);
    this.mesh.automatons.delete(toolId);
    this.factory.dissolve(toolId, reason);
    this.events.push(Object.freeze({ type: 'dissolve', toolId, reason }));
    return true;
  }

  restore(toolId) {
    const factoryTool = this.factory.restore(toolId);
    const native = this.nativeTools.get(toolId) || this.materialize(factoryTool);
    if (!this.mesh.automatons.has(toolId)) this.mesh.add(native);
    this.events.push(Object.freeze({ type: 'restore', toolId }));
    return native;
  }

  snapshot() {
    return Object.freeze({
      manifestVersion: 'ato.factory-bridge.v1',
      nativeTools: Object.freeze([...this.nativeTools.values()].map(tool => tool.manifest())),
      events: Object.freeze(this.events.map(event => Object.freeze({ ...event }))),
      mesh: this.mesh.snapshot(),
    });
  }
}
