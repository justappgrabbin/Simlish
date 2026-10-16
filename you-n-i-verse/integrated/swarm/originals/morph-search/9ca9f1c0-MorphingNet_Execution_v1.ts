/**
 * MORPHING NET — EXECUTION ENGINE v1.0
 * 
 * Pipeline: INGEST → DISSOLVE → ANALYZE → UNDERSTAND → SYNTHESIZE → EXECUTE → RENDER
 * 
 * The substrate doesn't just map input. It REBUILDS it.
 * 
 * ANALYSIS: What kind of thing is this? (code, data, UI, config, media)
 * UNDERSTANDING: What's its purpose? (app, API, dashboard, game, tool)
 * SYNTHESIS: Map purpose to gate-component architecture
 * EXECUTION: Generate actual working HTML/JS/CSS from substrate topology
 * 
 * 64 Gates = 64 Component Primitives
 * 5 Layers = 5 Rendering Stages (structure → style → logic → data → integration)
 * 81 Emergent = Feature Injections (unexpected capabilities that crystallize)
 */

// ============================================================================
// ANALYZER — Detects what kind of thing the input is
// ============================================================================

export type ContentType = 
  | "web_app" | "static_site" | "api" | "data_schema" | "config" 
  | "game" | "tool" | "document" | "media" | "unknown";

export type Purpose = 
  | "display" | "collect" | "transform" | "communicate" | "organize"
  | "calculate" | "visualize" | "control" | "entertain" | "educate";

export interface AnalysisResult {
  contentType: ContentType;
  purpose: Purpose;
  confidence: number;  // 0.0 - 1.0
  detectedFeatures: string[];
  entryPoints: string[];
  dependencies: string[];
  structure: any;
}

export class SubstrateAnalyzer {
  
  analyze(dissolvedState: DissolvedState, rawContent: any): AnalysisResult {
    const source = dissolvedState.source;
    const activations = dissolvedState.gateActivations;
    
    // Pattern detection from gate activation signatures
    const patterns = this.detectPatterns(activations);
    
    // Content type inference
    const contentType = this.inferContentType(source, patterns, rawContent);
    
    // Purpose inference from trigram flow
    const purpose = this.inferPurpose(dissolvedState.trigramSignature, patterns);
    
    // Feature detection
    const features = this.detectFeatures(patterns, rawContent);
    
    // Entry point discovery
    const entryPoints = this.findEntryPoints(source, rawContent);
    
    // Dependency mapping
    const dependencies = this.mapDependencies(source, rawContent);
    
    return {
      contentType,
      purpose,
      confidence: this.calculateConfidence(patterns, contentType),
      detectedFeatures: features,
      entryPoints,
      dependencies,
      structure: this.extractStructure(rawContent, contentType)
    };
  }
  
  private detectPatterns(activations: GateActivation[]): Record<string, number> {
    const patterns: Record<string, number> = {};
    
    // Gate clustering patterns
    const gateCounts = new Map<number, number>();
    for (const act of activations) {
      gateCounts.set(act.gate, (gateCounts.get(act.gate) || 0) + 1);
    }
    
    // High-activation gates = dominant features
    const dominant = Array.from(gateCounts.entries()).sort((a, b) => b[1] - a[1]).slice(0, 8);
    
    // Layer distribution
    const layerDist = [0, 0, 0, 0, 0];
    for (const act of activations) layerDist[act.layer - 1]++;
    
    // Pattern signatures based on dominant gates
    for (const [gate, count] of dominant) {
      const center = getCenter(gate);
      const family = getFamily(gate);
      patterns[`${center}_${family}`] = (patterns[`${center}_${family}`] || 0) + count;
    }
    
    // Layer-heavy patterns
    const maxLayer = layerDist.indexOf(Math.max(...layerDist)) + 1;
    patterns[`layer_${maxLayer}_dominant`] = layerDist[maxLayer - 1];
    
    return patterns;
  }
  
  private inferContentType(source: IngestedContent, patterns: Record<string, number>, raw: any): ContentType {
    const name = source.name.toLowerCase();
    const type = source.type;
    
    // File extension heuristics
    if (name.endsWith('.zip') || name.endsWith('.tar.gz')) {
      // Check if web app
      if (patterns['Throat_Awakening'] > 5 || patterns['Ajna_Understanding'] > 5) return "web_app";
      if (patterns['Sacral_Knowing'] > 5) return "api";
      return "static_site";
    }
    
    if (name.endsWith('.json')) {
      if (raw && (raw.routes || raw.endpoints || raw.paths)) return "api";
      if (raw && (raw.schema || raw.properties || raw.fields)) return "data_schema";
      if (raw && (raw.components || raw.widgets || raw.layout)) return "web_app";
      return "config";
    }
    
    if (name.endsWith('.html') || name.endsWith('.htm')) return "web_app";
    if (name.match(/\.(js|ts|jsx|tsx)$/)) return "web_app";
    if (name.match(/\.(css|scss|less)$/)) return "static_site";
    if (name.match(/\.(py|java|go|rs|c|cpp)$/)) return "tool";
    if (name.match(/\.(md|txt|rst)$/)) return "document";
    if (name.match(/\.(png|jpg|jpeg|gif|svg|webp|mp4|webm)$/)) return "media";
    
    // Content-based inference for text
    if (typeof raw === 'string') {
      if (raw.includes('<!DOCTYPE html>') || raw.includes('<html')) return "web_app";
      if (raw.includes('function') || raw.includes('class') || raw.includes('const') || raw.includes('let')) return "web_app";
      if (raw.includes('{') && raw.includes('}') && raw.includes('"')) return "config";
      if (raw.includes('## ') || raw.includes('# ')) return "document";
    }
    
    // Gate pattern inference
    if (patterns['Heart_Money'] > 10) return "game";
    if (patterns['Head_Understanding'] > 10) return "tool";
    if (patterns['Solar Plexus_Knowing'] > 10) return "api";
    
    return "unknown";
  }
  
  private inferPurpose(trigrams: Trigram[], patterns: Record<string, number>): Purpose {
    // Map trigram flow to purpose
    const trigramNames = trigrams.map(t => t.name);
    
    if (trigramNames.includes('Li') && trigramNames.includes('Dui')) return "display";
    if (trigramNames.includes('Kan') && trigramNames.includes('Kun')) return "collect";
    if (trigramNames.includes('Zhen') && trigramNames.includes('Qian')) return "transform";
    if (trigramNames.includes('Xun') && trigramNames.includes('Li')) return "communicate";
    if (trigramNames.includes('Gen') && trigramNames.includes('Kun')) return "organize";
    if (trigramNames.includes('Qian') && trigramNames.includes('Zhen')) return "calculate";
    if (trigramNames.includes('Li') && trigramNames.includes('Kan')) return "visualize";
    if (trigramNames.includes('Dui') && trigramNames.includes('Qian')) return "control";
    if (trigramNames.includes('Zhen') && trigramNames.includes('Xun')) return "entertain";
    if (trigramNames.includes('Gen') && trigramNames.includes('Li')) return "educate";
    
    // Fallback to pattern-based
    if (patterns['Throat_Centering'] > 5) return "display";
    if (patterns['G_Centering'] > 5) return "organize";
    if (patterns['Sacral_Knowing'] > 5) return "transform";
    
    return "display";
  }
  
  private detectFeatures(patterns: Record<string, number>, raw: any): string[] {
    const features: string[] = [];
    
    // Gate-based feature detection
    if (patterns['Head_Understanding'] > 3) features.push("navigation");
    if (patterns['Ajna_Understanding'] > 3) features.push("search");
    if (patterns['Throat_Centering'] > 3) features.push("forms");
    if (patterns['G_Centering'] > 3) features.push("layout");
    if (patterns['Heart_Centering'] > 3) features.push("auth");
    if (patterns['Spleen_Sensing'] > 3) features.push("realtime");
    if (patterns['Solar Plexus_Knowing'] > 3) features.push("data_binding");
    if (patterns['Sacral_Knowing'] > 3) features.push("animations");
    if (patterns['Root_Centering'] > 3) features.push("persistence");
    
    // Content-based features
    if (typeof raw === 'string') {
      if (raw.includes('fetch(') || raw.includes('axios') || raw.includes('XMLHttpRequest')) features.push("api_client");
      if (raw.includes('localStorage') || raw.includes('indexedDB')) features.push("storage");
      if (raw.includes('canvas') || raw.includes('webgl')) features.push("graphics");
      if (raw.includes('websocket') || raw.includes('ws://')) features.push("websocket");
      if (raw.includes('drag') || raw.includes('drop')) features.push("dnd");
      if (raw.includes('gesture') || raw.includes('touch')) features.push("mobile");
    }
    
    return [...new Set(features)];
  }
  
  private findEntryPoints(source: IngestedContent, raw: any): string[] {
    const entries: string[] = [];
    
    if (source.name.endsWith('.zip')) {
      entries.push('index.html');
      entries.push('src/index.js');
      entries.push('app.js');
    } else if (source.name.endsWith('.json')) {
      entries.push(source.name);
    } else {
      entries.push(source.name);
    }
    
    return entries;
  }
  
  private mapDependencies(source: IngestedContent, raw: any): string[] {
    const deps: string[] = [];
    
    if (typeof raw === 'string') {
      const importMatches = raw.match(/import\s+.*?\s+from\s+['"]([^'"]+)['"]/g) || [];
      const requireMatches = raw.match(/require\(['"]([^'"]+)['"]\)/g) || [];
      
      for (const match of importMatches) {
        const dep = match.match(/from\s+['"]([^'"]+)['"]/)?.[1];
        if (dep && !dep.startsWith('.')) deps.push(dep);
      }
      
      for (const match of requireMatches) {
        const dep = match.match(/require\(['"]([^'"]+)['"]\)/)?.[1];
        if (dep && !dep.startsWith('.')) deps.push(dep);
      }
    }
    
    return [...new Set(deps)];
  }
  
  private extractStructure(raw: any, contentType: ContentType): any {
    if (contentType === 'web_app' && typeof raw === 'string') {
      // Extract component structure from HTML
      const bodyMatch = raw.match(/<body[^>]*>([\s\S]*?)<\/body>/i);
      const scriptMatches = raw.match(/<script[^>]*>([\s\S]*?)<\/script>/gi) || [];
      const styleMatches = raw.match(/<style[^>]*>([\s\S]*?)<\/style>/gi) || [];
      
      return {
        body: bodyMatch ? bodyMatch[1] : '',
        scripts: scriptMatches.length,
        styles: styleMatches.length,
        hasHead: raw.includes('<head>'),
        hasMeta: raw.includes('<meta')
      };
    }
    
    if (contentType === 'data_schema' && typeof raw === 'object') {
      return { keys: Object.keys(raw), depth: this.objectDepth(raw) };
    }
    
    return { raw: typeof raw === 'string' ? raw.slice(0, 500) : raw };
  }
  
  private objectDepth(obj: any, depth = 0): number {
    if (typeof obj !== 'object' || obj === null) return depth;
    return Math.max(depth, ...Object.values(obj).map(v => this.objectDepth(v, depth + 1)));
  }
  
  private calculateConfidence(patterns: Record<string, number>, inferredType: ContentType): number {
    const totalWeight = Object.values(patterns).reduce((a, b) => a + b, 0);
    const typePattern = Object.entries(patterns).find(([k]) => k.includes(inferredType));
    return typePattern ? Math.min(1.0, typePattern[1] / totalWeight * 3) : 0.5;
  }
}

// ============================================================================
// SYNTHESIZER — Maps understanding to gate-component architecture
// ============================================================================

export interface ComponentSpec {
  id: string;
  type: string;
  gate: number;
  center: string;
  purpose: string;
  props: Record<string, any>;
  children: string[];
  logic: string[];
  style: Record<string, string>;
}

export interface AppArchitecture {
  contentType: ContentType;
  purpose: Purpose;
  components: ComponentSpec[];
  layout: string;  // "grid" | "flex" | "absolute" | "flow"
  dataFlow: string;  // "props" | "context" | "events" | "stream"
  entryComponent: string;
  features: string[];
  emergentInjections: string[];  // 81 emergent states as feature surprises
}

export class ArchitectureSynthesizer {
  private engine: MorphingNetEngine;
  
  constructor(engine: MorphingNetEngine) {
    this.engine = engine;
  }
  
  synthesize(analysis: AnalysisResult, dissolved: DissolvedState): AppArchitecture {
    // Generate component specs from dominant gates
    const components = this.gatesToComponents(dissolved.gateActivations, analysis);
    
    // Determine layout from trigram flow
    const layout = this.trigramsToLayout(dissolved.trigramSignature);
    
    // Determine data flow from circuit distribution
    const dataFlow = this.circuitsToDataFlow(dissolved.gateActivations);
    
    // Entry component from strongest activation
    const entryComponent = this.findEntryComponent(components);
    
    // Emergent feature injections
    const emergentInjections = this.emergentToFeatures(dissolved.emergentSeeds, analysis);
    
    return {
      contentType: analysis.contentType,
      purpose: analysis.purpose,
      components,
      layout,
      dataFlow,
      entryComponent,
      features: analysis.detectedFeatures,
      emergentInjections
    };
  }
  
  private gatesToComponents(activations: GateActivation[], analysis: AnalysisResult): ComponentSpec[] {
    const components: ComponentSpec[] = [];
    const seenGates = new Set<number>();
    
    // Sort by strength, take top 16
    const sorted = activations.sort((a, b) => b.strength - a.strength).slice(0, 16);
    
    for (const act of sorted) {
      if (seenGates.has(act.gate)) continue;
      seenGates.add(act.gate);
      
      const center = getCenter(act.gate);
      const family = getFamily(act.gate);
      const componentType = this.gateToComponentType(act.gate, center, family, analysis);
      
      components.push({
        id: `comp_${act.gate}_${act.layer}`,
        type: componentType,
        gate: act.gate,
        center,
        purpose: this.centerToPurpose(center),
        props: this.generateProps(act.gate, componentType, analysis),
        children: [],
        logic: this.generateLogic(act.gate, componentType),
        style: this.gateToStyle(act.gate, act.layer)
      });
    }
    
    // Wire children based on family clusters
    for (const comp of components) {
      const family = getFamily(comp.gate);
      const siblings = components.filter(c => getFamily(c.gate) === family && c.id !== comp.id);
      comp.children = siblings.slice(0, 3).map(c => c.id);
    }
    
    return components;
  }
  
  private gateToComponentType(gate: number, center: string, family: string, analysis: AnalysisResult): string {
    // Map gates to component primitives
    const typeMap: Record<string, Record<string, string>> = {
      "Head": { default: "navigation", Understanding: "search", Sensing: "filter" },
      "Ajna": { default: "input", Understanding: "textarea", Sensing: "select" },
      "Throat": { default: "button", Centering: "link", Knowing: "command" },
      "G": { default: "container", Centering: "layout", Sensing: "grid" },
      "Heart": { default: "badge", Centering: "alert", Sensing: "status" },
      "Spleen": { default: "list", Sensing: "timeline", Knowing: "chart" },
      "Solar Plexus": { default: "card", Knowing: "form", Sensing: "dialog" },
      "Sacral": { default: "media", Knowing: "animation", Sensing: "player" },
      "Root": { default: "footer", Centering: "header", Sensing: "sidebar" }
    };
    
    const circuit = getCircuit(gate);
    return typeMap[center]?.[circuit] || typeMap[center]?.default || "div";
  }
  
  private centerToPurpose(center: string): string {
    const map: Record<string, string> = {
      "Head": "direct", "Ajna": "process", "Throat": "express",
      "G": "contain", "Heart": "validate", "Spleen": "sense",
      "Solar Plexus": "feel", "Sacral": "create", "Root": "ground"
    };
    return map[center] || "contain";
  }
  
  private generateProps(gate: number, type: string, analysis: AnalysisResult): Record<string, any> {
    const props: Record<string, any> = {};
    
    // Base props from gate number
    props.id = `gate_${gate}`;
    props['data-gate'] = gate;
    props['data-center'] = getCenter(gate);
    props['data-family'] = getFamily(gate);
    
    // Type-specific props
    if (type === "button") {
      props.variant = gate % 2 === 0 ? "primary" : "secondary";
      props.size = gate % 3 === 0 ? "large" : "medium";
    }
    if (type === "input" || type === "textarea") {
      props.placeholder = HEXAGRAM_NAMES[gate - 1];
      props.required = gate % 7 === 0;
    }
    if (type === "card") {
      props.elevated = gate % 4 === 0;
      props.collapsible = gate % 5 === 0;
    }
    if (type === "navigation") {
      props.orientation = gate % 2 === 0 ? "horizontal" : "vertical";
      props.sticky = gate % 3 === 0;
    }
    
    // Inject detected features
    if (analysis.detectedFeatures.includes("auth")) {
      props.protected = true;
    }
    if (analysis.detectedFeatures.includes("realtime")) {
      props.live = true;
    }
    
    return props;
  }
  
  private generateLogic(gate: number, type: string): string[] {
    const logic: string[] = [];
    
    // Logic operators from trigram
    const upper = BinaryMorph.upperTrigram(gate);
    const lower = BinaryMorph.lowerTrigram(gate);
    const upperTrigram = BinaryMorph.getTrigram(upper);
    const lowerTrigram = BinaryMorph.getTrigram(lower);
    
    logic.push(`// ${upperTrigram.operator} logic from upper trigram`);
    logic.push(`// ${lowerTrigram.operator} logic from lower trigram`);
    
    // Component-specific logic
    if (type === "button") {
      logic.push(`onClick: () => { activateGate(${gate}); }`);
    }
    if (type === "input") {
      logic.push(`onChange: (e) => { dissolveInput(e.target.value); }`);
    }
    if (type === "card") {
      logic.push(`onExpand: () => { morphLayer(${gate % 5 + 1}); }`);
    }
    
    return logic;
  }
  
  private gateToStyle(gate: number, layer: number): Record<string, string> {
    const colors = ["#ff3333", "#ff8833", "#ffdd33", "#33ff88", "#3388ff"];
    const baseColor = colors[layer - 1] || colors[0];
    
    return {
      "--gate-color": baseColor,
      "--gate-binary": BinaryMorph.binaryString(gate),
      "border-color": baseColor + "40",
      "background": `linear-gradient(135deg, ${baseColor}10 0%, transparent 100%)`
    };
  }
  
  private trigramsToLayout(trigrams: Trigram[]): string {
    const names = trigrams.map(t => t.name);
    
    if (names.includes("Qian") && names.includes("Kun")) return "grid";
    if (names.includes("Zhen") && names.includes("Xun")) return "flex";
    if (names.includes("Kan") && names.includes("Li")) return "flow";
    if (names.includes("Gen") && names.includes("Dui")) return "absolute";
    
    return "flex";
  }
  
  private circuitsToDataFlow(activations: GateActivation[]): string {
    const circuitCounts: Record<string, number> = {};
    for (const act of activations) {
      const circuit = getCircuit(act.gate);
      circuitCounts[circuit] = (circuitCounts[circuit] || 0) + 1;
    }
    
    const dominant = Object.entries(circuitCounts).sort((a, b) => b[1] - a[1])[0]?.[0];
    
    const flowMap: Record<string, string> = {
      "Understanding": "props",
      "Sensing": "events",
      "Knowing": "context",
      "Centering": "stream",
      "Integration": "stream"
    };
    
    return flowMap[dominant] || "props";
  }
  
  private findEntryComponent(components: ComponentSpec[]): string {
    // Entry = strongest Head or G center component, or first
    const headComp = components.find(c => c.center === "Head");
    const gComp = components.find(c => c.center === "G");
    return headComp?.id || gComp?.id || components[0]?.id || "app";
  }
  
  private emergentToFeatures(seeds: number[], analysis: AnalysisResult): string[] {
    const injections: string[] = [];
    
    for (const seed of seeds.slice(0, 5)) {
      const tetragram = (seed % 81) + 1;
      
      // Map tetragram to surprise feature
      if (tetragram % 9 === 0) injections.push("auto_morph");
      if (tetragram % 7 === 0) injections.push("voice_input");
      if (tetragram % 5 === 0) injections.push("gesture_control");
      if (tetragram % 4 === 0) injections.push("dark_mode");
      if (tetragram % 3 === 0) injections.push("p2p_sync");
      if (tetragram % 2 === 0) injections.push("offline_mode");
      
      // Context-aware injections
      if (analysis.contentType === "web_app" && tetragram > 40) injections.push("pwa_shell");
      if (analysis.contentType === "game" && tetragram > 60) injections.push("procedural_gen");
      if (analysis.purpose === "visualize" && tetragram < 20) injections.push("3d_view");
    }
    
    return [...new Set(injections)];
  }
}

// ============================================================================
// EXECUTOR — Generates actual working HTML/JS/CSS
// ============================================================================

export class NativeExecutor {
  private engine: MorphingNetEngine;
  
  constructor(engine: MorphingNetEngine) {
    this.engine = engine;
  }
  
  execute(architecture: AppArchitecture, dissolved: DissolvedState): string {
    switch (architecture.contentType) {
      case "web_app":
      case "static_site":
        return this.generateWebApp(architecture, dissolved);
      case "api":
        return this.generateAPIClient(architecture, dissolved);
      case "data_schema":
        return this.generateDataViewer(architecture, dissolved);
      case "game":
        return this.generateGameShell(architecture, dissolved);
      case "tool":
        return this.generateTool(architecture, dissolved);
      case "document":
        return this.generateDocument(architecture, dissolved);
      default:
        return this.generateWebApp(architecture, dissolved);
    }
  }
  
  private generateWebApp(arch: AppArchitecture, dissolved: DissolvedState): string {
    const { components, layout, dataFlow, entryComponent, features, emergentInjections } = arch;
    
    // Generate component functions
    const componentFunctions = components.map(comp => this.renderComponentFunction(comp)).join("\n\n");
    
    // Generate app shell
    const appShell = this.renderAppShell(arch, dissolved);
    
    // Generate emergent feature injections
    const emergentCode = this.renderEmergentFeatures(emergentInjections);
    
    // Generate styles
    const styles = this.renderStyles(components, dissolved);
    
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Morphed: ${dissolved.source.name}</title>
  <style>
    ${styles}
  </style>
</head>
<body>
  <div id="app"></div>
  
  <script>
    // ============================================================================
    // MORPHING NET — NATIVE EXECUTION
    // Generated from ${dissolved.source.name} via 64-gate substrate
    // Content Type: ${arch.contentType} | Purpose: ${arch.purpose}
    // ============================================================================
    
    const GATE_REGISTRY = {};
    const MORPH_STATE = {
      activations: ${JSON.stringify(dissolved.gateActivations.slice(0, 8))},
      emergent: ${JSON.stringify(dissolved.emergentSeeds.slice(0, 5))},
      layer: 1
    };
    
    function activateGate(gate) {
      GATE_REGISTRY[gate] = (GATE_REGISTRY[gate] || 0) + 1;
      MORPH_STATE.layer = (MORPH_STATE.layer % 5) + 1;
      render();
    }
    
    function dissolveInput(value) {
      const bytes = new TextEncoder().encode(value);
      const gates = [];
      for (let i = 0; i < bytes.length; i += 6) {
        const window = bytes.slice(i, i + 6);
        if (window.length < 6) break;
        const bin = window.map(b => b > 127 ? 1 : 0);
        const val = bin.reduce((a, b, idx) => a + (b << (5 - idx)), 0);
        gates.push((val % 64) + 1);
      }
      gates.forEach(g => activateGate(g));
    }
    
    function morphLayer(layer) {
      MORPH_STATE.layer = layer;
      document.body.setAttribute('data-layer', layer);
      render();
    }
    
    ${componentFunctions}
    
    ${emergentCode}
    
    // App entry
    function App() {
      return ${entryComponent}();
    }
    
    function render() {
      const app = document.getElementById('app');
      app.innerHTML = '';
      app.appendChild(App());
    }
    
    // Initialize
    render();
    
    // Auto-morph if emergent
    ${emergentInjections.includes('auto_morph') ? `
    setInterval(() => {
      const randomGate = Math.floor(Math.random() * 64) + 1;
      activateGate(randomGate);
    }, 5000);
    ` : ''}
  </script>
</body>
</html>`;
  }
  
  private renderComponentFunction(comp: ComponentSpec): string {
    const { id, type, gate, props, children, logic, style } = comp;
    
    const propStr = Object.entries(props)
      .map(([k, v]) => {
        if (typeof v === 'boolean') return v ? k : '';
        if (typeof v === 'number') return `${k}="${v}"`;
        return `${k}="${v}"`;
      })
      .filter(Boolean)
      .join(' ');
    
    const styleStr = Object.entries(style)
      .map(([k, v]) => `${k}: ${v}`)
      .join('; ');
    
    const childCalls = children.map(c => `${c}()`).join(', ');
    
    const eventHandlers = logic
      .filter(l => l.includes('onClick') || l.includes('onChange') || l.includes('onExpand'))
      .map(l => {
        const match = l.match(/(\w+):\s*\(\)\s*=>\s*\{\s*(.*?)\s*\}/);
        if (match) return `el.${match[1].toLowerCase()} = () => { ${match[2]} };`;
        return '';
      })
      .filter(Boolean)
      .join('\n      ');
    
    return `function ${id}() {
      const el = document.createElement('${type === 'container' || type === 'layout' || type === 'grid' ? 'div' : type}');
      el.className = 'morph-component ${type}';
      el.setAttribute('data-gate', '${gate}');
      ${propStr ? `el.setAttribute('data-props', '${propStr}');` : ''}
      el.style.cssText = '${styleStr}';
      
      ${eventHandlers}
      
      ${childCalls ? `const children = [${childCalls}];
      children.forEach(c => el.appendChild(c));` : ''}
      
      return el;
    }`;
  }
  
  private renderAppShell(arch: AppArchitecture, dissolved: DissolvedState): string {
    return `// App shell generated from ${arch.contentType}`;
  }
  
  private renderEmergentFeatures(injections: string[]): string {
    const features: Record<string, string> = {
      "auto_morph": `
    // Auto-morph: substrate self-modifies every 5s
    setInterval(() => {
      const gates = Object.keys(GATE_REGISTRY).map(Number);
      if (gates.length > 0) {
        const g = gates[Math.floor(Math.random() * gates.length)];
        activateGate(g);
      }
    }, 5000);`,
      "voice_input": `
    // Voice input: emergent from tetragram resonance
    if ('webkitSpeechRecognition' in window) {
      const recognition = new webkitSpeechRecognition();
      recognition.continuous = true;
      recognition.onresult = (e) => {
        const transcript = e.results[e.results.length - 1][0].transcript;
        dissolveInput(transcript);
      };
      // Auto-start if emergent
      setTimeout(() => recognition.start(), 1000);
    }`,
      "gesture_control": `
    // Gesture control: emergent touch morphing
    let touchStart = null;
    document.addEventListener('touchstart', e => { touchStart = e.touches[0]; });
    document.addEventListener('touchend', e => {
      if (!touchStart) return;
      const dx = e.changedTouches[0].clientX - touchStart.clientX;
      const dy = e.changedTouches[0].clientY - touchStart.clientY;
      if (Math.abs(dx) > 100) morphLayer(dx > 0 ? 2 : 4);
      if (Math.abs(dy) > 100) morphLayer(dy > 0 ? 3 : 5);
    });`,
      "dark_mode": `
    // Dark mode: emergent from inverse binary
    function toggleDark() {
      document.body.classList.toggle('dark');
      const isDark = document.body.classList.contains('dark');
      activateGate(isDark ? 2 : 1); // Gate 2 (Receptive) for dark, 1 (Creative) for light
    }
    document.addEventListener('keydown', e => {
      if (e.key === 'd' && e.ctrlKey) toggleDark();
    });`,
      "p2p_sync": `
    // P2P sync: emergent from Integration gate resonance
    const channel = new BroadcastChannel('morph_net');
    channel.onmessage = (e) => {
      if (e.data.type === 'morph') {
        e.data.gates.forEach(g => activateGate(g));
      }
    };
    function broadcastMorph() {
      channel.postMessage({ type: 'morph', gates: Object.keys(GATE_REGISTRY).map(Number) });
    }`,
      "offline_mode": `
    // Offline mode: emergent from Root center grounding
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('data:text/javascript,' + encodeURIComponent('
        self.addEventListener("install", e => self.skipWaiting());
        self.addEventListener("fetch", e => e.respondWith(fetch(e.request).catch(() => new Response("Morphed offline"))));
      '));
    }`,
      "pwa_shell": `
    // PWA shell: emergent from web_app + high tetragram
    const manifest = {
      name: 'Morphed App',
      short_name: 'Morphed',
      start_url: '.',
      display: 'standalone',
      background_color: '#0a0a0f',
      theme_color: '#ffdd33'
    };
    const link = document.createElement('link');
    link.rel = 'manifest';
    link.href = 'data:application/json,' + encodeURIComponent(JSON.stringify(manifest));
    document.head.appendChild(link);`,
      "procedural_gen": `
    // Procedural generation: emergent from game + high tetragram
    function proceduralGen(seed) {
      const gates = [];
      let current = seed;
      for (let i = 0; i < 8; i++) {
        current = ((current * 3) % 64) + 1;
        gates.push(current);
      }
      return gates;
    }`,
      "3d_view": `
    // 3D view: emergent from visualization + low tetragram
    // Uses CSS 3D transforms as lightweight alternative
    document.addEventListener('mousemove', e => {
      const x = (e.clientX / window.innerWidth - 0.5) * 30;
      const y = (e.clientY / window.innerHeight - 0.5) * 30;
      document.getElementById('app').style.transform = 
        \`perspective(1000px) rotateY(\${x}deg) rotateX(\${-y}deg)\`;
    });`
    };
    
    return injections.map(i => features[i] || '').filter(Boolean).join('\n\n');
  }
  
  private renderStyles(components: ComponentSpec[], dissolved: DissolvedState): string {
    const baseStyles = `
      :root {
        --bg: #0a0a0f; --fg: #e0e0e0; --muted: #666;
        --l1: #ff3333; --l2: #ff8833; --l3: #ffdd33; --l4: #33ff88; --l5: #3388ff;
      }
      * { margin: 0; padding: 0; box-sizing: border-box; }
      body {
        background: var(--bg); color: var(--fg);
        font-family: 'Courier New', monospace;
        min-height: 100vh; transition: all 0.3s;
      }
      body.dark { filter: invert(1) hue-rotate(180deg); }
      #app { padding: 20px; }
      .morph-component {
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px; padding: 12px; margin: 8px 0;
        transition: all 0.2s; position: relative;
      }
      .morph-component:hover {
        border-color: var(--l3);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255,221,51,0.1);
      }
      .morph-component::before {
        content: attr(data-gate);
        position: absolute; top: -8px; right: 8px;
        background: var(--bg); padding: 0 6px;
        font-size: 0.7em; color: var(--muted);
      }
      .button {
        background: linear-gradient(135deg, var(--l1)20 0%, transparent 100%);
        cursor: pointer; text-align: center;
      }
      .button:hover { background: var(--l1)40; }
      .input, .textarea, .select {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        color: var(--fg); font-family: inherit;
      }
      .card { background: rgba(255,255,255,0.03); }
      .navigation {
        display: flex; gap: 10px; padding: 10px;
        background: rgba(255,255,255,0.02);
      }
      .layout { display: flex; flex-wrap: wrap; gap: 12px; }
      .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }
      .container { width: 100%; }
      .badge {
        display: inline-block; padding: 2px 8px;
        border-radius: 4px; font-size: 0.75em;
      }
      .alert { border-left: 3px solid var(--l1); }
      .status { border-left: 3px solid var(--l4); }
      .list { list-style: none; }
      .timeline { border-left: 2px solid var(--l5); padding-left: 12px; }
      .chart { min-height: 100px; background: rgba(255,255,255,0.02); }
      .form { display: flex; flex-direction: column; gap: 10px; }
      .dialog {
        position: fixed; top: 50%; left: 50%;
        transform: translate(-50%, -50%);
        background: var(--bg); border: 1px solid var(--l3);
        padding: 20px; border-radius: 12px; z-index: 1000;
      }
      .media { max-width: 100%; }
      .animation { animation: pulse 2s infinite; }
      @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
      }
      .footer, .header, .sidebar {
        padding: 15px; background: rgba(255,255,255,0.02);
      }
      .header { border-bottom: 1px solid rgba(255,255,255,0.1); }
      .footer { border-top: 1px solid rgba(255,255,255,0.1); }
      .sidebar { border-right: 1px solid rgba(255,255,255,0.1); min-height: 100vh; }
    `;
    
    // Add component-specific styles
    const componentStyles = components.map(comp => {
      const styleEntries = Object.entries(comp.style)
        .map(([k, v]) => `${k}: ${v}`)
        .join('; ');
      return `.${comp.id} { ${styleEntries} }`;
    }).join('\n');
    
    return baseStyles + '\n' + componentStyles;
  }
  
  private generateAPIClient(arch: AppArchitecture, dissolved: DissolvedState): string {
    const endpoints = arch.structure?.keys || ['api'];
    
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>API Client: ${dissolved.source.name}</title>
  <style>
    :root { --bg: #0a0a0f; --fg: #e0e0e0; --l3: #ffdd33; --l4: #33ff88; --l5: #3388ff; }
    body { background: var(--bg); color: var(--fg); font-family: monospace; padding: 20px; }
    .endpoint { border: 1px solid #333; border-radius: 8px; padding: 15px; margin: 10px 0; }
    .method { color: var(--l4); font-weight: bold; }
    .url { color: var(--l5); }
    button { background: var(--l3); color: var(--bg); border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; }
    pre { background: #111; padding: 10px; border-radius: 4px; overflow-x: auto; }
  </style>
</head>
<body>
  <h1>↯ API Client</h1>
  <div id="endpoints"></div>
  
  <script>
    const ENDPOINTS = ${JSON.stringify(endpoints.map(e => ({ name: e, method: 'GET', path: `/${e}` })))};
    
    function renderEndpoints() {
      const container = document.getElementById('endpoints');
      ENDPOINTS.forEach(ep => {
        const div = document.createElement('div');
        div.className = 'endpoint';
        div.innerHTML = `
          <div><span class="method">${ep.method}</span> <span class="url">${ep.path}</span></div>
          <button onclick="callEndpoint('${ep.path}')">Call</button>
          <pre id="res_${ep.name}"></pre>
        `;
        container.appendChild(div);
      });
    }
    
    async function callEndpoint(path) {
      try {
        const res = await fetch(path);
        const data = await res.json();
        document.getElementById('res_' + path.slice(1)).textContent = JSON.stringify(data, null, 2);
      } catch(e) {
        document.getElementById('res_' + path.slice(1)).textContent = 'Error: ' + e.message;
      }
    }
    
    renderEndpoints();
  </script>
</body>
</html>`;
  }
  
  private generateDataViewer(arch: AppArchitecture, dissolved: DissolvedState): string {
    return this.generateWebApp(arch, dissolved); // Fallback to web app with data focus
  }
  
  private generateGameShell(arch: AppArchitecture, dissolved: DissolvedState): string {
    const { components } = arch;
    
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Game: ${dissolved.source.name}</title>
  <style>
    :root { --bg: #0a0a0f; --fg: #e0e0e0; }
    body { background: var(--bg); color: var(--fg); font-family: monospace; margin: 0; overflow: hidden; }
    canvas { display: block; }
    #ui { position: fixed; top: 10px; left: 10px; z-index: 100; }
  </style>
</head>
<body>
  <canvas id="game"></canvas>
  <div id="ui">
    <div>Score: <span id="score">0</span></div>
    <div>Gate: <span id="gate">1</span></div>
  </div>
  
  <script>
    const canvas = document.getElementById('game');
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    
    let score = 0;
    let currentGate = 1;
    const entities = [];
    
    // Generate entities from gate activations
    ${dissolved.gateActivations.slice(0, 8).map((act, i) => `
    entities.push({
      x: ${100 + i * 80}, y: ${150 + (i % 3) * 100},
      gate: ${act.gate}, layer: ${act.layer},
      color: ['#ff3333', '#ff8833', '#ffdd33', '#33ff88', '#3388ff'][${act.layer - 1}],
      size: ${20 + act.strength * 30}
    });`).join('')}
    
    function gameLoop() {
      ctx.fillStyle = '#0a0a0f';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      entities.forEach(e => {
        ctx.fillStyle = e.color;
        ctx.beginPath();
        ctx.arc(e.x, e.y, e.size, 0, Math.PI * 2);
        ctx.fill();
        
        ctx.fillStyle = '#fff';
        ctx.font = '12px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(e.gate, e.x, e.y + 4);
        
        // Float animation
        e.y += Math.sin(Date.now() / 1000 + e.gate) * 0.5;
      });
      
      requestAnimationFrame(gameLoop);
    }
    
    canvas.addEventListener('click', e => {
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      
      entities.forEach(ent => {
        const dist = Math.hypot(ent.x - x, ent.y - y);
        if (dist < ent.size) {
          score += ent.gate;
          currentGate = ent.gate;
          document.getElementById('score').textContent = score;
          document.getElementById('gate').textContent = currentGate;
          
          // Morph entity
          ent.size *= 1.2;
          ent.layer = (ent.layer % 5) + 1;
          ent.color = ['#ff3333', '#ff8833', '#ffdd33', '#33ff88', '#3388ff'][ent.layer - 1];
        }
      });
    });
    
    gameLoop();
  </script>
</body>
</html>`;
  }
  
  private generateTool(arch: AppArchitecture, dissolved: DissolvedState): string {
    return this.generateWebApp(arch, dissolved);
  }
  
  private generateDocument(arch: AppArchitecture, dissolved: DissolvedState): string {
    const text = dissolved.source.raw || '';
    
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>${dissolved.source.name}</title>
  <style>
    :root { --bg: #0a0a0f; --fg: #e0e0e0; --l3: #ffdd33; }
    body { background: var(--bg); color: var(--fg); font-family: Georgia, serif; max-width: 800px; margin: 0 auto; padding: 40px 20px; line-height: 1.6; }
    h1, h2, h3 { color: var(--l3); font-family: 'Courier New', monospace; }
    .gate-mark { display: inline-block; width: 20px; height: 20px; border-radius: 50%; margin-right: 8px; vertical-align: middle; }
    pre { background: #111; padding: 15px; border-radius: 8px; overflow-x: auto; }
    code { background: #111; padding: 2px 6px; border-radius: 3px; font-family: monospace; }
  </style>
</head>
<body>
  <div id="content"></div>
  
  <script>
    const text = `${String(text).replace(/`/g, '\\`').replace(/\\/g, '\\\\')}`;
    
    function renderDocument() {
      const container = document.getElementById('content');
      const lines = text.split('\n');
      
      lines.forEach((line, idx) => {
        const gate = (idx % 64) + 1;
        const colors = ['#ff3333', '#ff8833', '#ffdd33', '#33ff88', '#3388ff'];
        const color = colors[gate % 5];
        
        if (line.startsWith('# ')) {
          const h1 = document.createElement('h1');
          h1.innerHTML = `<span class="gate-mark" style="background:${color}"></span>${line.slice(2)}`;
          container.appendChild(h1);
        } else if (line.startsWith('## ')) {
          const h2 = document.createElement('h2');
          h2.innerHTML = `<span class="gate-mark" style="background:${color}"></span>${line.slice(3)}`;
          container.appendChild(h2);
        } else if (line.startsWith('```')) {
          // Code block handled separately
        } else {
          const p = document.createElement('p');
          p.textContent = line;
          container.appendChild(p);
        }
      });
    }
    
    renderDocument();
  </script>
</body>
</html>`;
  }
}

// ============================================================================
// UNIFIED EXECUTION PIPELINE
// ============================================================================

export class ExecutionPipeline {
  private engine: MorphingNetEngine;
  private analyzer: SubstrateAnalyzer;
  private synthesizer: ArchitectureSynthesizer;
  private executor: NativeExecutor;
  
  constructor(engine?: MorphingNetEngine) {
    this.engine = engine || new MorphingNetEngine();
    this.analyzer = new SubstrateAnalyzer();
    this.synthesizer = new ArchitectureSynthesizer(this.engine);
    this.executor = new NativeExecutor(this.engine);
  }
  
  // Main entry: feed anything, get executable artifact back
  async execute(input: File | Blob | string, outputType: "html" | "zip" | "json" = "html"): Promise<{
    artifact: string;
    analysis: AnalysisResult;
    architecture: AppArchitecture;
    dissolved: DissolvedState;
  }> {
    // 1. INGEST (same as before)
    const ingester = new UniversalIngester();
    const content = await ingester.ingest(input);
    
    // 2. DISSOLVE (same as before)
    const dissolver = new SubstrateDissolver();
    const dissolved = dissolver.dissolve(content);
    
    // 3. ANALYZE (NEW)
    const rawContent = typeof content.raw === 'string' ? content.raw : null;
    const analysis = this.analyzer.analyze(dissolved, rawContent);
    
    // 4. SYNTHESIZE (NEW)
    const architecture = this.synthesizer.synthesize(analysis, dissolved);
    
    // 5. EXECUTE (NEW)
    const artifact = this.executor.execute(architecture, dissolved);
    
    return { artifact, analysis, architecture, dissolved };
  }
}

export default ExecutionPipeline;
