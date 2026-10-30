/**
 * Klein capability/module contracts served by Synthia's addressed ATO state space.
 * A Klein-named tool is a composition contract, not a single opaque function.
 * Modules remain independently addressable/reusable so different residents can
 * express different lawful compositions of the same tool family.
 */
export const KLEIN_MODULES = Object.freeze({
  morphology: Object.freeze({ id:'morphology', role:'segment/classify surface forms', level:4, reusable:true, sources:['vendor/resonance-engine/generators/MorphoAnalyzer.js'] }),
  semanticNetwork: Object.freeze({ id:'semantic-network', role:'store/traverse semantic relations and inheritance', level:4, reusable:true, sources:['vendor/ato-core/src/klein-tools.mjs','runtime/SemanticArtifactCompiler.js'] }),
  distributionalInference: Object.freeze({ id:'distributional-inference', role:'distributional semantic inference (DISEMINER family)', level:4, reusable:true, sources:['runtime/diseminer.js','UPGRADES/adapters/EnhancedDiseminer.js','vendor/ato-core/src/klein-tools.mjs'] }),
  phraseStructure: Object.freeze({ id:'phrase-structure', role:'learn/apply phrase structure', level:5, reusable:true, sources:['runtime/DeepStructureLearner.js','organism/LawfulGrammarConstructor.mjs'] }),
  transformationRules: Object.freeze({ id:'transformation-rules', role:'learn/apply deep↔surface transformations', level:5, reusable:true, sources:['runtime/DeepStructureLearner.js','runtime/SurfaceTransformEngine.js'] }),
  grammarInduction: Object.freeze({ id:'grammar-induction', role:'induce/revise generative grammar from evidence', level:5, reusable:true, sources:['organism/LawfulGrammarConstructor.mjs','substrate/mesh072/core/klein_autoling_learner.mjs'] }),
  recognition: Object.freeze({ id:'recognition', role:'recognize structures using learned rules', level:5, reusable:true, sources:['vendor/ato-core/src/klein-tools.mjs','runtime/autoling.js'] }),
  generation: Object.freeze({ id:'generation', role:'generate structures from semantic/grammar state', level:5, reusable:true, sources:['vendor/ato-core/src/klein-tools.mjs','organs/legacy/automatic-novel-writer.js'] }),
  paraphrase: Object.freeze({ id:'paraphrase', role:'generate structurally equivalent surface realizations', level:5, reusable:true, sources:['runtime/SurfaceTransformEngine.js','vendor/ato-core/src/quality-transfer.mjs'] }),
  styleControl: Object.freeze({ id:'style-control', role:'control generative realization/style without losing structure', level:5, reusable:true, sources:['runtime/StyleControlEngine.js'] }),
  symbolicValidation: Object.freeze({ id:'symbolic-validation', role:'validate structural/equivalence constraints', level:5, reusable:true, sources:['vendor/ato-core/src/boolean-ato.mjs','runtime/ArtifactValidator.js'] }),
  analogy: Object.freeze({ id:'analogy', role:'complete/transfer relational structure by analogy', level:5, reusable:true, sources:['runtime/SacralAnalogyEngine.js','vendor/ato-core/src/klein-tools.mjs'] }),
  temporalCausalStructure: Object.freeze({ id:'temporal-causal-structure', role:'maintain ordered causal/temporal progression', level:6, reusable:true, sources:['organs/legacy/automatic-novel-writer.js','organism/GenerativeEpisodicMemory.mjs'] }),
  narrativeStructure: Object.freeze({ id:'narrative-structure', role:'compose scenes/characters/events/themes into long-form structures', level:7, reusable:true, sources:['organs/legacy/automatic-novel-writer.js','vendor/ato-core/src/klein-tools.mjs'] }),
  persistentLearning: Object.freeze({ id:'persistent-learning', role:'retain learned rules/examples/consequences across runs', level:7, reusable:true, sources:['substrate/mesh072/core/klein_autoling_learner.mjs','organism/GenerativeEpisodicMemory.mjs'] }),
});

export const KLEIN_TOOL_CONTRACTS = Object.freeze({
  autoling: Object.freeze({
    name:'AutoLing',
    minimumModules:Object.freeze(['morphology','semantic-network','distributional-inference','phrase-structure','transformation-rules','grammar-induction','recognition','generation','persistent-learning','symbolic-validation']),
    description:'Bidirectional language learning/generation architecture operating over structured semantic state.'
  }),
  diseminer: Object.freeze({
    name:'DISEMINER',
    minimumModules:Object.freeze(['semantic-network','distributional-inference','persistent-learning']),
    description:'Distributional-semantics inference architecture with retained evidence.'
  }),
  autonovel: Object.freeze({
    name:'AutoNovel',
    minimumModules:Object.freeze(['semantic-network','phrase-structure','transformation-rules','grammar-induction','generation','paraphrase','style-control','temporal-causal-structure','narrative-structure','persistent-learning','symbolic-validation']),
    description:'Long-form automatic narrative construction architecture, not a beat planner.'
  }),
  paraphraser: Object.freeze({
    name:'Paraphraser', minimumModules:Object.freeze(['semantic-network','transformation-rules','generation','paraphrase','symbolic-validation']),
    description:'Structure-preserving alternative realization architecture.'
  }),
});

const byId = new Map(Object.values(KLEIN_MODULES).map(module => [module.id, module]));
export class KleinModuleRegistry {
  constructor(extra=[]){ this.modules=new Map(byId); for(const module of extra) this.register(module); }
  register(module){ if(!module?.id) throw new TypeError('Klein module requires id'); this.modules.set(module.id,Object.freeze({...module})); return this; }
  get(id){ return this.modules.get(id)||null; }
  compose(toolFamily, requested=[]){
    const contract=KLEIN_TOOL_CONTRACTS[String(toolFamily||'').toLowerCase()]||null;
    const ids=[...new Set([...(contract?.minimumModules||[]),...requested])];
    const modules=ids.map(id=>this.get(id)).filter(Boolean);
    const missing=ids.filter(id=>!this.get(id));
    return Object.freeze({toolFamily:toolFamily||null,contract,modules:Object.freeze(modules),missing:Object.freeze(missing),complete:missing.length===0});
  }
  list(){ return Object.freeze([...this.modules.values()]); }
}
export default KleinModuleRegistry;
