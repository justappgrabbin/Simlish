// Clean hook for Lil Trinity's separate HD body-and-science module (built on its own branch; NOT touched here).
// Contract only. A module registers {id, version, describe(chart) -> {title, sections:[{heading,text,source}]}};
// the play page shows each registered module's output in the "Body & science" panel for the current chart.
// Nothing is registered on this branch.
const modules=new Map();
export const HOOK_CONTRACT=Object.freeze({register:'registerBodyScienceModule({id, version, describe(chart)})',describe:'describe(chart) -> {title, sections:[{heading, text, source}]}',input:'chart = integrated/chart-engine.mjs calculateBirthChart() output'});
export function registerBodyScienceModule(m){if(!m||typeof m.id!=='string'||typeof m.describe!=='function')throw new TypeError('module needs id and describe(chart)');modules.set(m.id,m);return ()=>modules.delete(m.id)}
export const bodyScienceModules=()=>[...modules.values()];
export function describeBody(chart){return bodyScienceModules().map(m=>{try{return {id:m.id,version:m.version??null,...m.describe(chart)}}catch(e){return {id:m.id,error:String(e.message||e)}}})}
if(typeof globalThis!=='undefined')globalThis.SynthiaBodyScienceHook=Object.freeze({registerBodyScienceModule,HOOK_CONTRACT});
