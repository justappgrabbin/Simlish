// Adaptive programming per user (deterministic, no Math.random). Built only from her sources:
//  - Architecture §1.7 (combine/combined.md:1530): relevance of each charting system is personalized by the person's DEFINED CENTERS
//    ("defined Head/Ajna and an active Integration circuit -> Sidereal significant"; "defined Sacral -> Body/Tropical dominant").
//  - Architecture §5 Agent Constraint: an agent reflects only the energy the person's own design carries, never more
//    -> content and tools are drawn ONLY from the person's own gates.
//  - Architecture §6.3 success-outcome feedback loop, §6.6 "games that teach HD by having the person watch their own field-organism behave".
//  - Evolution = memory ("I Remember", combined.md:1231/1239): every interaction is an appended transition (before -> after), never overwritten.
//  - Relational rule: the user's address moves only by a one-line change (Hamming-1), as in her game-path.mjs (selected line toggles).
//  - PSO genotype -> GA phenotype: her SwarmPair (swarm-nsnf.mjs), six members in packs 3-2-1.
// Every rule below that is NOT one of those is marked FILLED (filled in, no source) in FILLED_IN.
import {SwarmComponent,SwarmPair,EventLog} from '../../swarm-nsnf.mjs';
import {gateToFuXiDecimal,fuXiDecimalToGate} from '../../vendor/kingwen.mjs';
import {GATES,GATE_SOURCE} from '../../source-gates.mjs';
import {CHANNELS,CENTER_OF_GATE} from '../chart-engine.mjs';
import {sha256} from '../../sha256.mjs';
export const INVESTIGATING=Object.freeze(['five vs six bases','Mawangdui version (Evolution layer)','EmergentEdgeResolver vs the 36 channels (channels here are only her chart-engine reference list)','302° vs 272° wheel','gate 22 center','Draconic relevance: §1.7 names no defined-center rule for it']);
export const FILLED_IN=Object.freeze([
 'need score weights (hanging gate +2, design-only +1, success -1.5, miss +0.5)',
 'level = successful interactions / 3 (floor), capped at 3',
 'view order rule (most-used view first after the chart default)',
 'which game teaches which kind of need (see GAME_LESSONS in games-host.mjs)']);
const gateOf=n=>GATES.find(g=>g.number===n);
const INTEGRATION=new Set(['10-20','10-34','10-57','20-34','20-57','34-57']); // the Integration channels named in her chart-engine CHANNELS list
export const startAddress=chart=>{const sun=chart.activations.find(a=>a.orientation==='personality'&&a.body==='Sun');
 return {gate:sun.gate,line:sun.line,color:sun.color,tone:sun.tone,base:sun.base,state:gateToFuXiDecimal(sun.gate),source:'personality Sun'}};
// §1.7 relevance by defined centers
export function systemRelevance(chart){const d=new Set(chart.hd.definedCenters),ch=new Set(chart.hd.channels.map(([a,b])=>`${Math.min(a,b)}-${Math.max(a,b)}`));
 const integration=[...ch].some(c=>INTEGRATION.has(c)),w={tropical:d.has('Sacral')?1:0,sidereal:d.has('Head')&&d.has('Ajna')&&integration?1:0,draconic:0};
 const order=Object.entries(w).sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0])).map(([k])=>k);
 return {weights:w,order,rule:'architecture §1.7',investigating:w.tropical+w.sidereal===0?'no §1.7 rule applies to this chart; all systems shown equally':null}}
// Evolution memory: append-only, hash-chained transitions
export function newMemory(){return {transitions:[],head:'0'.repeat(64)}}
export function recordTransition(memory,t){const before=t.fromGate??(memory.transitions.length?memory.transitions.at(-1).after:null);
 if(t.toGate!=null&&t.fromGate!=null){const h=gateToFuXiDecimal(t.fromGate)^gateToFuXiDecimal(t.toGate);if(h&&(h&(h-1)))throw new RangeError('relational rule: an address moves by one line (Hamming-1) only')}
 const entry={n:memory.transitions.length+1,kind:t.kind,game:t.game,fromGate:t.fromGate??null,toGate:t.toGate??null,line:t.line??null,gate:t.gate??t.toGate??null,view:t.view??null,outcome:t.outcome,before,after:t.toGate??before,prev:memory.head};
 entry.hash=sha256(JSON.stringify(entry));return {transitions:[...memory.transitions,Object.freeze(entry)],head:entry.hash}}
export function verifyMemory(m){let prev='0'.repeat(64);for(const e of m.transitions){const {hash,...rest}=e;if(rest.prev!==prev||sha256(JSON.stringify(rest))!==hash)return false;prev=hash}return prev===m.head}
const tally=m=>{const g={},v={};for(const t of m.transitions){if(t.gate!=null){const x=g[t.gate]||(g[t.gate]={success:0,miss:0});x[t.outcome==='success'?'success':'miss']++}if(t.view)v[t.view]=(v[t.view]||0)+1}return {g,v}};
export function needs(chart,memory){const own=new Set(chart.hd.gates),pers=new Set(chart.hd.personalityGates),{g}=tally(memory);
 return chart.hd.gates.map(n=>{const partners=CHANNELS.filter(([a,b])=>a===n||b===n).map(([a,b])=>a===n?b:a),hanging=partners.length>0&&!partners.some(p=>own.has(p)),designOnly=!pers.has(n),s=g[n]||{success:0,miss:0};
  const score=(hanging?2:0)+(designOnly?1:0)-1.5*s.success+0.5*s.miss;const gate=gateOf(n);
  return {gate:n,name:gate?.name??null,center:CENTER_OF_GATE[n],hanging,designOnly,missingPartners:hanging?partners:[],successes:s.success,misses:s.miss,score}}).sort((a,b)=>b.score-a.score||a.gate-b.gate)}
export const levelOf=m=>Math.min(3,Math.floor(m.transitions.filter(t=>t.outcome==='success').length/3));
export const currentAddress=(chart,m)=>{const s=startAddress(chart),last=[...m.transitions].reverse().find(t=>t.toGate!=null);return last?{...s,gate:last.toGate,line:last.line,state:gateToFuXiDecimal(last.toGate),source:'last transition #'+last.n}:s};
// six-line moves from the current address (her game-path rule) with her line keynotes
export const lineMoves=gate=>gateOf(gate).lines.map(l=>({line:l.line,label:l.name.trim(),keynote:l.keynote,toGate:fuXiDecimalToGate(gateToFuXiDecimal(gate)^(1<<(l.line-1)))}));
// PSO genotype: checks over the chart and memory -> hexagram code
const genotype=new SwarmComponent({id:'adaptive-program',kind:'PSO',purpose:'Check what this person\'s programming can be built from',members:[
 {id:'start-address',check:i=>!!startAddress(i.chart)},{id:'own-gates',check:i=>i.chart.hd.gates.length>0},{id:'authority-known',check:i=>!!i.chart.hd.authority},
 {id:'has-hanging-gate',check:i=>needs(i.chart,i.memory).some(n=>n.hanging)},{id:'has-open-center',check:i=>i.chart.hd.openCenters.length>0},
 {id:'has-history',check:i=>i.memory.transitions.length>0}]});
const VIEWS=['chart','line-walk','agents','worlds','build'];
const trait=(g,i)=>{const {chart,memory}=i,need=needs(chart,memory),{v}=tally(memory),addr=currentAddress(chart,memory),level=levelOf(memory),rel=systemRelevance(chart);
 const defaultViews=chart.hd.openCenters.length>=5?['chart','agents','line-walk','worlds','build']:VIEWS; // FILLED: many open centers -> show contact with agents earlier
 const views=[...defaultViews].sort((a,b)=>(v[b]||0)-(v[a]||0)||defaultViews.indexOf(a)-defaultViews.indexOf(b));
 const focus=need.slice(0,3),target=focus[0];
 const lessons=focus.map(n=>{const gate=gateOf(n.gate),line=chart.activations.find(a=>a.gate===n.gate)?.line??1,l=gate?.lines.find(x=>x.line===line);
  return {gate:n.gate,line,title:`${n.gate}.${line} ${gate?.name??''}: ${l?.name.trim()??''}`,text:l?.keynote??null,source:GATE_SOURCE,why:[n.hanging?'hanging gate (its channel partner '+n.missingPartners.join('/')+' is not in your chart)':null,n.designOnly?'design-only (unconscious) activation':null,n.successes?`${n.successes} success(es) recorded`:null].filter(Boolean)}});
 const tools=[...rel.order.map(s=>({id:'frame.'+s,label:s+' frame',weight:rel.weights[s],rule:'§1.7'})),{id:'gate-lookup',label:'your gates only ('+chart.hd.gates.length+')',rule:'§5 agent constraint'}];
 return {address:addr,level,target,lessons,tools,views,gates:chart.hd.gates,moves:lineMoves(addr.gate),genotype:{hexagram_code:g.hexagram_code,packs:g.packs,failed:g.failed}}};
const phenotype=new SwarmComponent({id:'adaptive-view',kind:'GA',purpose:'What the person sees',members:[
 {id:'has-lessons',check:i=>i.visible.lessons.length>0},{id:'own-gates-only',check:i=>i.visible.lessons.every(l=>i.input.chart.hd.gates.includes(l.gate))},{id:'six-moves',check:i=>i.visible.moves.length===6},
 {id:'tools',check:i=>i.visible.tools.length>0},{id:'views',check:i=>i.visible.views.length===VIEWS.length},{id:'address',check:i=>Number.isInteger(i.visible.address.state)}]});
export function program(chart,memory=newMemory()){const pair=new SwarmPair({id:'adaptive',genotype,phenotype:{component:phenotype,trait}}),r=pair.forward({chart,memory});
 const p=r.phenotype.visible;return {...p,status:r.status,phenotypeStatus:r.phenotype.status,fingerprint:sha256(JSON.stringify({l:p.lessons.map(x=>x.gate),t:p.tools.map(x=>x.id),v:p.views,a:p.address.gate,lv:p.level})).slice(0,16),relevance:systemRelevance(chart),investigating:INVESTIGATING,filledIn:FILLED_IN}}
// Frames x wheels (Alexis, 2026-10-19, NOT confirmed -> INVESTIGATING; neither hypothesis is chosen).
// Default 'current' is unchanged behaviour: every frame reads gates through her Fu Xi decimal (gateToFuXiDecimal).
export const FRAME_WHEEL_MAPPINGS=Object.freeze({
 current:Object.freeze({default:true,draconic:'fu-xi',tropical:'fu-xi',sidereal:'fu-xi',note:'unchanged code path'}),
 'hypothesis-A':Object.freeze({draconic:'king-wen',tropical:'fu-xi',sidereal:'mawangdui',note:'Draconic = King Wen, Tropical = Fu Xi, Sidereal = Mawangdui (INVESTIGATING)'}),
 'hypothesis-B':Object.freeze({draconic:'fu-xi',tropical:'fu-xi',sidereal:'fu-xi',note:'all three frames use Fu Xi because we are in the dimension of Being (INVESTIGATING)'})});
export const frameWheel=(frame,mapping='current')=>{const m=FRAME_WHEEL_MAPPINGS[mapping];if(!m)throw new RangeError('unknown frame/wheel mapping '+mapping);return m[frame]??null};
