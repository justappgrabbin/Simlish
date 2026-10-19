// Agents in the gaming area, BUILT BY THE SWARM from her SynthWorld agent data (vendor/synthworld-data):
//  template agents/generator.json, trait list agents/cynthia_traits.json + traits/*.json, behaviours behaviors/utility_*.json,
//  nudges.json, yijing/scenes.json, cnl/sentences.txt.
// Decision rule = deterministic port of her Agent.gd/_decide_and_act + Utility.gd.score: score every ready behaviour with its
// linear expression over the drives, take the best (strict >, so the first listed wins ties), apply Agent.gd's effects
// (CookMeal hunger -0.6 creation +0.1; Socialize social +0.2; Wander: movement only), set the cooldown.
// Agent.gd metabolism per second (hunger +0.02, rest -0.01) is applied once per turn: ONE TURN = ONE SECOND (her decide-every-second rule).
// Her Agent.gd uses rng.randomize() only for wander direction; here Wander picks a direction from the turn number (FILLED).
// FILLED (no source): agents per user level (level+1, max 3), which trait each agent takes (her cynthia_traits order),
// agent authority = the user's own authority (§5 agent constraint: an agent reflects only the person's own energy),
// the line an agent says (her CNL sentence pattern "agent:… | verb:… | object:…" rendered as text).
import {SwarmComponent} from '../../swarm-component.mjs';
import {SYNTHWORLD as D} from './synthworld-data.mjs';
export const AGENT_FILLED_IN=Object.freeze(['agents per level (level+1, max 3)','trait assignment order','agent authority = user authority (§5)','Wander direction from turn number','CNL line rendering','1 turn = 1 second']);
// Utility.gd.score: linear expression over drives. Small parser (numbers, names, + - * ( )), no eval.
export function score(expr,ctx){let i=0;const s=expr.replace(/\s+/g,'');
 const num=()=>{const m=/^\d+(\.\d+)?/.exec(s.slice(i));if(m){i+=m[0].length;return +m[0]}const n=/^[a-z_]+/i.exec(s.slice(i));if(n){i+=n[0].length;return +(ctx[n[0]]??0)}if(s[i]==='('){i++;const v=add();if(s[i++]!==')')throw new Error('Utility parse error');return v}if(s[i]==='-'){i++;return -num()}throw new Error('Utility parse error for '+expr)};
 const mul=()=>{let v=num();while(s[i]==='*'){i++;v*=num()}return v};const add=()=>{let v=mul();while(s[i]==='+'||s[i]==='-'){const o=s[i++],w=mul();v=o==='+'?v+w:v-w}return v};
 const v=add();if(i!==s.length)throw new Error('Utility parse error for '+expr);return v}
const BEHAVIORS=['behaviors/utility_cook.json','behaviors/utility_social.json','behaviors/utility_wander.json'].map(k=>D[k]);
const clamp=x=>Math.min(1,Math.max(0,x));
const builder=new SwarmComponent({id:'agent-builder',kind:'PSO',purpose:'Build an agent from her SynthWorld data',members:[
 {id:'template',check:i=>!!i.template.drives},{id:'behaviours',check:i=>i.behaviors.every(b=>b.action&&b.score)},{id:'traits-known',check:i=>i.traits.every(t=>D['traits/'+t+'.json'])},
 {id:'authority',check:i=>!!i.authority},{id:'nudges',check:i=>Array.isArray(D['nudges.json'])},{id:'scores-parse',check:i=>i.behaviors.every(b=>Number.isFinite(score(b.score,{})))}]});
export function buildAgents({chart,level=0}){const tpl=D['agents/generator.json'],traitIds=D['agents/cynthia_traits.json'],n=Math.min(3,level+1);
 return Array.from({length:n},(_,k)=>{const traits=[traitIds[k%traitIds.length]],drives={social:.5,creation:.5,rest:.5,exploration:.5,hunger:.2,...tpl.drives};
  for(const t of traits)for(const [d,v] of Object.entries(D['traits/'+t+'.json'].modifiers?.drives||{}))drives[d]=clamp(drives[d]+v);
  const r=builder.runSync({template:tpl,behaviors:BEHAVIORS,traits,authority:chart.hd.authority});
  return {id:`agent-${k+1}-${traits.join('+')}`,name:`${tpl.name} · ${traits.join(', ')}`,template:tpl.name,traits,authority:chart.hd.authority,drives,cooldowns:{},turn:0,hexagram_code:r.hexagram.bits,hexagram_number:r.hexagram.number,swarmStatus:r.status,log:[]}})}
export function applyNudge(agent,nudgeId){const n=D['nudges.json'].find(x=>x.id===nudgeId);if(!n)throw new Error('unknown nudge '+nudgeId);const a=structuredClone(agent);for(const [d,v] of Object.entries(n.effect))a.drives[d]=clamp(a.drives[d]+v*(D['agents/generator.json'].nudgeSensitivity?.symbol??1));a.log.push({turn:a.turn,nudge:n.id});return a}
export function agentTurn(agent){const a=structuredClone(agent);a.turn++;a.drives.hunger=clamp(a.drives.hunger+0.02);a.drives.rest=clamp(a.drives.rest-0.01);
 let best=null,bs=-1e9;for(const b of BEHAVIORS){const s=score(b.score,a.drives),ready=!(b.action in a.cooldowns)||a.turn>=a.cooldowns[b.action];if(s>bs&&ready){best=b;bs=s}}
 if(best){a.cooldowns[best.action]=a.turn+Number(best.cooldownSec??60);if(best.action==='CookMeal'){a.drives.hunger=Math.max(0,a.drives.hunger-0.6);a.drives.creation=clamp(a.drives.creation+0.1)}
  else if(best.action==='Socialize')a.drives.social=clamp(a.drives.social+0.2);else if(best.action==='Wander')a.heading=['north','east','south','west'][a.turn%4]}
 const line=best?`agent:${a.name} | verb:${best.action} | score:${bs.toFixed(3)}`:`agent:${a.name} | verb:wait`;a.log.push({turn:a.turn,action:best?.action??null,score:bs,line});a.last={action:best?.action??null,score:bs,line};return a}
export const YIJING_SCENES=D['yijing/scenes.json'],NUDGES=D['nudges.json'],CNL=D['cnl/sentences.txt'];
