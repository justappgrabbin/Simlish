// Life lessons per game, tied to the user's chart + state. Text comes from her sources:
//  - gate/line keynotes: source-gates.mjs (her donor table, GATE_SOURCE)
//  - her game's own mechanic text: hdRpg decisionMechanic / npcInteractionStyle / connectionMechanic,
//    iching-story hexagramInPlay + conflict, oracles-thread BaZi element analysis (dominant / weakest)
// FILLED (no source): WHICH game carries WHICH need (below), and the outcome rule that counts a turn as
// success / miss / played (OUTCOME_RULES). Her §6.6 says games teach HD by letting the person watch their own field behave.
import {GATES} from '../../source-gates.mjs';
export const GAME_LESSON_FILL='FILLED: game-to-need assignment and outcome rules are not in her sources';
const kn=(gate,line)=>{const g=GATES.find(x=>x.number===gate),l=g?.lines.find(x=>x.line===line);return g?{gate,line,title:`${gate}.${line} ${g.name}: ${l?.name.trim()??''}`,keynote:l?.keynote??null}:null};
export function lessonFor(world,{program,entry,chart}){const t=program.lessons,pick=i=>t[Math.min(i,t.length-1)];
 if(world==='hd-rpg'){const m=entry.mechanics,l=pick(0);return {world,need:l,chart:kn(l.gate,l.line),
   game:`Decision mechanic: ${m.decisionMechanic}; NPCs meet you with ${m.npcInteractionStyle}; connection: ${m.connectionMechanic}.`,
   practise:`Use the ability of gate ${l.gate} (your ${l.why.join(', ')||'gate'}).`,suggested:{type:'use-ability',abilityGate:l.gate},filled:GAME_LESSON_FILL}}
 if(world==='iching-story'){const sc=entry.hexagramInPlay?entry:entry.scene||{},h=sc.hexagramInPlay||{number:program.address.gate,line:program.address.line}; // re-entry returns her last scene, whose hexagram sits under .scene
  return {world,need:{gate:h.number,line:h.line},chart:kn(h.number,h.line),game:sc.conflict??null,
   practise:'Read the situation through your hexagram in play before you choose.',suggested:'observe',filled:GAME_LESSON_FILL}}
 if(world==='oracles-thread'){const e=entry.bazi?.elements,l=pick(1);return {world,need:l,chart:kn(l.gate,l.line),
   game:e?`Your birth pillars run ${e.dominant}-dominant and ${e.weakest}-starved.`:null,practise:'Wait, gesture or combine — watch which band your act lands in.',suggested:{mode:'wait'},filled:GAME_LESSON_FILL}}
 if(world==='line-walk'){const a=program.address;return {world,need:{gate:a.gate,line:a.line},chart:kn(a.gate,a.line),game:'Each step changes exactly one line of your current hexagram (her game-path rule).',
   practise:'Pick the line whose keynote you recognise in yourself today.',suggested:{line:a.line},filled:GAME_LESSON_FILL}}
 throw new Error('no lesson for '+world)}
export const OUTCOME_RULES=Object.freeze({
 'hd-rpg':'success: used the lesson gate\'s ability, or passed a confrontation; miss: failed confrontation; else played',
 'iching-story':'success: an NPC disposition shift > 0; miss: < 0; else played',
 'oracles-thread':'success: band harmonic or perfect-harmony (her BANDS); miss: distortion or severe-distortion; else played',
 'line-walk':'success: the step lands on one of your own gates; else played'});
export function outcomeOf(world,result,{lesson,chart}){
 if(world==='hd-rpg'){const c=result.consequence;if(c.kind==='ability')return c.ability.canonicalGate===lesson.need.gate?'success':'played';if(c.kind==='confrontation')return c.passed?'success':'miss';return 'played'}
 if(world==='iching-story'){const s=(result.resolution?.relationshipEffects||[]).reduce((a,r)=>a+(r.dispositionShift||0),0);return s>0?'success':s<0?'miss':'played'}
 if(world==='oracles-thread'){const b=result.band;return b==='harmonic'||b==='perfect-harmony'?'success':/distortion/.test(b||'')?'miss':'played'}
 if(world==='line-walk')return chart.hd.gates.includes(result.toGate)?'success':'played';
 throw new Error('unknown world '+world)}
