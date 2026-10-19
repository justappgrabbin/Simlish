// One player's session in the gaming area: chart -> adaptive program -> her worlds (wrapped) + the repo's line-walk game,
// agents built by the swarm playing alongside, every turn recorded as an Evolution-memory transition, the program
// recomputed after every turn, build mode in the same world. Deterministic apart from her own game code
// (her oracles-thread reads the current time for its changing lines; her seed store stamps createdAt).
import {program,newMemory,recordTransition,lineMoves} from './adaptive.mjs';
import {createGamesHost,HER_WORLDS} from './games-host.mjs';
import {buildAgents,agentTurn} from './agents.mjs';
import {lessonFor,outcomeOf,OUTCOME_RULES} from './lessons.mjs';
import {buildWithAgents} from './build.mjs';
import {describeBody} from './hooks.mjs';
export const WORLDS=Object.freeze([...HER_WORLDS,'line-walk']);
export function createPlaySession({chart,dataDir,visualIdentity,memory=newMemory()}){
 const host=createGamesHost({dataDir,chart}),player=host.player(visualIdentity);let mem=memory,prog=program(chart,mem),agents=buildAgents({chart,level:prog.level}),current=null,entry=null,lesson=null;
 const refresh=()=>{prog=program(chart,mem);const want=Math.min(3,prog.level+1);if(agents.length<want)agents=[...agents,...buildAgents({chart,level:prog.level}).slice(agents.length)];return prog};
 const crewTurn=()=>{agents=agents.map(agentTurn);return agents.map(a=>({agent:a.id,name:a.name,line:a.last.line,action:a.last.action}))};
 return {host,player,get memory(){return mem},get program(){return prog},get agents(){return agents},get world(){return current},body:()=>describeBody(chart),outcomeRules:OUTCOME_RULES,
  async enter(world){if(current&&current!=='line-walk')host.worlds.exitWorld(player.playerId);current=world;
   entry=world==='line-walk'?{moves:lineMoves(prog.address.gate),address:prog.address}:await host.worlds.enterWorld(player.playerId,world);
   lesson=lessonFor(world,{program:prog,entry,chart});return {world,entry,lesson,agents:crewTurn(),program:prog}},
  async choose(choice){if(!current)throw new Error('enter a world first');let result,t;
   if(current==='line-walk'){const mv=lineMoves(prog.address.gate).find(m=>m.line===Number(choice?.line??choice));if(!mv)throw new Error('line 1-6');
    result={toGate:mv.toGate,line:mv.line,keynote:mv.keynote};const outcome=outcomeOf('line-walk',result,{lesson,chart});
    t={kind:'line-walk',game:'line-walk',fromGate:prog.address.gate,toGate:mv.toGate,line:mv.line,gate:outcome==='success'?mv.toGate:null,outcome,view:'line-walk'}}
   else{result=await host.worlds.handleChoice(player.playerId,choice);const outcome=outcomeOf(current,result,{lesson,chart});
    t={kind:'game',game:current,gate:lesson.need.gate,outcome,view:'worlds'}}
   mem=recordTransition(mem,t);const before=prog.fingerprint;refresh();
   if(current==='line-walk')entry={moves:lineMoves(prog.address.gate),address:prog.address};
   lesson=lessonFor(current,{program:prog,entry:current==='line-walk'?entry:{...entry,...(result.mechanics?{mechanics:result.mechanics}:{})},chart});
   return {result,transition:mem.transitions.at(-1),lesson,agents:crewTurn(),program:prog,programChanged:before!==prog.fingerprint}},
  async exit(){if(current&&current!=='line-walk')host.worlds.exitWorld(player.playerId);current=null},
  build(opts){return buildWithAgents({...opts,chart,program:prog,agents}).then(r=>{agents=r.agents;mem=recordTransition(mem,{kind:'build',game:'build',gate:null,outcome:'played',view:'build'});refresh();return r})}}}
