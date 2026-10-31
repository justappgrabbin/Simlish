import {previewSentence} from './sentence-preview.mjs';
import {beginPath,choosePath,pathChoices} from './game-path.mjs';

// These are GAME locations and resource rules, not newly assigned Human Design meanings.
// The canonical gate/line transitions remain delegated to the existing game-path module.
export const BEACONS = Object.freeze([
  {id:'movement',name:'Movement',title:'The first step',x:0,z:-7.5,color:0xf5b566,challenge:'A path appears only when someone takes the first step.'},
  {id:'evolution',name:'Evolution',title:'The changing grove',x:7.1,z:-2.3,color:0x6bebae,challenge:'What could this place become if you tried something new?'},
  {id:'being',name:'Being',title:'The mirror pool',x:4.4,z:6.2,color:0x69bdf5,challenge:'Someone needs a place where they can belong.'},
  {id:'design',name:'Design',title:'The workshop',x:-4.4,z:6.2,color:0xeab0f7,challenge:'There are broken pieces here. What will you make of them?'},
  {id:'space',name:'Space',title:'The open sky',x:-7.1,z:-2.3,color:0xf5dd7f,challenge:'Five separate lights can make one shared space.'}
]);
const maxFocus=9;
const ids=new Set(BEACONS.map(b=>b.id));

export function startRun(now=Date.now()) {
  return {version:1,focus:maxFocus,score:0,recharges:0,lit:[],
    path:beginPath(previewSentence('Movement Conflict')),
    journal:[],startedAt:now,finishedAt:null};
}

export function offerChoices(run) {
  return pathChoices(run.path).map(choice=>({
    ...choice,cost:Math.ceil(choice.id/2),points:choice.id*4,
    affordable:run.focus>=Math.ceil(choice.id/2)
  }));
}

export async function activateBeacon(run,beaconId,line,now=Date.now()) {
  if(!ids.has(beaconId))throw Error('Unknown light.');
  if(run.finishedAt!==null)throw Error('This journey is already complete.');
  if(run.lit.includes(beaconId))throw Error('This light is already awake.');
  const choice=offerChoices(run).find(item=>item.id===line);
  if(!choice)throw Error('Choose one of the six source-backed lines.');
  if(!choice.affordable)throw Error('Not enough focus. Return to the central camp to recharge.');
  const {path,scene,transition}=await choosePath(run.path,line);
  const beacon=BEACONS.find(item=>item.id===beaconId);
  const lit=[...run.lit,beaconId];
  const journal=[...run.journal,{
    type:'light',beacon:beaconId,dimension:beacon.name,choice:choice.label,
    line,cost:choice.cost,points:choice.points,fromGate:transition.fromGate,
    toGate:transition.toGate,transition,at:now
  }];
  const next={...run,focus:run.focus-choice.cost,score:run.score+choice.points,
    lit,path,journal,finishedAt:lit.length===BEACONS.length?now:null};
  return {run:next,scene,transition};
}

export function rechargeAtCamp(run,now=Date.now()) {
  if(run.finishedAt!==null)throw Error('Journey already complete.');
  if(run.focus>=maxFocus)throw Error('Your focus is already full.');
  const focus=Math.min(maxFocus,run.focus+5),penalty=Math.min(run.score,10);
  return {...run,focus,score:run.score-penalty,recharges:run.recharges+1,
    journal:[...run.journal,{type:'rest',focusBefore:run.focus,focusAfter:focus,scoreCost:penalty,at:now}]};
}

export function restoreRun(data) {
  if(!data||data.version!==1||!Number.isInteger(data.focus)||data.focus<0||data.focus>maxFocus
    ||!Number.isInteger(data.score)||data.score<0||!Number.isInteger(data.recharges)||data.recharges<0
    ||!Array.isArray(data.lit)||new Set(data.lit).size!==data.lit.length
    ||data.lit.some(id=>!ids.has(id))||!Array.isArray(data.journal)
    ||!data.path||!Number.isInteger(data.path.gate)||data.path.gate<1||data.path.gate>64
    ||!Array.isArray(data.path.history)||data.path.history.length!==data.lit.length
    ||!Array.isArray(data.path.views)||data.path.views.length!==5
    ||!(data.finishedAt===null||Number.isFinite(data.finishedAt))
    ||!Number.isFinite(data.startedAt)||data.lit.length===5 !== (data.finishedAt!==null)){
    throw Error('Saved journey is damaged or from another version. Start a new journey.');
  }
  return data;
}
