import {GATES,GATE_SOURCE} from './source-gates.mjs';
import {gateToFuXiDecimal,fuXiDecimalToGate} from './vendor/kingwen.mjs';
import {autoNovelAutomaton} from './vendor/klein/index.mjs';
const dimensions=['Movement','Evolution','Being','Design','Space'];
const gateFor=number=>{const gate=GATES.find(g=>g.number===number);if(!gate)throw Error('Choose a source-backed gate.');return gate;};
export function beginPath(preview,identity=null){
 if(preview.status!=='source-match')throw Error('Your description needs one clear theme to start. Refine it and preview again.');
 const gate=gateFor(preview.structure.slots.gate);
 return {version:1,identityId:identity?.id??null,opening:structuredClone(preview),gate:gate.number,views:dimensions.map(dimension=>({dimension,openingLexemes:structuredClone(preview.structure.positioned),observations:[]})),history:[]};
}
export function pathChoices(path){return gateFor(path.gate).lines.map(line=>({id:line.line,label:line.name.trim(),meaning:line.keynote,source:GATE_SOURCE}));}
export async function choosePath(path,line){
 if(!Number.isInteger(line)||line<1||line>6)throw Error('Choose an available line.');
 const current=gateFor(path.gate),choice=current.lines.find(item=>item.line===line);
 const next=gateFor(fuXiDecimalToGate(gateToFuXiDecimal(current.number)^(1<<(line-1))));
 const novel=autoNovelAutomaton();
 await novel.call({operation:'register',domain:{id:'sentence-path',primitives:[{id:'situation'},{id:'choice'},{id:'consequence'}],combinators:[{id:'source-line-transition.v1',type:'sequence',inputs:['situation','choice','consequence'],output:'scene',apply:(situation,choice,consequence)=>({situation,choice,consequence})}]}});
 const story=await novel.call({operation:'generate',spec:{domain:'sentence-path',seeds:[{type:'situation',value:current.name},{type:'choice',value:choice.name.trim()},{type:'consequence',value:next.name}]}});
 const transition={index:path.history.length,fromGate:current.number,toGate:next.number,line,choice:choice.name.trim(),meaning:choice.keynote,source:GATE_SOURCE,lineRule:'King Wen/Fu Xi: bottom line is bit 0; selected line toggles',lineage:story.lineage};
 const result=structuredClone(path);result.history.push(transition);result.gate=next.number;
 for(const view of result.views)view.observations.push({...transition,dimension:view.dimension});
 return {path:result,scene:{title:next.name,description:choice.keynote,theme:next.theme},transition};
}
