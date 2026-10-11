import {GATES,GATE_SOURCE} from './source-gates.mjs';
import {SYNTAX_OPERATORS} from './source-syntax.mjs';
import {LexicalFillEngine} from './lexical-fill.mjs';
const contains=(text,phrase)=>(' '+text.toLowerCase().replace(/[^\p{L}\p{N}]+/gu,' ')+' ').includes(' '+phrase.toLowerCase().replace(/[^\p{L}\p{N}]+/gu,' ').trim()+' ');
export function previewSentence(description,{memory=[],persist=()=>{}}={}) {
 if(typeof description!=='string'||!description.trim())throw Error('Describe the situation first.');
 const candidates=GATES.filter(g=>contains(description,g.name)||g.theme.split('/').some(word=>contains(description,word.trim())));
 const dimensions=['Movement','Evolution','Being','Design','Space'].filter(d=>contains(description,d));
 const lexemes=candidates.map(g=>({slot:'gate',word:g.name,position:g.number,source:GATE_SOURCE}));
 lexemes.push(...dimensions.map(d=>({slot:'dimension',word:d,source:{driveId:'1x7xzzzaxg_SdhHM5uteiZ21kGsoRfdK5',section:'Core Interpretive Framework'}})));
 const deterministic={};
 if(candidates.length===1){const gate=candidates[0];deterministic.gate=gate.number;deterministic.center=gate.center;
  lexemes.push({slot:'center',word:gate.center,source:GATE_SOURCE},{slot:'biology',word:gate.organ,source:GATE_SOURCE});
  const lines=gate.lines.filter(line=>contains(description,line.name.trim()));
  if(lines.length===1){deterministic.line=lines[0].line;lexemes.push({slot:'line',word:lines[0].name.trim(),source:GATE_SOURCE});}
 }
 if(dimensions.length===1)deterministic.dimension=dimensions[0];
 const punctuationMarks=SYNTAX_OPERATORS.filter(op=>op.mark.length===1&&description.includes(op.mark)).map(op=>op.mark);
 const structure=new LexicalFillEngine({memory,persist}).resolve({lexemes,deterministic,punctuationMarks,rulebookVersion:'supplied-master-and-hd-donor.v1'});
 return {description,structure,gateCandidates:candidates.map(g=>({gate:g.number,name:g.name,theme:g.theme,source:GATE_SOURCE})),
  summary:candidates.length ? candidates.map(g=>g.name).join(' · ') : 'No gate-name or theme match in the connected source table.',
  status:candidates.length===1?'source-match':candidates.length>1?'multiple-source-matches':'no-source-match'};
}
