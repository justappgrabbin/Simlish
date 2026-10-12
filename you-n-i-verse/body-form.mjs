import {bodyControls} from './body-proportions.mjs';
// A local, deterministic planner for descriptive human proportions. It does not
// infer someone's anatomy from a portrait or pretend to understand every role.
export function planBodyForm(description=''){
 const text=String(description).trim().toLowerCase(),proportions=Object.fromEntries(bodyControls.map(([key])=>[key,1])),applied=[];
 const rules=[
  [/\b(female|feminine)\b/,{shoulders:.96,chest:1.08,waist:.92,hips:1.12},'requested feminine form'],
  [/\b(male|masculine)\b/,{shoulders:1.08,chest:1.08,waist:1,hips:.96},'requested masculine form'],
  [/\b(curvy|hourglass)\b/,{waist:.76,hips:1.36,chest:1.18},'curvy'],
  [/\b(slender|slim|lean)\b/,{waist:.88,hips:.94,chest:.94,uArm:.94,fArm:.96,thigh:.95,calf:.96},'slender'],
  [/\b(athletic|muscular|warrior|fighter|guardian)\b/,{shoulders:1.18,chest:1.15,waist:.95,uArm:1.08,thigh:1.08},'athletic'],
  [/\b(stocky|sturdy)\b/,{height:.92,shoulders:1.12,chest:1.16,waist:1.12,hips:1.12},'stocky'],
  [/\b(tall|long[- ]legged)\b/,{height:1.12,thigh:1.04,calf:1.04},'tall'],
  [/\b(short|petite)\b/,{height:.9,uArm:.96,fArm:.96},'short'],
  [/\b(broad|wide)[- ]shoulders?\b/,{shoulders:1.2},'broad shoulders'],
  [/\b(narrow|small)[- ]waist\b/,{waist:.78},'narrow waist'],
  [/\b(wide|full)[- ]hips\b/,{hips:1.3},'wide hips']
 ];
 rules.sort((a,b)=>((a[2]==='athletic'?1:a[2].startsWith('requested')?0:2)-(b[2]==='athletic'?1:b[2].startsWith('requested')?0:2)));
 for(const [pattern,changes,name] of rules){const match=pattern.exec(text);if(!match)continue;const prefix=text.slice(Math.max(0,match.index-25),match.index);if(/\b(no|not|without|avoid|don['’]t|do not)\b(?:\s+\w+){0,4}\s*$/.test(prefix))continue;Object.assign(proportions,changes);applied.push(name);}
 if(applied.includes('curvy')&&applied.includes('slender'))return {ok:false,reason:'The request combines curvy and slender body changes. Describe which proportions you want emphasized.'};
 if(applied.includes('tall')&&applied.includes('short'))return {ok:false,reason:'The request asks for both tall and short proportions.'};
 if(text&&!applied.length&&/^(?:(?:a|my|the|an|default|neutral|human|agent|character|person|body|please)\s*)+$/.test(text))return {ok:true,description:String(description),proportions,applied,method:'neutral human proportions'};
 if(text&&!applied.length)return {ok:false,reason:'I could not map that request to this human body yet. Try describing proportions such as curvy, athletic, tall, short, shoulders, waist, or hips. Your saved character has not changed.'};
 return {ok:true,description:String(description),proportions,applied,method:'local descriptive human proportion rules'};
}
