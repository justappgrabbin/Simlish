import {autoLingAutomaton, autoNovelAutomaton} from './vendor/klein/index.mjs';
import {LocalGraphModel} from './model.mjs';
export async function runSlice(){
 const K=globalThis.AgentomagotchiKernel,S=globalThis.AgentomagotchiSim;if(!K||!S)throw Error('Load supplied world scripts first');
 const ling=autoLingAutomaton();await ling.call({operation:'learn',rule:{id:'scene-acts.v1',relation:'ACTS_ON',pattern:['$subject','$verb','$object'],template:'$subject $verb $object'}});
 const parsed=await ling.call({operation:'recognize',text:'agent transforms chair'});const structure=parsed.matches[0].structure;
 const novel=autoNovelAutomaton();await novel.call({operation:'register',domain:{id:'scene',primitives:[{id:'actor'},{id:'effect'}],combinators:[{id:'scene.sequence.v1',type:'sequence',inputs:['actor','effect'],output:'event',apply:(a,b)=>({actor:a,effect:b})}]}});
 const story=await novel.call({operation:'generate',spec:{domain:'scene',seeds:[{type:'actor',value:structure[0]},{type:'effect',value:structure[1]}]}});
 const edges=K.CHANNELS.map(c=>c.id.split('-').map(v=>Number(v)-1));const model=new LocalGraphModel();
 // Authored fixtures teach this narrowly scoped classifier, not general intelligence.
 const examples=[0,1,2].flatMap(label=>Array.from({length:8},(_,i)=>{const bits=Array(64).fill(label===2?1:0);if(label===1){bits[0]=1;bits[7]=1;}bits[20+i]=1;return {features:model.features(bits,edges),label}}));
 const loss=()=>examples.reduce((s,e)=>s-Math.log(model.predict(e.features)[e.label]),0)/examples.length;const beforeLoss=loss();model.train(examples,600);const afterLoss=loss();
 const heldout=[0,1,2].map(label=>{const bits=Array(64).fill(label===2?1:0);if(label===1){bits[0]=1;bits[7]=1;}bits[40]=1;const probs=model.predict(model.features(bits,edges));return {label,predicted:probs.indexOf(Math.max(...probs))}});
 const world=new S.AgentomagotchiWorld({seed:777}),agent=world.agent('agent-a');world.select('chair');for(let i=0;i<6;i++)world.morphSelected('transform');
 const capability=[...agent.assemblies.capabilities.values()].find(c=>c.composition.includes('alter:transform'));const legal=world.bridge.legalPossibilities(agent,world,capability);
 const bits=Array(64).fill(0);bits[0]=bits[7]=1;const scores=model.predict(model.features(bits,edges));const allowed=structure[1]==='transforms'&&scores[1]>scores[0]&&scores[1]>scores[2]&&legal.some(p=>p.targetId==='chair'&&p.operation==='transform-object');
 const before=world.object('chair').morphState;const executed=allowed&&world.executeBestCapability(agent,true);const after=world.object('chair').morphState;
 const bundle=S.completeAddressBundleFromKey(agent,'morph',null,0);const event={id:'klein-neural-scene-1',grammarRule:'scene-acts.v1',structure,storyLineage:story.lineage,projections:bundle.addresses.map(a=>{const full=a.toObject();return Object.fromEntries(K.ADDRESS_FIELDS.map(key=>[key,full[key]]))}),scores,executed,before,after,worldReceipt:world.worldLedger.filter(e=>e.type==='EMERGENT_EXECUTION').at(-1)};
 return {event,model:model.export(),evaluation:{beforeLoss,afterLoss,heldout,trainingExamples:examples.length,scope:'authored synthetic graph fixtures; not user-trained or visual quality evidence'}};
}
