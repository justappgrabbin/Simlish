import test from 'node:test';
import assert from 'node:assert/strict';
import {connectGenerativeStateSpace} from './generative-state-space-plug.mjs';
import './vendor/kernel.js';
import './vendor/world-bridge.js';
import './vendor/mesh-assembly.js';
import './vendor/sim-core.js';

const makeWorld=()=>new globalThis.AgentomagotchiSim.AgentomagotchiWorld({seed:777});
const fullAddress=()=>({planetary:1,dimension:'Movement',gate:3,line:1,color:1,tone:1,base:1,
  degree:1,minute:0,second:0,arcAxis:{arcUnit:1,axis:'Vertical'},zodiac:1,house:1});
function sourceHarness(authorize=async()=>true) {
  const observations=[];
  const assembly={
    structure:(subject,context)=>({kind:'structural-state',subject,context}),
    createSwarmRuntime:({capabilities})=>({
      capabilityNames:()=>capabilities.map(x=>x.name),
      async execute(request) {
        observations.push(request);
        if (!request.address || Object.keys(request.address).length!==13)
          throw new TypeError('missing canonical address field');
        const cap=capabilities.find(x=>x.name===request.steps[0].op);
        if(await authorize({op:cap.name})!==true) return {ok:false,execution:[{error:'OWNER_AUTHORIZATION_REQUIRED'}]};
        return {ok:true,output:cap.execute(request.steps[0].input)};
      }
    })
  };
  return {assembly,observations};
}
test('Simlish receives semantic context and expresses the phenotype using its own world',async()=>{
  const world=makeWorld(),{assembly,observations}=sourceHarness();
  const plug=connectGenerativeStateSpace({assembly,world});
  const old=world.object('chair').morphState;
  const record=await plug.morph({intent:'morph the chair',genotypeAddress:fullAddress(),
    genotypeSource:'owner-explicit',objectId:'chair',subject:{text:'chair transforms'}});
  assert.equal(record.genotype.address.gate,3);
  assert.equal(record.phenotype.status,'executed-and-observed');
  assert.notEqual(world.object('chair').morphState,old);
  assert.equal(record.phenotype.expression.worldEvent.type,'MORPH');
  assert.equal(observations[0].steps[0].op,'simlish.phenotype.morph');
  assert.equal(world.agent('agent-a').activeState.addresses.length,5);
  assert.equal(plug.state().history.length,1);
});
test('No fabricated genotype from the demo seed or destructive world operations',async()=>{
  const world=makeWorld(),{assembly}=sourceHarness(),plug=connectGenerativeStateSpace({assembly,world});
  const before=world.worldLedger.length;
  await assert.rejects(()=>plug.morph({intent:'remove',genotypeAddress:fullAddress(),
    genotypeSource:'owner-explicit',objectId:'chair',operation:'remove'}),/non-destructive/);
  await assert.rejects(()=>plug.morph({intent:'transform',genotypeAddress:fullAddress(),
    genotypeSource:'world-hash',objectId:'chair'}),/provenance/);
  assert.equal(world.worldLedger.length,before);
});
test('No world mutation when owner policy blocks capability',async()=>{
  const world=makeWorld(),{assembly}=sourceHarness(async()=>false);
  const plug=connectGenerativeStateSpace({assembly,world});
  const old=world.object('chair').morphState;
  const record=await plug.morph({intent:'transform',genotypeAddress:fullAddress(),
    genotypeSource:'owner-explicit',objectId:'chair'});
  assert.equal(record.phenotype.status,'blocked');
  assert.equal(world.object('chair').morphState,old);
});
test('Real Generative State Space to Simlish through shared swarm execution', {
  skip:!process.env.SYNTHIA_STATE_SPACE_ROOT
},async()=>{
  const {bootSimlishStateSpace}=await import('./generative-state-space-runtime.mjs');
  const {assembly,world,plug}=await bootSimlishStateSpace({
    authorize:async({op})=>op==='simlish.phenotype.morph'
  });
  assert.ok(assembly.generativeMorphology);
  assert.ok(plug.state().capabilities.includes('simlish.phenotype.morph'));
  const prior=world.object('chair').morphState;
  const record=await plug.morph({intent:'transform a chair through the Simlish world',
    genotypeAddress:fullAddress(),genotypeSource:'owner-explicit',objectId:'chair',
    subject:{text:'a chair changes through a witnessed transformation'},context:{view:'being'}});
  assert.equal(record.phenotype.status,'executed-and-observed',JSON.stringify(record.swarmReceipt));
  assert.equal(world.object('chair').morphState,(prior+1)%3);
  assert.ok(record.swarmReceipt.kleinEvaluationId);
});
