// Runs ONLY when SYNTHIA_STATE_SPACE_ROOT points to the verified live source checkout.
import test from 'node:test';import assert from 'node:assert/strict';
import {bootUnifiedSimlishStateSpace} from './unified-swarm-entry.mjs';
const canonical={planetary:1,dimension:'Movement',gate:3,line:1,color:1,tone:1,base:1,
  degree:1,minute:0,second:0,arcAxis:{arcUnit:1,axis:'Vertical'},zodiac:1,house:1};
test('one Genotype authority, real Simlish phenotype executor, and additive sourced swarm', {
 skip:!process.env.SYNTHIA_STATE_SPACE_ROOT
}, async()=>{
 const {assembly,world,swarm}=await bootUnifiedSimlishStateSpace({authorize:async({op})=>op==='simlish.phenotype.morph'});
 assert.equal(typeof assembly.structure,'function');assert.deepEqual(swarm.primitives,['tool-factory','neural-net-maker','model-maker','foundry']);
 swarm.add({id:'thought',label:'Meaning',x:-.4,y:0});swarm.add({id:'chairNode',label:'Chair',x:.3,y:0});
 swarm.relate({from:'thought',to:'chairNode',predicate:'motivates-change',provenance:'integration-test-explicit'});
 assert.equal(swarm.cloud().edges.length,1);
 const previous=world.object('chair').morphState;
 const result=await swarm.morphSimlish({intent:'transform chair using connected world',genotypeAddress:canonical,
  genotypeSource:'owner-explicit',objectId:'chair',operation:'transform',subject:{text:'chair changes'}});
 assert.equal(result.status,'executed-and-observed',JSON.stringify(result));
 assert.equal(world.object('chair').morphState,(previous+1)%3);
 const reopened=swarm.snapshot();assert.equal(reopened.swarm.nodes.length,2);assert.equal(reopened.receipts.length,1);
});
