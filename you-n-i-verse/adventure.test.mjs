import test from 'node:test';
import assert from 'node:assert/strict';
import {BEACONS,startRun,offerChoices,activateBeacon,rechargeAtCamp,restoreRun} from './adventure-core.mjs';

test('all five locations begin dark and the starting path is source-backed',()=>{
  const run=startRun(100);assert.equal(run.path.gate,6);assert.equal(run.lit.length,0);
  assert.equal(BEACONS.length,5);assert.equal(offerChoices(run).length,6);
  assert.deepEqual(offerChoices(run).map(c=>c.cost),[1,1,2,2,3,3]);
  assert.equal(run.finishedAt,null);
});
test('choices perform canonical source transitions and accrue score/focus, without mutating prior run',async()=>{
  const old=startRun(100),{run,transition}=await activateBeacon(old,'movement',1,200);
  assert.equal(old.lit.length,0);assert.equal(old.path.gate,6);
  assert.equal(run.lit[0],'movement');assert.equal(run.path.gate,10);assert.equal(transition.toGate,10);
  assert.equal(run.focus,8);assert.equal(run.score,4);assert.equal(run.journal[0].dimension,'Movement');
  assert.equal(restoreRun(JSON.parse(JSON.stringify(run))).lit.length,1);
  await assert.rejects(activateBeacon(run,'movement',2),/already awake/);
});
test('camp costs points, allows recovery and prevents full-focus farming',async()=>{
  let run=startRun();run=(await activateBeacon(run,'movement',6)).run;
  assert.equal(run.focus,6);
  run=rechargeAtCamp(run,300);
  assert.equal(run.focus,9);assert.equal(run.score,14);assert.equal(run.recharges,1);
  assert.throws(()=>rechargeAtCamp(run),/already full/);
});
test('five distinct lights win and the saved run remains complete',async()=>{
  let run=startRun(100);
  for(const light of BEACONS)run=(await activateBeacon(run,light.id,1,200)).run;
  assert.equal(run.finishedAt,200);assert.equal(run.lit.length,5);
  assert.equal(run.path.history.length,5);assert.equal(run.journal.length,5);
  assert.equal(restoreRun(JSON.parse(JSON.stringify(run))).finishedAt,200);
  await assert.rejects(activateBeacon(run,'movement',1),/complete/);
});
test('unsupported choices, unrecognized places, impossible saves are rejected',async()=>{
  const run=startRun();
  await assert.rejects(activateBeacon(run,'void',1),/Unknown/);
  await assert.rejects(activateBeacon(run,'movement',7),/six/);
  assert.throws(()=>restoreRun({...run,lit:['movement','movement']}),/damaged/);
  assert.throws(()=>restoreRun({...run,lit:['movement']}),/damaged/);
});
