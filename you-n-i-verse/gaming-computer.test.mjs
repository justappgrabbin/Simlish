import test from 'node:test';import assert from 'node:assert/strict';
import {PROGRAMS,createGamingComputerHost,resolveProgram} from './gaming-computer-core.mjs';
test('each bundled program has a unique resolver route',async()=>{
 const host=createGamingComputerHost();
 assert.equal(PROGRAMS.length,5);
 for(const p of PROGRAMS){
   const faculty=resolveProgram(host,p.id);
   assert.equal(faculty.id,'simlish.'+p.id);
   const run=await host.execute(['local','program',...p.caps],{action:'launch'});
   assert.equal(run.facultyId,faculty.id);
   assert.equal(run.result.entry,p.entry);
   assert.match(p.entry,/^[a-z-]+\.html$/);
 }
 assert.equal(host.replay().length,5);
});
test('real source-backed game state available through local resolver',async()=>{
 const host=createGamingComputerHost();
 const result=await host.execute(['local','game','state','inspect'],{action:'inspect'});
 assert.equal(result.facultyId,'simlish.source-backed-game-preview');
 assert.ok(Number.isInteger(result.result.gate));
 assert.equal(result.result.choiceCount,6);
 assert.equal(result.result.beacons,5);
 assert.equal(result.result.focus,9);
});
test('unknown program and incompatible execution are refused',async()=>{
 const host=createGamingComputerHost();
 assert.throws(()=>resolveProgram(host,'missing'),/Unknown/);
 await assert.rejects(()=>host.execute(['local','program','five-lights'],{action:'delete'}),/Unsupported/);
});
