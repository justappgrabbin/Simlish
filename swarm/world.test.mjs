import test from 'node:test';
import assert from 'node:assert/strict';
import { SwarmWorld } from './world.mjs';
function world() {
 const w = new SwarmWorld();
 w.addParticle({id:'a',x:4,y:4,bit:0});
 w.addParticle({id:'b',x:5,y:4,bit:1});
 w.compose('pair',['a','b']); w.compose('organism',['pair'],'o_sequence');
 w.connect('a','b'); return w;
}
test('requested flavor changes trajectory while retaining identity and nested lineage',()=>{
 const a=world(),b=world();b.requestFlavor('buoyant');
 for(let i=0;i<120;i++){a.step();b.step();}
 assert.ok(a.particles.get('a').y>b.particles.get('a').y);
 assert.deepEqual(a.composites.get('organism'),b.composites.get('organism'));
 assert.equal(b.particles.get('a').bit,0);
});
test('opposing spring forces preserve momentum without external forces',()=>{
 const w=world(); w.requestFlavor({gravity:0,damping:0,stiffness:10,restitution:1});
 w.particles.get('b').x=5.2;w.step();
 assert.ok(Math.abs(w.particles.get('a').vx+w.particles.get('b').vx)<1e-12);
});
test('interaction ruptures bond with trace and leaves constituents intact',()=>{
 const w=world();w.particles.get('b').x=7;w.step();
 assert.equal(w.edges[0].active,false);assert.equal(w.particles.size,2);
 assert.equal(w.history.at(-1).operation,'bond-rupture');
 assert.equal(w.history.at(-1).provenance,w.edges[0].provenance);
});
test('task phase suspends embodiment and rejects physical actions',()=>{
 const w=world();w.setPhase('task');const before=w.snapshot();w.step();
 assert.deepEqual(w.snapshot(),before);assert.throws(()=>w.impulse('a',1,0));
});
test('invalid inputs do not silently default',()=>{
 const w=world();assert.throws(()=>w.requestFlavor('missing'));
 assert.throws(()=>w.requestFlavor({gravity:NaN,damping:0,stiffness:1,restitution:1}));
 assert.throws(()=>w.step(1));
});
