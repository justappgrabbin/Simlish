import test from 'node:test';import assert from 'node:assert/strict';
import {StateSpace,DIMENSIONS,PRIMITIVE_DIMENSIONS,capacity,capacityTable,gateToMacro,macroToGate,nuclear,complement,reverseLines,exchangeTrigrams,flipLines,hammingNeighbors,bitsOf,codeOf,nodeId,parseNodeId,INVESTIGATING} from './state-space.mjs';
test('five dimensions, Space last and non-primitive',()=>{assert.deepEqual([...DIMENSIONS],['Movement','Evolution','Being','Design','Space']);assert.deepEqual([...PRIMITIVE_DIMENSIONS],['Movement','Evolution','Being','Design'])});
test('capacity: 64 macro nodes per level, 12 or six per split, five levels (both of Alexis\'s numbers kept)',()=>{
 assert.equal(capacity().positionsPerLevel,768);assert.equal(capacity().positionsPerDimensionAtFullDepth,String(768n**5n));
 assert.equal(capacity({split:6}).positionsPerDimensionAtFullDepth,String(384n**5n));
 assert.equal(capacityTable().length,2);assert.ok(BigInt(capacity({split:6}).positionsAcrossFiveDimensions)>10n**12n,'trillions');
 assert.throws(()=>new StateSpace({split:7}),/12 or 6/);assert.throws(()=>new StateSpace({levels:0}));
});
test('sparse and on demand: nothing exists until touched; children are generated lazily',()=>{
 const s=new StateSpace();assert.equal(s.size,0);
 const n=s.ensure('Being',[{macro:5,sub:3}]);assert.equal(s.size,1);assert.equal(s.ensure('Being',[{macro:5,sub:3}]),n);assert.equal(s.size,1);
 let c=0;for(const ch of s.children(n.id)){c++;if(c>=100)break}assert.equal(c,100);assert.equal(s.size,1,'enumerating children does not create nodes');
 assert.equal([...s.children(n.id,{macros:[0]})].length,12);
 assert.equal([...new StateSpace({split:6}).children(nodeId('Being',[{macro:1,sub:1}]),{macros:[0]})].length,6);
});
test('node ids round-trip; paths are validated; depth limited by levels',()=>{
 const s=new StateSpace({levels:3}),p=[{macro:63,sub:11},{macro:0,sub:0},{macro:7,sub:5}];
 assert.deepEqual(parseNodeId(nodeId('Design',p)),{dimension:'Design',path:p});
 s.ensure('Design',p);assert.throws(()=>s.ensure('Design',[...p,{macro:1,sub:1}]),/depth/);
 assert.throws(()=>s.ensure('Design',[{macro:64,sub:0}]),/macro/);assert.throws(()=>s.ensure('Design',[{macro:1,sub:12}]),/sub/);assert.throws(()=>parseNodeId('9:1.1'));
 assert.equal(s.ancestors(nodeId('Design',p)).length,2);
});
test('Space is emergent: 64 slots reported, never written, read only through an injected rule (no invented default)',()=>{
 const s=new StateSpace();
 assert.throws(()=>s.ensure('Space',[{macro:1,sub:0}]),/emergent/);assert.throws(()=>s.setDimensionCode('Space',3),/read, not set/);
 const r=s.readSpace();assert.equal(r.status,'investigating');assert.equal(r.code,null);assert.equal(r.slots,64);assert.equal(r.derivedOnly,true);
 const e=new StateSpace({emergence:four=>(four.Movement^four.Evolution^four.Being^four.Design)&63});
 assert.equal(e.readSpace().status,'investigating','needs all four primitive states');
 PRIMITIVE_DIMENSIONS.forEach((d,i)=>e.setDimensionCode(d,i+1));
 assert.equal(e.readSpace().status,'derived');assert.equal(e.readSpace().code,4);assert.equal(e.codes().Space,4);
 assert.equal(e.maskHex().Space,'0000000000000000','Space slots are never directly active');
});
test('64-bit masks mark active level-1 slots per dimension',()=>{
 const s=new StateSpace(),a=s.ensure('Movement',[{macro:0,sub:0}]),b=s.ensure('Movement',[{macro:63,sub:2},{macro:9,sub:1}]);
 s.activate(a.id);s.activate(b.id);assert.equal(s.masks().Movement,(1n<<63n)|1n);assert.equal(s.masks().Being,0n);s.clearActive();assert.equal(s.masks().Movement,0n);
});
test('King Wen and Fu Xi are separate named maps and round-trip for all 64',()=>{
 const seen=new Set();for(let g=1;g<=64;g++){const m=gateToMacro(g);seen.add(m);assert.equal(macroToGate(m),g)}assert.equal(seen.size,64);
 assert.equal(gateToMacro(1),63);assert.equal(gateToMacro(2),0);assert.throws(()=>gateToMacro(65));assert.throws(()=>macroToGate(64));
});
test('Spec §36 operators: line flip, complement, reverse, trigram exchange, nuclear (lines 2,3,4,3,4,5)',()=>{
 assert.equal(bitsOf(17),'010001');assert.equal(codeOf('010001'),17);
 assert.equal(flipLines(17,[1]),16);assert.equal(flipLines(0,[6]),32);assert.throws(()=>flipLines(0,[7]));
 assert.equal(complement(17),46);assert.equal(reverseLines(17),0b100010);assert.equal(exchangeTrigrams(0b001010),0b010001);
 assert.equal(macroToGate(nuclear(gateToMacro(3))),23,'nuclear hexagram of gate 3 is gate 23');
 assert.equal(nuclear(63),63);assert.equal(nuclear(0),0);
 assert.equal(hammingNeighbors(0,1).length,6);assert.equal(hammingNeighbors(0,2).length,21);
});
test('siblings are Hamming neighbours under the same parent; center_binding stays null (no modulo-nine)',()=>{
 const s=new StateSpace(),n=s.ensure('Evolution',[{macro:10,sub:4},{macro:20,sub:2}]);
 const sib=s.siblings(n.id);assert.equal(sib.length,6);assert.ok(sib.every(x=>x.node.parent===n.parent&&x.node.sub===2));assert.equal(n.center_binding,null);
 assert.ok(INVESTIGATING.some(x=>/modulo-nine/.test(x)));
});
test('snapshot and restore reproduce nodes and codes; shape mismatch is refused',()=>{
 const a=new StateSpace();a.ensure('Being',[{macro:3,sub:1}]);a.setDimensionCode('Being',9);const b=new StateSpace();b.restore(a.snapshot());
 assert.equal(b.size,1);assert.equal(b.codes().Being,9);assert.throws(()=>new StateSpace({split:6}).restore(a.snapshot()),/differs/);
});
