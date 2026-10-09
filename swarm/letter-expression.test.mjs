import test from 'node:test';
import assert from 'node:assert/strict';
import { SwarmWorld } from './world.mjs';
import { expressionMembers, letterState, resolveExpressionColor } from './letter-expression.mjs';
test('nested expressions retain ordered identity, color provenance and rupture visibility',()=>{
 const world=new SwarmWorld();
 for(const [id,x,glyph] of [['m',1,'M'],['e',2,'E'],['u',3,'U']]) world.addParticle({id,x,y:1,qualities:{glyph}});
 const edge=world.connect('m','e');world.compose('word',['m','e'],'o_sequence');world.compose('phrase',['word','u'],'o_sequence');
 assert.deepEqual(expressionMembers(world,'phrase').map(p=>p.id),['m','e','u']);
 assert.throws(()=>resolveExpressionColor(world,'phrase',{color:'#ffaa00'}),/requires/);
 resolveExpressionColor(world,'phrase',{color:'#ffaa00',address:'source-position',dimension:'Being',provenance:'test-resolver'});
 assert.equal(world.particles.get('m').expressionColor.color,'#ffaa00');
 world.particles.get('e').x=2.4;assert.equal(letterState(world,'m').state,'strained');
 edge.active=false;assert.equal(letterState(world,'m').state,'separated');
 assert.equal(world.particles.get('m').qualities.glyph,'M');
 assert.equal(world.snapshot().particles[0].expressionColor.provenance,'test-resolver');
});

test('binary and hexadecimal represent the same RGB channels without wrapping',async()=>{
 const {rgbToHex,binaryRgbToHex}=await import('./letter-expression.mjs');
 assert.equal(binaryRgbToHex('11111111','10101010','00000000'),'#ffaa00');
 assert.equal(rgbToHex(255,170,0),'#ffaa00');
 assert.throws(()=>rgbToHex(256,0,0),/0..255/);
 assert.throws(()=>binaryRgbToHex('101010','00000000','00000000'),/eight/);
});
