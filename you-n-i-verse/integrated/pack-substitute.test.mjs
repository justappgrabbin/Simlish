import test from 'node:test';import assert from 'node:assert/strict';
import {packSubstitute,packDistance} from './pack-substitute.mjs';import {packsOf} from './bantu.mjs';import {hammingNeighbors} from '../state-space.mjs';
test('replaces exactly one pack, leaves the others',()=>{for(let c=0;c<64;c++)for(let p=0;p<3;p++)for(let b=0;b<4;b++){const n=packSubstitute(c,p,b);assert.equal(packsOf(n)[p],b);packsOf(c).forEach((x,i)=>{if(i!==p)assert.equal(packsOf(n)[i],x)})}});
test('identity when the bigram is unchanged; distance within the pack is at most 2',()=>{for(let c=0;c<64;c++){assert.equal(packSubstitute(c,1,packsOf(c)[1]),c);for(let b=0;b<4;b++)assert.ok(packDistance(c,packSubstitute(c,2,b))[2]<=2)}});
test('single-line change stays a Hamming-1 neighbour of the existing state space',()=>{const n=hammingNeighbors(0b101010,1).map(x=>x.code);assert.ok(n.includes(packSubstitute(0b101010,0,packsOf(0b101010)[0]^1)))});
