import test from 'node:test';
import assert from 'node:assert/strict';
import {matchGrid,intermediatePoints} from './frame-morph.mjs';
test('local appearance matching finds a translated patch, with intermediate geometry',()=>{
 const w=64,h=64,a=new Uint8ClampedArray(w*h*4),b=new Uint8ClampedArray(a.length);
 for(let y=20;y<44;y++)for(let x=20;x<44;x++){const v=(x*37+y*71)%256;for(let c=0;c<3;c++){a[(y*w+x)*4+c]=v;b[(y*w+x+4)*4+c]=v}}
 const m=matchGrid(a,b,w,h,{columns:4,rows:4,radius:6});const i=12;
 assert.equal(m.target[i].x-m.source[i].x,4);assert.equal(m.target[i].y-m.source[i].y,0);
 assert.equal(intermediatePoints(m,.5)[i].x,m.source[i].x+2);
 assert.deepEqual(intermediatePoints(m,0),m.source);assert.deepEqual(intermediatePoints(m,1),m.target);
 assert.equal(m.fallback,false);
});
