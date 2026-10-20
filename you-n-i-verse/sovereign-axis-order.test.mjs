import test from 'node:test';import assert from 'node:assert/strict';
import {survey,sovereignOrders,sovereignShapes} from './sovereign-axis-order.mjs';
import {survey as capSurvey,HU} from './hatcher-scales-probe.mjs';
import {complement} from './state-space.mjs';import {facesForHexagram} from './axis-order.mjs';
test('Sovereign Gua never equal the 12 caps',()=>{const s=capSurvey();assert.equal(s.sovereignEqualsCaps,0);assert.equal(s.maxSovereignAmongCaps,6)});
test('max-6 + Qian cap + linear ring = 240 setups, centre line 3 or 4 only',()=>{const s=survey();assert.equal(s.maxSovereignCaps,6);assert.equal(s.atMax,1920);assert.equal(s.withQianCap,960);assert.equal(s.withLinearRing,240);assert.deepEqual(s.centreLines,{3:120,4:120})});
test('in both candidates the Sovereign Gua show only 2 shapes, 6 + 6',()=>{for(const {order} of sovereignOrders()){const sh=sovereignShapes(order);assert.equal(sh.filter(x=>x==='333335').length,6);assert.equal(sh.filter(x=>x==='222444').length,6)}});
test('16 Hu Gua; Pang Tong opposite = opposite faces',()=>{assert.equal(HU.length,16);const {order}=sovereignOrders()[0];for(let c=0;c<64;c++){const a=facesForHexagram(c,order),b=facesForHexagram(complement(c),order);assert.ok(a.every(f=>!b.includes(f)))}});
