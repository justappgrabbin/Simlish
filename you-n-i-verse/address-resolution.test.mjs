import test from 'node:test';
import assert from 'node:assert/strict';
import './vendor/kernel.js';
import {ADDRESS_ORDER,resolveAddress} from './address-resolution.mjs';
const kernel=globalThis.AgentomagotchiKernel;
const coords={planetary:'Sun',zodiac:'Virgo',gate:6,line:4,color:4,tone:4,base:2,degree:25,minute:59,second:31,arcSecond:92,house:5};
test('canonical address order and dimension binding',()=>{
 assert.deepEqual(ADDRESS_ORDER,['planetary','dimension','zodiac','gate','line','color','tone','base','degree','minute','second','arcSecond','house']);
 const r=resolveAddress({kernel,leadingDimension:'Being',coordinates:coords,evidence:{source:'test fixture'}});
 assert.equal(r.ok,true);assert.deepEqual(Object.keys(r.address),ADDRESS_ORDER);
 assert.equal(r.address.dimension,'Being');assert.equal(r.address.arcSecond,92);
});
test('legacy arc maps without losing precision',()=>{
 const {arcSecond,...rest}=coords;
 const r=resolveAddress({kernel,leadingDimension:'Design',coordinates:{...rest,arc:92},evidence:{source:'fixture'}});
 assert.equal(r.address.arcSecond,92);
});
test('dimension conflict is not silently rewritten',()=>{
 const r=resolveAddress({kernel,leadingDimension:'Design',coordinates:{...coords,dimension:'Space'},evidence:{source:'fixture'}});
 assert.equal(r.ok,false);
});
