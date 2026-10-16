import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';import vm from 'node:vm';
import {calculateBirthChart,compareCharts,birthToUtc,substructure,derive,definedChannels,aspects,channelEmerges,ascendant,houseOf,CHANNELS,CENTER_OF_GATE,CENTERS,dms} from './chart-engine.mjs';
import {calculateGeonatalChart,humanDesignResolution} from '../vendor/isohuman/GeonatalCalculator.browser.mjs';
import {sha256} from '../sha256.mjs';
const here=p=>new URL(p,import.meta.url),read=p=>readFileSync(here(p),'utf8');
const ATTR=JSON.parse(read('./primitive-attributes.json')).attributes;
const F=JSON.parse(read('./reference/free-human-design-fixture.json'));
test('the vendored donor files are byte-identical to her SynthAi originals (hashes recorded)',()=>{const h=p=>sha256(new Uint8Array(readFileSync(here(p))));
 assert.equal(h('../vendor/isohuman/GeonatalCalculator.browser.mjs'),'ada0f6a8f36c66a7baf44f73a5131174e4c48bec4e395ecc6fc0ee38f34e3922');assert.equal(h('../vendor/isohuman/astronomy-engine.mjs'),'068f1445ed0c636c94818fe6d20d7d125120e605e0bab9fc4675c3d531be5ad7')});
test('local birth time to UTC through IANA zones (daylight saving) and explicit offsets',()=>{
 assert.equal(birthToUtc({date:'1990-06-15',time:'14:30',timeZone:'America/Los_Angeles'}).toISOString(),'1990-06-15T21:30:00.000Z');
 assert.equal(birthToUtc({date:'1990-12-15',time:'14:30',timeZone:'America/Los_Angeles'}).toISOString(),'1990-12-15T22:30:00.000Z');
 assert.equal(birthToUtc({date:'2000-01-01',time:'00:00',timeZone:'Asia/Kolkata'}).toISOString(),'1999-12-31T18:30:00.000Z');
 assert.equal(birthToUtc({date:'2000-01-01',time:'12:00',utcOffsetMinutes:-300}).toISOString(),'2000-01-01T17:00:00.000Z');
 assert.throws(()=>birthToUtc({date:'2000-1-1',time:'12:00'}),/timeZone/);assert.throws(()=>birthToUtc({date:'bad',time:'12:00',utcOffsetMinutes:0}),/YYYY-MM-DD/)});
test('the wheel her donor uses: gate 41 starts at 2°00\' Aquarius (302°), gate 25 at 28°15\' Pisces (358.25°), 64 gates x 6 lines, 5 bases per tone',()=>{
 assert.equal(humanDesignResolution(302.0001).gate,41);assert.equal(humanDesignResolution(301.9999).gate,60);assert.equal(humanDesignResolution(358.2501).gate,25);assert.equal(humanDesignResolution(358.2499).gate,36);
 const seen=new Set();for(let g=0;g<64;g++)seen.add(humanDesignResolution(302+g*5.625+0.001).gate);assert.equal(seen.size,64);
 const r=humanDesignResolution(302+5.625*3+0.9375*2+0.15625*3+(0.15625/6)*4+(0.15625/30)*4+1e-9);assert.deepEqual(r,{gate:49,line:3,color:4,tone:5,base:5})});
test('independent check: positions, ascendant, gates and lines against Free Human Design (MIT, VSOP87 via astronomia) on 48 random births, 1248 placements',()=>{
 const nm={sun:'Sun',earth:'Earth',moon:'Moon',north_node:'North Node',south_node:'South Node',mercury:'Mercury',venus:'Venus',mars:'Mars',jupiter:'Jupiter',saturn:'Saturn',uranus:'Uranus',neptune:'Neptune',pluto:'Pluto'};
 const near=(lon,w,size)=>{const r=(((lon-302)%360)+360)%360%size;return Math.min(r,size-r)<w};const dl=(a,b)=>{const d=Math.abs(a-b);return d>180?360-d:d};
 const maxd={};let n=0,gateDiffs=0,lineDiffsOutsideWindow=0,compared=0;
 for(const c of F.charts){const me=calculateBirthChart({date:c.birthdate,time:c.birthtime,utcOffsetMinutes:0,latitude:c.lat,longitude:c.lng});
  assert.ok(dl(me.astrology.ascendant.longitude,c.ascendant)<0.02,'ascendant '+c.utc);let skip=false;
  for(const [o,arr] of [['personality',c.personality],['design',c.design]])for(const r of arr){const m=me.activations.find(a=>a.orientation===o&&a.body===nm[r.body]);n++;const node=/node/.test(r.body);
   maxd[nm[r.body]]=Math.max(maxd[nm[r.body]]||0,dl(m.longitude,r.longitude));
   if(m.gate!==r.gate)gateDiffs++;
   if(m.line!==r.line&&!near(r.longitude,0.0517,0.9375)&&!node)lineDiffsOutsideWindow++;
   if(near(r.longitude,node?0.32:0.06,5.625)||(r.body==='sun'&&near(r.longitude,0.06,0.9375)))skip=true}
  if(skip)continue;compared++;
  assert.equal(me.hd.type,c.type,'type '+c.utc);assert.equal(me.hd.profile,c.profile,'profile '+c.utc);assert.equal(me.hd.definition,c.definition,'definition '+c.utc);
  assert.ok(c.authority.startsWith(me.hd.authority.split(' ')[0]),'authority '+c.utc+' '+me.hd.authority+' vs '+c.authority);
  assert.deepEqual(me.hd.channels.map(x=>x.join('-')).sort(),c.definedChannels.slice().sort(),'channels '+c.utc);
  assert.deepEqual(me.hd.definedCenters.map(x=>x.toLowerCase().replace(' ','')).sort(),c.definedCenters.slice().sort(),'centers '+c.utc)}
 assert.equal(n,1248);for(const [b,v] of Object.entries(maxd)){if(/Node/.test(b))assert.ok(v<0.35,b+' '+v);else assert.ok(v<0.01,b+' '+v)}
 assert.equal(lineDiffsOutsideWindow,0);assert.ok(compared>=12,'compared '+compared+' charts end to end');
 // gate differences are only nodes (her donor interpolates the true node between node events; the reference uses its own node model) and 2.5-minute boundary cases
 assert.ok(gateDiffs<=12,'gate differences '+gateDiffs)});
test('DEFECT recorded: her donor\'s ascendant is the descendant (180° off), so its house numbers are shifted by 6; the app corrects this and the original is untouched',()=>{
 const utc=new Date('1990-06-15T21:30:00Z'),loc={latitude:34.05,longitude:-118.24},d=calculateGeonatalChart(utc,loc),sun=d.placements.find(p=>p.frame==='tropical'&&p.orientation==='personality'&&p.planetary==='Sun');
 const asc=ascendant(utc,loc);assert.ok(Math.abs(asc-195.2845)<0.01);assert.equal(sun.house,((houseOf(sun.longitude,asc)+5)%12)+1);
 const mine=calculateBirthChart({date:'1990-06-15',time:'14:30',timeZone:'America/Los_Angeles',latitude:34.05,longitude:-118.24});assert.equal(mine.activations.find(a=>a.orientation==='personality'&&a.body==='Sun').house,houseOf(sun.longitude,asc))});
test('standard tables: 36 channels, every gate in exactly one center, her Black Book channels are all in the table; the one center disagreement in her data (gate 22) is recorded',()=>{
 assert.equal(CHANNELS.length,36);assert.equal(new Set(CHANNELS.map(c=>c.join('-'))).size,36);assert.equal(Object.keys(CENTER_OF_GATE).length,64);assert.deepEqual(Object.values(CENTER_OF_GATE).filter((v,i,a)=>a.indexOf(v)===i).sort(),CENTERS.slice().sort());
 const std=new Set(CHANNELS.map(c=>c.join('-')));for(let g=1;g<=64;g++){const ch=ATTR['gate.'+g].blackBook.channel;if(ch){const [x,y]=ch.split('-').map(Number);assert.ok(std.has(Math.min(x,y)+'-'+Math.max(x,y)),ch)}}
 const differ=[];for(let g=1;g<=64;g++)if(ATTR['gate.'+g].centerSG!==CENTER_OF_GATE[g])differ.push([g,ATTR['gate.'+g].centerSG,CENTER_OF_GATE[g],ATTR['gate.'+g].blackBook.center.split('.')[0]]);
 assert.deepEqual(differ,[[22,'Throat','Solar Plexus','Solar Plexus']]); // her centerSG field says Throat; her Black Book text and the standard say Solar Plexus
 for(const [a,b] of CHANNELS)for(const g of [a,b])assert.ok(CENTER_OF_GATE[g])});
test('type / authority / definition rules on constructed gate sets',()=>{
 assert.equal(derive([]).type,'Reflector');assert.equal(derive([34,20]).type,'Manifesting Generator');assert.equal(derive([5,15]).type,'Generator');assert.equal(derive([21,45]).type,'Manifestor');assert.equal(derive([21,45]).authority,'Ego Manifested');
 assert.equal(derive([47,64]).type,'Projector');assert.equal(derive([6,59]).authority,'Emotional');assert.equal(derive([57,10]).authority,'Splenic');assert.equal(derive([1,8]).definition,'single');assert.equal(derive([5,15,21,45]).definition,'split');assert.deepEqual(definedChannels([1,8,2]).map(c=>c.join('-')),['1-8'])});
test('aspects use conventional angles and orbs',()=>{const a=aspects([{body:'Sun',longitude:10},{body:'Moon',longitude:100},{body:'Mars',longitude:191},{body:'Earth',longitude:190}]);
 assert.deepEqual(a.map(x=>[x.a,x.b,x.aspect]).sort(),[['Moon','Mars','square'],['Sun','Mars','opposition'],['Sun','Moon','square']].sort())});
test('her field_core.js channelEmerges (original text run in a vm) equals the port at the gate/line and color/tone/base levels',()=>{
 const ctx={module:{exports:{}},console:{log(){}},require:()=>({}),exports:{}};vm.runInNewContext(read('./reference/her/field_core.js.txt').replace(/if \(require\.main === module\)[\s\S]*$/,''),ctx);const orig=ctx.module.exports.channelEmerges;assert.equal(typeof orig,'function');
 let s=7;const r=m=>(s=(s*1103515245+12345)>>>0)%m;let hits=0;for(let i=0;i<3000;i++){const A={gate:1+r(64),line:1+r(6),color:1+r(6),tone:1+r(6),base:1+r(5)},B=r(3)?{...A,line:1+r(6),gate:r(2)?A.gate:1+r(64)}:{gate:1+r(64),line:1+r(6),color:1+r(6),tone:1+r(6),base:1+r(5)};
  const o=orig(A,B),m=channelEmerges(A,B);assert.equal(m.gateLine,o.gateLine);assert.equal(m.colorToneBase,o.colorToneBase);assert.equal(m.any,o.any);if(o.any)hits++}assert.ok(hits>50)});
test('two-person comparison: a chart with itself is all companionship; relationship kinds follow the stated definitions',()=>{
 const a=calculateBirthChart({name:'A',date:'1990-06-15',time:'14:30',timeZone:'America/Los_Angeles',latitude:34.05,longitude:-118.24}),b=calculateBirthChart({name:'B',date:'1984-02-03',time:'03:15',timeZone:'Europe/London',latitude:51.5,longitude:-0.12});
 const same=compareCharts(a,a);assert.equal(same.relationship.filter(r=>r.kind==='companionship').length,a.hd.channels.length);assert.ok(same.relationship.every(r=>r.kind==='companionship'));
 const ab=compareCharts(a,b),ba=compareCharts(b,a);assert.equal(ab.relationship.length,ba.relationship.length);for(const r of ab.relationship)assert.ok(['companionship','electromagnetic','dominance','compromise'].includes(r.kind));assert.ok(ab.labels.relationship.includes('confirm'));assert.equal(ab.centers.length,9)});
test('a full chart carries substructure ids that exist as primitives, and nothing from the sources layer',()=>{
 const P=JSON.parse(read('./primitive-functions.json')).primitives,ids=new Set(P.map(p=>p.id)),c=calculateBirthChart({date:'1990-06-15',time:'14:30',timeZone:'America/Los_Angeles',latitude:34.05,longitude:-118.24});
 assert.equal(c.activations.length,26);for(const a of c.activations){for(const k of ['gate','line','color','tone','base'])assert.ok(ids.has(a.ids[k]),a.ids[k]);assert.ok(ids.has(a.ids.planet),a.ids.planet)}
 assert.equal(c.astrology.natal.length,13);assert.ok(c.methods.derivation.includes('confirm'));assert.equal(dms(84.5).text,'24°30\'00" Gemini');assert.ok(!JSON.stringify(c).includes('pdfPage'))});
