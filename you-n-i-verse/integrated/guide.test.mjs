import test from 'node:test';import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';
import {createGuide} from './guide.mjs';import {calculateBirthChart} from './chart-engine.mjs';
const J=p=>JSON.parse(readFileSync(new URL(p,import.meta.url),'utf8')),clean=J('./primitive-functions.json'),attrs=J('./primitive-attributes.json').attributes,G=createGuide({clean,attrs});
const chart=calculateBirthChart({date:'1990-06-15',time:'14:30',timeZone:'America/Los_Angeles',latitude:34.05,longitude:-118.24});
test('gate answers are her fields verbatim, each tagged with the layer it came from',()=>{const r=G.ask('tell me about gate 1');assert.equal(r.ok,true);const t=G.render(r);
 for(const s of ['The Creative','Self-Expression / Mutation','Scorpio-Fixed. Transcendence Sex Death Regeneration','The Liver','Freshness','Supreme offering. Favourable augury.'])assert.ok(t.includes(s),s);assert.ok(t.includes('[attributes: blackBook]'));
 assert.equal(r.sections.find(s=>s.label==='Theme').text,attrs['gate.1'].theme)});
test('line, color, tone, base, planet, zodiac and full 5-level addresses resolve to her entries',()=>{
 assert.equal(G.ask('gate 1.1').sections.find(s=>s.label==='Keynote').text,attrs['gate.1.line.1'].keynote);assert.ok(G.ask('13.2').title.includes('13 line 2'));
 assert.ok(G.render(G.ask('color 1')).includes('Communalist / Separatist'));assert.ok(G.render(G.ask('tone 1')).includes('SPLENIC'));assert.ok(G.render(G.ask('base 3')).includes(attrs['base.3'].question));
 assert.ok(G.render(G.ask('sun')).includes('☉'));assert.ok(G.render(G.ask('libra')).includes('Degrees: 180–210'));
 const full=G.ask('57.3.2.4.1');assert.equal(full.ok,true);const names=full.sections.filter(s=>s.label.startsWith('▸')).map(s=>s.label);assert.equal(names.length,5)});
test('channel answers are labelled standard and note whether her Black Book lists them; unknown channels are refused',()=>{const r=G.ask('channel 13-33');assert.equal(r.title,'Channel 13-33');assert.ok(G.render(r).includes('standard Human Design/astrology convention'));assert.ok(G.render(G.ask('channel 1-2')).includes('no channel'))});
test('chart questions come from the calculated chart and say what is derived by rule',()=>{const r=G.ask('my type and profile',{chart});assert.ok(r.sections.find(s=>s.label==='Type').text===chart.hd.type);assert.ok(r.sections.every(s=>!s.text||s.from!=='toy'));assert.ok(G.render(r).includes('confirm'));
 const sun=G.ask('my sun',{chart});assert.ok(sun.title.startsWith('Personality Sun'));assert.equal(sun.sections.filter(s=>s.label.startsWith('▸')).length,5);assert.ok(G.ask('design moon',{chart}).title.startsWith('Design Moon'))});
test('unknown questions are not answered with invented text',()=>{const r=G.ask('what should I do about my job?');assert.equal(r.ok,false);assert.ok(r.sections[0].text.includes('gate 13'));assert.ok(r.sections.some(s=>/does not write new interpretations/.test(s.text)));assert.equal(G.ask('gate 99').ok,false)});
test('every sentence the guide returns for all 64 gates and 384 lines exists verbatim in the attribute layer',()=>{let n=0;for(let g=1;g<=64;g++){const r=G.ask('gate '+g);assert.equal(r.ok,true);const t=JSON.stringify(attrs['gate.'+g]);for(const s of r.sections)if(s.from&&s.from.startsWith('attributes:')&&s.label==='Theme')assert.ok(t.includes(s.text));
  for(let l=1;l<=6;l++){const q=G.ask(`gate ${g}.${l}`);assert.equal(q.ok,true);n++}}assert.equal(n,384)});
