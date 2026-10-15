import test from 'node:test';import assert from 'node:assert/strict';
import {personalize,readPersonalization,PERSONALIZATION_KEY} from './personalize.mjs';
const mem=()=>({d:{},getItem(k){return this.d[k]??null},setItem(k,v){this.d[k]=v}});
test('personalization is saved and carries name, theme, target, address; incomplete is investigating',()=>{
 const s=mem(),p=personalize({name:'Alexis',worldTheme:'Dungeons and Dragons'},s);
 assert.equal(p.status,'investigating');assert.equal(readPersonalization(s).name,'Alexis');assert.equal(p.swarm.phenotype.worldClass,'world-dungeons-and-dragons');assert.equal(p.target,'reality');assert.equal(p.swarm.genotype.packs.length,3);
});
test('world theme is set once; a later different theme is kept out, not an error',()=>{
 const s=mem();personalize({name:'A',worldTheme:'Dungeons and Dragons'},s);
 const p=personalize({worldTheme:'Ocean'},s);assert.equal(p.worldTheme,'Dungeons and Dragons');assert.match(p.swarm.phenotype.themeNote,/stays set/);
 assert.equal(personalize({name:'B'},s).name,'B');assert.equal(readPersonalization(s).worldTheme,'Dungeons and Dragons');
});
test('unreadable stored personalization is preserved',()=>{const s=mem();s.setItem(PERSONALIZATION_KEY,'junk');assert.throws(()=>personalize({name:'x'},s),/Unreadable/);assert.equal(s.getItem(PERSONALIZATION_KEY),'junk')});
