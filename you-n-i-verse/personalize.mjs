// Personalize the app build for the uploader. Swarm pair: PSO genotype (checks, mechanics) -> GA phenotype (what is shown).
// Alexis's rule: a user picks their world's theme once; once set, it's set (a later different theme is kept out, not an error).
import {SwarmComponent,SwarmPair} from './swarm-nsnf.mjs';
import {missingPlaces} from './registration-swarm.mjs';
export const PERSONALIZATION_KEY='synthia.simlish.personalization.v1';
export const TARGETS=Object.freeze(['reality','game','2d']);
const slug=s=>String(s||'').trim().toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'');
const genotype=new SwarmComponent({id:'personalize',kind:'PSO',purpose:'Check the uploader\'s personalization before it is carried into a build',
 members:[{id:'uploader-name',check:i=>!!i.config.name},{id:'world-theme',check:i=>!!i.config.worldTheme},
  {id:'address-complete',check:i=>i.config.address?{ok:missingPlaces(i.config.address).missing.length===0,note:'missing '+missingPlaces(i.config.address).missing.join(', ')}:{ok:false,note:'no address yet'}},
  {id:'theme-set-once',check:i=>!i.themeConflict},{id:'target-known',check:i=>TARGETS.includes(i.config.target)},{id:'build-title',check:i=>!!i.config.name&&!!i.config.worldTheme}]});
const phenotypeComponent=new SwarmComponent({id:'personalize-view',kind:'GA',purpose:'Show the uploader\'s name and world on screen',members:[
 {id:'title',check:i=>!!i.visible.title},{id:'world-class',check:i=>typeof i.visible.worldClass==='string'},{id:'target',check:i=>!!i.visible.target},
 {id:'locked-flag',check:i=>typeof i.visible.themeLocked==='boolean'},{id:'status-matches',check:i=>i.visible.status===i.genotype.status},{id:'kept',check:i=>i.visible.kept===true}]});
const trait=(g,i)=>({title:(i.config.name||'Uploader')+(i.config.worldTheme?' · '+i.config.worldTheme:''),worldClass:i.config.worldTheme?'world-'+slug(i.config.worldTheme):'world-unset',
 target:i.config.target,themeLocked:!!i.config.worldTheme,themeNote:i.themeConflict?'Theme was already set to "'+i.config.worldTheme+'" and stays set.':null,status:g.status,kept:true});
export const personalizePair=()=>new SwarmPair({id:'personalize',genotype,phenotype:{component:phenotypeComponent,trait}});
const unit=personalizePair();
export function readPersonalization(storage){const raw=storage?.getItem(PERSONALIZATION_KEY);if(!raw)return null;try{return JSON.parse(raw)}catch{throw new Error('Unreadable '+PERSONALIZATION_KEY+'; original preserved')}}
// input: {name, worldTheme, address, target}. Returns {config, swarm}; saves when storage given. Never throws on uncertain input.
export function personalize(input={},storage=null){
 const old=(storage&&readPersonalization(storage))||{};
 const themeConflict=!!(old.worldTheme&&input.worldTheme&&String(input.worldTheme).trim()!==old.worldTheme);
 const config={name:(input.name??old.name??'')&&String(input.name??old.name).trim(),
  worldTheme:old.worldTheme||(input.worldTheme?String(input.worldTheme).trim():''),
  address:input.address??old.address??null,target:input.target??old.target??'reality',updatedAt:new Date().toISOString()};
 const swarm=unit.forward({config,themeConflict});
 const record={...config,status:swarm.status==='complete'?'personalized':'investigating',swarm:{genotype:{packs:swarm.genotype.packs,hexagram_code:swarm.genotype.hexagram_code,failed:swarm.genotype.failed},phenotype:swarm.phenotype.visible}};
 if(storage)storage.setItem(PERSONALIZATION_KEY,JSON.stringify(record));
 return record;
}
