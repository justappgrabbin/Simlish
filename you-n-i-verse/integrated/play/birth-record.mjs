// NEW, derived — please confirm: maps integrated/chart-engine.mjs output into HER BirthDataStore record shape
// (vendor/youniverse-games/birth/birthDataStore.js, sample 'bd-test-001'), so her games read the user's real chart
// instead of her test traveler. Field names/values are hers where her code reads them; nothing is invented:
//  - strategy / signature / notSelfTheme / incarnationCross / variables / channel names: our chart engine does not compute
//    them and her game code never reads them -> null (not filled).
//  - iching.activeHexagram/activeLine = the user's CURRENT adaptive address (starts at Personality Sun, then moves by their own
//    one-line transitions) — her sample comments "Sun/Earth personality".
import {gateToFuXiDecimal} from '../../vendor/kingwen.mjs';
const CENTER_KEY={Head:'head',Ajna:'ajna',Throat:'throat',G:'g',Heart:'heart',Sacral:'sacral','Solar Plexus':'solarPlexus',Spleen:'spleen',Root:'root'};
const DEF={none:'No Definition',single:'Single Definition',split:'Split Definition','triple split':'Triple Split Definition','quadruple split':'Quadruple Split Definition'};
import {CENTER_OF_GATE} from '../chart-engine.mjs';
export function utcOffset(chart){const [y,m,d]=chart.input.date.split('-').map(Number),[hh,mm]=chart.input.time.split(':').map(Number);
 const mins=Math.round((Date.UTC(y,m-1,d,hh,mm)-new Date(chart.utc).getTime())/60000),s=mins<0?'-':'+',a=Math.abs(mins);return `${s}${String(Math.floor(a/60)).padStart(2,'0')}:${String(a%60).padStart(2,'0')}`}
export function chartToBirthRecord(chart,{birthDataId,address}={}){const hd=chart.hd,def=new Set(hd.definedCenters);
 const id=birthDataId||('bd-'+chart.utc.replace(/[^0-9]/g,'').slice(0,12)+'-'+Math.round((chart.input.latitude+90)*100)+'-'+Math.round((chart.input.longitude+180)*100));
 const centers={};for(const [k,v] of Object.entries(CENTER_KEY))centers[v]={defined:def.has(k)};
 const gates=chart.activations.map(a=>({gate:a.gate,line:a.line,center:CENTER_KEY[CENTER_OF_GATE[a.gate]],planet:a.body,designation:a.orientation}));
 const activeGateStates={};for(const g of gates)if(!(g.gate in activeGateStates))activeGateStates[g.gate]=g.line;
 const sun=chart.activations.find(a=>a.orientation==='personality'&&a.body==='Sun'),at=address||{gate:sun.gate,line:sun.line};
 return {birthDataId:id,name:chart.name||'Traveler',
  birth:{date:chart.input.date,time:chart.input.time,timezone:utcOffset(chart),place:{city:null,country:null,lat:chart.input.latitude,lon:chart.input.longitude},longitude:chart.input.longitude},
  humanDesign:{type:hd.type,authority:hd.authority,profile:hd.profile,definition:DEF[hd.definition]??hd.definition,strategy:null,signature:null,notSelfTheme:null,incarnationCross:null,centers,
   channels:hd.channels.map(([a,b])=>({id:`${a}-${b}`,gates:[a,b],centers:[CENTER_KEY[CENTER_OF_GATE[a]],CENTER_KEY[CENTER_OF_GATE[b]]],name:`${a}-${b}`})),gates,variables:null,rules:hd.rules},
  iching:{activeHexagram:at.gate,activeLine:at.line,activeGateStates,fuXiDecimal:gateToFuXiDecimal(at.gate)},
  provenance:'NEW, derived — please confirm: chart-engine -> her BirthDataStore shape'}}
