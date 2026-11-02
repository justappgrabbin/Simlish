import {HostRegistry} from './vendor/klein/host.mjs';
import {startRun, offerChoices} from './adventure-core.mjs';

// All routes point to existing, bundled Simlish experiences. No simulated execution claims.
export const PROGRAMS=Object.freeze([
  {id:'five-lights',label:'THE FIVE LIGHTS',type:'PLAY',description:'Walk a 3D world, restore five lights, and make source-backed gate/line choices.',entry:'adventure.html',color:'#ae94ff',caps:['launch','five-lights','game']},
  {id:'world',label:'LIVING WORLD',type:'EXPLORE',description:'Open the existing world simulation and inspect its episode history.',entry:'world.html',color:'#84e6d8',caps:['launch','world','simulation']},
  {id:'morph',label:'MORPH STUDIO',type:'CREATE',description:'Build a character from an image, save it, and hand it into the adventure.',entry:'index.html',color:'#ff9dc7',caps:['launch','morph','studio']},
  {id:'laboratory',label:'ENGINE LAB',type:'ENGINE',description:'Run the existing local Klein neural/world experiment and inspect its output.',entry:'laboratory.html',color:'#e9bd75',caps:['launch','laboratory','experiment']},
  {id:'rig',label:'3D RIG LAB',type:'CREATE',description:'Explore the existing rig and character tools.',entry:'rig.html',color:'#96bdff',caps:['launch','rig','character']}
]);

export function createGamingComputerHost(){
  const host=new HostRegistry();
  for(const item of PROGRAMS){
    host.register({
      id:'simlish.'+item.id,priority:item.id==='five-lights'?1:20,
      capabilities:['local','program',...item.caps],
      execute:async input=>{
        if(input?.action!=='launch'&&input?.action!=='inspect')throw new Error('Unsupported program request');
        return Object.freeze({id:item.id,entry:item.entry,action:input.action,mode:'bundled-android-assets'});
      }
    });
  }
  host.register({
    id:'simlish.source-backed-game-preview',priority:5,
    capabilities:['local','game','state','inspect'],
    execute:async input=>{
      if(input?.action!=='inspect')throw new Error('Game preview is read-only');
      const run=startRun(0);
      const choices=offerChoices(run);
      return Object.freeze({
        gate:run.path.gate,focus:run.focus,score:run.score,
        beacons:5,choiceCount:choices.length,
        lines:choices.map(c=>({line:c.id,cost:c.cost,points:c.points}))
      });
    }
  });
  return host;
}

export function resolveProgram(host,id){
  const p=PROGRAMS.find(p=>p.id===id);
  if(!p)throw new Error('Unknown Simlish program');
  return host.resolve(['local','program',...p.caps]);
}
