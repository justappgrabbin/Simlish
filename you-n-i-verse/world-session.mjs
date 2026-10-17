// YOU-N-I-VERSE: connect existing Simlish World, SwarmWorld physics, and ATO/Klein formations.
// Preserves the source engines. Visual consumers must render these actual snapshots.
import { ExecutableFormation } from './formation-spine.mjs';
import { FunctionRegistry } from './vendor/klein/index.mjs';
import { SwarmWorld } from './swarm/world.mjs';

const ADDRESS = Object.freeze({mode:'macro',gate:17,line:1,color:1,tone:1,base:1});
const MODES = Object.freeze(['Play','Create','Mission']);
const OPS = Object.freeze(['transform','move','duplicate','remove']);
const clone = value => structuredClone(value);

function graph(mode) {
  const functions = mode === 'Create'
    ? ['world.apply','world.compose','world.capture']
    : mode === 'Mission'
      ? ['world.impulse','world.apply','world.capture']
      : ['world.apply','world.capture'];
  const nodes = [{id:'input'}, ...functions.map((_,i)=>({id:'step'+i})), {id:'output'}];
  const ids = nodes.map(n=>n.id);
  return {nodes,edges:functions.map((fn,i)=>({id:mode+'-'+i,from:ids[i],to:ids[i+1],functionId:fn}))};
}
function items(world,physics) {
  const positions = new Map(physics.particles);
  return world.objects.map(object => {
    const particle=positions.get(object.id);
    return {
      id:object.id,name:object.name,type:object.type,removed:!!object.removed,
      x:object.x,z:object.z,sx:object.sx,sy:object.sy,sz:object.sz,
      morphState:object.morphState,
      particle:particle ? {x:particle.x,y:particle.y,vx:particle.vx,vy:particle.vy} : null
    };
  });
}
export class WorldSession {
  constructor({seed=777}={}) {
    const Sim = globalThis.AgentomagotchiSim;
    if(!Sim?.AgentomagotchiWorld)throw new Error('The existing Simlish world engine is not loaded');
    this.world = new Sim.AgentomagotchiWorld({seed});
    this.physics = new SwarmWorld();
    for(const object of this.world.objects) {
      this.physics.addParticle({
        id:object.id,x:object.x+5,y:object.z+5,
        mass:Math.max(.2,object.sx*object.sz),bit:object.morphState%2,
        qualities:{type:object.type}
      });
    }
    this.commands=[];
    this.currentMode=null;
    const registry=new FunctionRegistry();
    registry.register('world.apply', input => {
      const before=this.object(input.targetId);
      this.world.select(input.targetId);
      if(!this.world.morphSelected(input.op))throw new Error('World engine rejected the operation');
      const after=this.object(input.targetId);
      const changed=JSON.stringify(before)!==JSON.stringify(after);
      if(!changed)throw new Error('World reported success without state change');
      this.syncObjects();
      return {...input,before,after,changed};
    });
    registry.register('world.impulse', input => {
      this.physics.impulse(input.targetId,1.7,-1.2);
      for(let i=0;i<18;i++)this.physics.step(1/120);
      return {...input,physicsSteps:18};
    });
    registry.register('world.compose', input => {
      const created=this.world.objects.find(o=>o.id!==input.targetId && o.name===input.before.name+' copy' && !o.removed);
      if(!created)throw new Error('The original world did not create a new object');
      const compositeId='formation-'+this.commands.length+'-'+created.id;
      const comp=this.physics.compose(compositeId,[input.targetId,created.id],'o_bundle');
      return {...input,compositeId:comp.id};
    });
    registry.register('world.capture', input => {
      const event=this.world.worldLedger.filter(e=>e.type==='MORPH'||e.type==='EMERGENT_EXECUTION').at(-1);
      if(!event)throw new Error('No world execution receipt exists');
      return {...input,event:clone(event),objects:items(this.world,this.physics),physicsTick:this.physics.tick};
    });
    this.formation=new ExecutableFormation({
      id:'simlish-world-formation',address:ADDRESS,registry,
      graph:{nodes:[{id:'input'},{id:'output'}],edges:[{id:'observe',from:'input',to:'output',functionId:'world.capture'}]}
    });
  }
  object(id) {
    const object=this.world.object(id);
    if(!object)throw new Error('World target is unavailable: '+id);
    const {name,type,x,z,sx,sy,sz,morphState}=object;
    return {id,name,type,x,z,sx,sy,sz,morphState};
  }
  syncObjects() {
    for(const object of this.world.objects) {
      if(object.removed)continue;
      const node=this.physics.particles.get(object.id);
      if(node) {
        // The same world object determines the simulation particle's addressable position.
        if(object.type==='plant')node.mass=Math.max(.2,object.sy*.45);
        if(object.x+5!==node.x && this.currentMode!=='Mission')node.x=object.x+5;
        if(object.z+5!==node.y && this.currentMode!=='Mission')node.y=object.z+5;
      } else {
        this.physics.addParticle({id:object.id,x:object.x+5,y:object.z+5,
          mass:Math.max(.2,object.sx*object.sz),bit:object.morphState%2,qualities:{type:object.type}});
      }
    }
  }
  snapshot() {
    return {
      objects:items(this.world,this.physics),
      composites:[...this.physics.composites.values()].map(clone),
      edges:this.physics.edges.map(clone),
      physicsTick:this.physics.tick,
      activeRevision:this.formation.describe().activeRevision,
      formationRevisions:this.formation.describe().revisions.length,
      events:this.world.worldLedger.length,
      interactions:this.commands.length
    };
  }
  async perform({mode='Play',targetId='chair',op='transform'}={}) {
    if(!MODES.includes(mode))throw new Error('Unknown experience mode');
    if(!OPS.includes(op))throw new Error('Unknown physical operation');
    this.object(targetId);
    // Create is a real duplication + swarm composition. Mission is a measured chair transform.
    const effectiveOp=mode==='Create'?'duplicate':mode==='Mission'?'transform':op;
    const input={mode,targetId,op:effectiveOp};
    const prior=this.snapshot();
    if(this.currentMode!==mode) {
      this.formation.transform(graph(mode),{reason:'world-mode-'+mode.toLowerCase()});
      this.currentMode=mode;
    }
    const run=await this.formation.execute(input);
    const output=run.outputs.output;
    const completed=mode==='Mission' ? targetId==='chair'&&output.changed&&!!output.event : null;
    const command={mode,targetId,op:effectiveOp};
    this.commands.push(command);
    return {command,completed,prior,after:this.snapshot(),
      event:output.event,changed:output.changed,
      compositeId:output.compositeId??null,
      formationRevision:this.formation.describe().activeRevision,
      formationArtifact:run.artifactId};
  }
  export() {
    return {format:'you-n-i-verse-world-session-v1',seed:this.world.seed,commands:clone(this.commands),
      state:this.snapshot(),formationHistory:this.formation.history(),
      worldLedger:clone(this.world.worldLedger),swarmHistory:clone(this.physics.history)};
  }
  static async fromCommands(saved) {
    if(!saved||saved.format!=='you-n-i-verse-world-session-v1'||!Array.isArray(saved.commands))
      throw new Error('Unrecognized world session format');
    const session=new WorldSession({seed:saved.seed});
    for(const c of saved.commands)await session.perform(c);
    return session;
  }
}
