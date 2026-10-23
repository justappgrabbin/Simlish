import {ConsciousnessSwarm} from './consciousness-swarm.mjs';

/**
 * Adapter to CURRENT SynthAi ComputerRuntime service contracts (verified source paths):
 *  computer/ComputerRuntime.mjs
 *  computer/services/address-service.mjs
 *  computer/services/automata-engine.mjs
 *  computer/events/event-emitter.mjs
 *
 * Does not replace any donor, resolver, ATO, or registry. No direct code evaluation.
 * Install by passing the real already-booted ComputerRuntime. This add-on is
 * intentionally non-invasive until tested inside the complete host tree.
 */
export function connectToComputer(runtime,{swarm=new ConsciousnessSwarm(),trajectoryId='consciousness-swarm'}={}){
  for(const method of ['resolveAddress','resolveRelationship','executeOnSwarm','emitEvent'])
    if(typeof runtime?.[method]!=='function')throw new Error(`Existing ComputerRuntime must provide ${method}()`);
  const emit=async(event_type,input,result,status='success')=>runtime.emitEvent({
    actor_id:'synthia:consciousness-layer',actor_type:'automaton',event_type,
    input,provider:'Synthia Computer existing services + additive consciousness layer',
    result,result_status:status,trajectory_id:trajectoryId,observable_effect:`${event_type}: ${status}`
  });
  return Object.freeze({
    swarm,
    async resolveAndIngest(input,options={}){
      // One canonical address, recovered through the existing provider rather than invented here.
      const resolved=await runtime.resolveAddress(input,options);
      if(resolved.status==='unknown')throw new Error(`Existing address provider could not resolve input: ${resolved.reason??'unknown reason'}`);
      const primitive=swarm.createPrimitive({value:input,kind:'addressed-observation',address:resolved.address,properties:{addressProvider:resolved.provider}});
      await emit('observation',{input,addressProvider:resolved.provider},{nodeId:primitive.id,address:resolved.address});
      return {primitive,address:resolved.address,provider:resolved.provider};
    },
    async connectResolved(a,b,context={}){
      const nodeA=swarm._get(a),nodeB=swarm._get(b);
      if(nodeA.address.gate==null||nodeB.address.gate==null)throw new Error('Actual gate addresses required for host edge resolution');
      const resolved=await runtime.resolveRelationship(nodeA.address,nodeB.address,context);
      const edge=swarm.connect(a,b,{relation:resolved.edge?.edge_type??'host-resolved-relationship',context:resolved,
        provenance:{provider:resolved.edge?.provider??resolved.provider??'host-resolver',status:'host-returned'}});
      await emit('network_connection',{a,b,context},{edgeId:edge.id,hostEdge:resolved});
      return {edge,hostResolution:resolved};
    },
    async execute(capability,input,context={}){
      // Delegate work to the real installed donor manifest. No Function/eval/new Function.
      const r=await runtime.executeOnSwarm(capability,input,context);
      const status=r.status==='complete'||r.status==='success'?'success':r.status==='failed'?'failure':'partial';
      await emit('tool_execution',{capability,input},r,status);
      return r;
    },
    async landFormation(targets,options={}){
      const step=swarm.stepFormation(targets,options);
      await emit('state_transition',{operation:'formation-step',targets},step);
      return step;
    },
    async persistSnapshot(namespace='consciousness-swarm'){
      if(typeof runtime.state?.set!=='function'||typeof runtime.state?.get!=='function')
        throw new Error('Host StateStore get/set unavailable: snapshot was not saved');
      // ComputerRuntime defaults to MemoryPersistence. Calling it durable would be misleading:
      // a process restart with a fresh MemoryPersistence loses the snapshot.
      if(runtime.state.persistence?.constructor?.name==='MemoryPersistence')
        throw new Error('Restart-unsafe host: MemoryPersistence cannot retain the swarm across app restarts');
      const state=swarm.snapshot();
      await runtime.state.set(`${namespace}.snapshot`,state,{source:'consciousness-swarm'});
      const readback=runtime.state.get(`${namespace}.snapshot`);
      if(!readback||readback.nextEvent!==state.nextEvent||readback.nodes?.length!==state.nodes.length)
        throw new Error('Host state readback failed: snapshot is not confirmed');
      await emit('observation',{operation:'persistSnapshot',namespace},{eventCount:state.events.length,particleCount:state.formation.length});
      return {namespace,eventCount:state.events.length,nodeCount:state.nodes.length,stateStoreReadbackConfirmed:true,hostPersistenceType:runtime.state.persistence?.constructor?.name??'not-exposed'};
    }
  });
}

/** Mount in the EXISTING Computer services registry after its boot(), never replace registrations. */
export function mountConsciousnessService(runtime, options={}) {
  if(typeof runtime?.services?.register!=='function') throw new Error('Existing Computer services Registry.register required');
  if(runtime.services.has?.('consciousness-swarm')) throw new Error('consciousness-swarm already registered: preserve existing instance');
  // Resume the real saved state BEFORE registration. Never silently start a blank swarm
  // if a stored checkpoint is corrupt. The host owns persistence and storage policy.
  const namespace=options.namespace??'consciousness-swarm';
  const saved=runtime.state?.get?.(`${namespace}.snapshot`);
  const existing=options.swarm;
  if(saved!==undefined && existing!==undefined)
    throw new Error('Saved swarm exists: explicit new instance would reset history');
  const swarm=existing??(saved!==undefined?ConsciousnessSwarm.restore(saved):new ConsciousnessSwarm());
  const bridge=connectToComputer(runtime,{...options,swarm});
  runtime.services.register('consciousness-swarm',{
    provider:bridge,
    contract:'resolveAndIngest|connectResolved|execute|landFormation|persistSnapshot'
  });
  runtime.bus?.emit('consciousness-swarm:mounted',{serviceId:'consciousness-swarm', provider:'additive Synthia layer'});
  return bridge;
}
