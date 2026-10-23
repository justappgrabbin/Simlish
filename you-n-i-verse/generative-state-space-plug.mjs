// Simlish is the consumer of Generative-address-system; genotype and phenotype remain distinct.
const clone = x => structuredClone(x);
const OPS = Object.freeze(['transform', 'move', 'duplicate']);
export function connectGenerativeStateSpace({assembly, world, authorize, renderer = null, actionRegistry = null} = {}) {
  if (typeof assembly?.structure !== 'function' || typeof assembly?.createSwarmRuntime !== 'function')
    throw new TypeError('Live Generative State Space assembly required');
  if (typeof world?.object !== 'function' || typeof world?.morphSelected !== 'function' || !Array.isArray(world.worldLedger))
    throw new TypeError('Live Simlish world required');
  const records = [];
  const swarm = assembly.createSwarmRuntime({
    authorize, renderer, actionRegistry, capabilities: [{
      name: 'simlish.phenotype.morph', sideEffect: true,
      execute: ({objectId, operation}) => {
        if (typeof objectId !== 'string' || !OPS.includes(operation)) throw new TypeError('Unsafe Simlish morph');
        const object = world.object(objectId);
        if (!object || object.usable === false) throw new Error('Target unavailable');
        const before = clone(object), previous = world.selectedId;
        let ok = false;
        try { world.select(objectId); ok = world.morphSelected(operation); }
        finally { world.select(previous); }
        if (!ok) throw new Error('SIMLISH_MORPH_NOT_EXECUTED');
        return {kind:'phenotype-expression', objectId, operation, before,
          after: clone(world.object(objectId)), worldEvent:clone(world.worldLedger.at(-1)),
          addressPolicy:'Simlish event hashes are simulation coordinates, not verified genotype'};
      }
    }]
  });
  const inspect = (subject, context = {}) => assembly.structure(subject,
    {...context, addressMode:'structural'});
  async function morph({intent, genotypeAddress, genotypeSource, objectId, operation='transform',
                        subject=null, context={}} = {}) {
    if (!['owner-explicit','externally-verified'].includes(genotypeSource))
      throw new TypeError('Canonical genotype provenance required; world hashes are not authoritative');
    if (typeof intent !== 'string' || !intent.trim()) throw new TypeError('Intent required');
    if (!OPS.includes(operation)) throw new TypeError('Only non-destructive morph operations allowed');
    if (!world.object(objectId)) throw new RangeError('Unknown world object');
    const structural = subject == null ? null : inspect(subject, context);
    // Existing Generative State Space owns address validation, witness and authorization.
    const result = await swarm.execute({intent,address:genotypeAddress,
      autonomy:context.autonomy === 'permissioned' ? 'permissioned' : 'manual',
      context:{...context,genotypeSource,phenotypeSystem:'Simlish'},
      steps:[{id:'phenotype',op:'simlish.phenotype.morph',input:{objectId,operation}}]});
    const record = {schema:'simlish.genotype-phenotype.v1',
      genotype:{address:clone(genotypeAddress),provenance:genotypeSource},
      phenotype:{status:result.ok?'executed-and-observed':'blocked',
        expression:result.ok?clone(result.output):null},
      semanticCloud:{status:structural?'structural-analysis':'not-evaluated',observation:structural},
      swarmReceipt:clone(result)};
    records.push(clone(record));
    return record;
  }
  return Object.freeze({kind:'simlish-generative-state-space-plug',dualStateSpace:true,
    inspect,morph, state:()=>({history:clone(records),capabilities:swarm.capabilityNames(),
      world:world.summary?.()??null})});
}
