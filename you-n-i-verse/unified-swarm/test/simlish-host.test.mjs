import test from 'node:test';import assert from 'node:assert/strict';
import {attachUnifiedSwarmToSimlish} from '../src/simlish-host.mjs';
test('host integrates one source and one Simlish plug, no competing semantic source',()=>{
 const assembly={structure:(subject,context)=>({subject,context})};
 const simlishPlug={morph:async()=>({phenotype:{status:'blocked'}})};
 const s=attachUnifiedSwarmToSimlish({assembly,simlishPlug});
 s.add({id:'agent',label:'Agent'});
 const resumed=attachUnifiedSwarmToSimlish({assembly,simlishPlug,snapshot:s.snapshot()});
 assert.equal(resumed.swarm.nodes.size,1);
 assert.deepEqual(resumed.inspect('thought').context.addressMode,'structural');
 assert.throws(()=>attachUnifiedSwarmToSimlish({assembly}),/CONNECTED_SOURCE_AND_SIMLISH_REQUIRED/);
});
