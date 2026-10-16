// Lossless round trip, measured per file type, reported as found. NOT every file/program is claimed reducible and reconstructible.
// Three different questions, answered separately:
//  1 bytes -> six-bit primitives -> bytes (Alexis's descend packing; needs the byte length)      exact for any bytes by construction
//  2 bytes -> Bantu 6-bit tokens -> bytes (her bridge.js encode/decode)                          exact for any bytes (her codec)
//  3 ontological address ALONE -> bytes                                                          NOT possible: the address is a projection; her ingestor keeps the original bytes as stored data
//  4 behaviour: the primitive organization re-resolved from the primitives gives the same address (deterministic resolution)
//  Source-level reconstruction (turning primitives back into the author's JS/Python/HTML text) is NOT attempted: not claimed.
import {ExecutionAddressResolver} from './spine/execution-address-resolver.mjs';
import {OntologicalFileIngestor} from './ontological/ingestion/ontological-file-ingestor.mjs';
import {encodeBytes,decodeBytes} from './bantu.mjs';
import {bytesToAtoms,atomsToBytes} from './population.mjs';
const eq=(a,b)=>a.length===b.length&&a.every((v,i)=>v===b[i]);
export async function roundTrip(name,bytes){
 const res=new ExecutionAddressResolver(),ing=new OntologicalFileIngestor();
 const dec=res.descend({name,bytes}),truncatedStates=bytes.length*8/6>dec.sixBitStates.length;
 const atoms=bytesToAtoms(bytes),viaPrimitives=eq(atomsToBytes(atoms).slice(0,bytes.length),bytes)&&atoms.length===Math.ceil(bytes.length*8/6);
 const herDescendMatches=!truncatedStates&&eq([...dec.sixBitStates],atoms);
 let bantu=false;try{bantu=eq(decodeBytes(encodeBytes(bytes)),bytes)}catch{}
 let ingest=false;try{const r=await ing.ingest({name,bytes},{supersede:true}),back=await ing.reconstruct(r.address?r.combinatorialAddress:r.id);ingest=!!back&&eq(new Uint8Array(back.bytes??back),bytes)}catch(e){ingest='error: '+e.message}
 const a1=res.resolve({artifact:{name,bytes}}).key,a2=res.resolve({artifact:{name,bytes:bytes.slice()}}).key;
 return {name,bytes:bytes.length,primitives:atoms.length,viaPrimitives,herDescendMatches,herDescendTruncated:truncatedStates,bantuTokens:bantu,ingestReconstruct:ingest,fromAddressAlone:false,deterministicAddress:a1===a2,sourceLevel:'not attempted'};
}
