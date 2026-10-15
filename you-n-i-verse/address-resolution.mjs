// Canonical positional address; preserve legacy kernel 'arc' field.
export const ADDRESS_ORDER=Object.freeze(['planetary','dimension','zodiac','gate','line','color','tone','base','degree','minute','second','arcSecond','house']);
export const ADDRESS_SCHEMA_VERSION=2;
export function normalizeAddress(input){
 if(!input||typeof input!=='object')return {};
 if(input.arcSecond!==undefined&&input.arc!==undefined&&input.arcSecond!==input.arc)throw new Error('Conflicting arcSecond and legacy arc');
 return {...input,arcSecond:input.arcSecond??input.arc};
}
export function resolveAddress({kernel,leadingDimension,coordinates,evidence}={}){
 if(!kernel?.FullAddress||!Array.isArray(kernel.DIMENSIONS))throw new Error('Address kernel unavailable');
 if(!kernel.DIMENSIONS.includes(leadingDimension))return {ok:false,reason:'Leading dimension unresolved',needs:['leadingDimension']};
 const normalized=normalizeAddress(coordinates);
 if(normalized.dimension!==undefined&&normalized.dimension!==leadingDimension)return {ok:false,reason:'Address dimension conflicts with situational leading dimension',needs:['dimension alignment']};
 const missing=ADDRESS_ORDER.filter(k=>k!=='dimension'&&(normalized[k]===undefined||normalized[k]===null||normalized[k]===''));
 if(missing.length)return {ok:false,reason:'Insufficient semantic coordinates',needs:missing,leadingDimension};
 if(!evidence||typeof evidence!=='object'||!evidence.source)return {ok:false,reason:'Address provenance required',needs:['evidence.source']};
 const raw={...normalized,dimension:leadingDimension,arc:normalized.arcSecond};
 const validated=new kernel.FullAddress(raw).toObject();
 const address=Object.fromEntries(ADDRESS_ORDER.map(k=>[k,k==='arcSecond'?validated.arc:validated[k]]));
 return {ok:true,schemaVersion:ADDRESS_SCHEMA_VERSION,address,leadingDimension,evidence,projections:[address],kernelAddress:validated};
}
