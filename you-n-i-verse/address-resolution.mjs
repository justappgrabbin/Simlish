// Semantic address resolution: refuse fabricated coordinates.
// The leading dimension is determined by the caller's analyzed situation.
export const ADDRESS_ORDER=['gate','line','color','tone','base','degree','minute','second','arc','zodiac','house'];
export function resolveAddress({kernel,leadingDimension,coordinates,evidence}={}){
 if(!kernel?.FullAddress||!Array.isArray(kernel.DIMENSIONS))throw new Error('Address kernel unavailable');
 if(!kernel.DIMENSIONS.includes(leadingDimension))return {ok:false,reason:'Leading dimension unresolved',needs:['leadingDimension']};
 const missing=ADDRESS_ORDER.filter(k=>coordinates?.[k]===undefined||coordinates?.[k]===null);
 if(!coordinates?.planetary)missing.push('planetary');
 if(missing.length)return {ok:false,reason:'Insufficient semantic coordinates',needs:missing,leadingDimension};
 if(!evidence||typeof evidence!=='object'||!evidence.source)return {ok:false,reason:'Address provenance required',needs:['evidence.source']};
 const address=new kernel.FullAddress({...coordinates,dimension:leadingDimension}).toObject();
 return {ok:true,address,leadingDimension,evidence,projections:[address]};
}
