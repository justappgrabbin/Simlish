// The player's composed body always outranks an unpersonalized animation sheet.
// A failed automatic morph remains visible through the manual photo overlay.
export function selectMorphPreview({body=null,sprites=null,morphedBody=null,morphedSprites=null}={}){
 if(sprites && morphedSprites)return {kind:'personalized-animation',image:morphedSprites,needsManualFace:false};
 if(body)return {kind:'body',image:morphedBody??body,needsManualFace:!morphedBody};
 if(sprites)return {kind:'unpersonalized-sheet',image:sprites,needsManualFace:true};
 return {kind:'empty',image:null,needsManualFace:false};
}
