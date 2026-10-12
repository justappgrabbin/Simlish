export function inspectGLB(buffer){
 if(!(buffer instanceof ArrayBuffer)||buffer.byteLength<20||buffer.byteLength>32*1024*1024)throw Error('Choose a GLB model under 32 MB.');
 const view=new DataView(buffer);if(view.getUint32(0,true)!==0x46546c67||view.getUint32(4,true)!==2||view.getUint32(8,true)!==buffer.byteLength)throw Error('This is not a valid GLB 2.0 file.');
 const length=view.getUint32(12,true);if(view.getUint32(16,true)!==0x4e4f534a||20+length>buffer.byteLength)throw Error('GLB JSON chunk is missing or truncated.');
 const json=JSON.parse(new TextDecoder().decode(new Uint8Array(buffer,20,length)).replace(/\0+$/,''));
 for(const item of [...(json.buffers??[]),...(json.images??[])])if(item.uri&&!item.uri.startsWith('data:'))throw Error('Use a self-contained GLB with embedded textures and buffers.');
 if(json.extensionsRequired?.some(x=>['KHR_draco_mesh_compression','EXT_meshopt_compression','KHR_texture_basisu'].includes(x)))throw Error('Export an uncompressed GLB; this offline player does not bundle these compression decoders.');
 return {json,skins:json.skins?.length??0,animations:(json.animations??[]).map((a,i)=>a.name??'Animation '+(i+1)),morphPrimitives:(json.meshes??[]).flatMap(m=>m.primitives??[]).filter(p=>p.targets?.length).length};
}
