// Cache the original encoded source once; derived forms never replace this record.
export function validatePhotoIdentity(identity){
 if(identity===undefined)return;
 if(!identity||identity.version!==1||identity.id!=='photo:'+identity.sha256||!/^[a-f0-9]{64}$/.test(identity.sha256))throw Error('Invalid photo identity');
}
export class PhotoIdentityStore{
 #records=new WeakMap();#pending=new WeakMap();
 get(photo){return photo?this.#records.get(photo):undefined;}
 async capture(photo,serialize,{encoded,expected}={}){
  if(!photo)return null;
  if(this.#records.has(photo))return this.#records.get(photo);
  if(this.#pending.has(photo))return this.#pending.get(photo);
  const pending=(async()=>{
   const source=encoded??serialize(photo);
   if(typeof source!=='string'||!/^data:image\/(png|jpeg|webp);base64,/.test(source))throw Error('Invalid identity photo');
   const payload=atob(source.split(',')[1]),bytes=Uint8Array.from(payload,c=>c.charCodeAt(0));
   const sha256=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');
   const identity=Object.freeze({version:1,id:'photo:'+sha256,sha256});
   validatePhotoIdentity(expected);
   if(expected&&expected.id!==identity.id)throw Error('Saved identity does not match its source photo');
   const record=Object.freeze({photo:source,identity});this.#records.set(photo,record);return record;
  })();this.#pending.set(photo,pending);
  try{return await pending;}finally{this.#pending.delete(photo);}
 }
}
