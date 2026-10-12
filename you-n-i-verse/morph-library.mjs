// Target forms are supplied artwork, with open names rather than a type whitelist.
export class MorphLibrary {
  constructor(morph) { this.morph=morph; this.targets=new Map(); this.source=null; this.revision=0; }
  setSource(source) { if(source===this.source)return; this.source=source; this.revision++; for(const target of this.targets.values()) {target.result=null;target.pending=null;} }
  add(id,name,image) {this.targets.set(id,{id,name,image,result:null,pending:null});}
  async prepare(id) {
    if(!this.source)throw new Error('Choose your identity photo first.');
    const target=this.targets.get(id);if(!target)throw new Error('Target form not found.');
    if(target.result)return target.result;
    if(target.pending)return target.pending;
    const revision=this.revision,source=this.source;
    const promise=Promise.resolve().then(()=>this.morph(source,target.image)).then(result=>{
      if(revision!==this.revision || this.targets.get(id)!==target)throw new Error('Photo or target changed; morph again.');
      target.result=result;return result;
    });
    target.pending=promise;
    try{return await promise;}finally{if(target.pending===promise)target.pending=null;}
  }
  async prepareAll(onProgress=()=>{}) {
    const results=[];
    // Keep local inference sequential to avoid competing for mobile memory.
    for(const target of this.targets.values()){
      try{results.push({id:target.id,result:await this.prepare(target.id)});}catch(error){results.push({id:target.id,error});}
      onProgress(results.length,this.targets.size,results.at(-1));
    }
    return results;
  }
}
