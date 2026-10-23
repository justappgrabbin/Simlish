/** Deterministic RGBA atlas packing. No image model, network, DOM, or opaque format fallback. */
const integer=(n,min,max,name)=>{if(!Number.isInteger(n)||n<min||n>max)throw new RangeError(`${name} must be ${min}..${max}`);return n;};
export function packSpriteSheet(frames,{columns=4,padding=1,maxSide=4096}={}){
  if(!Array.isArray(frames)||frames.length<1||frames.length>256)throw new RangeError('sprite frames must be 1..256');
  integer(columns,1,256,'columns');integer(padding,0,32,'padding');integer(maxSide,1,16384,'maxSide');
  const normalized=frames.map((f,i)=>{
    integer(f?.width,1,maxSide,`frame ${i} width`);integer(f?.height,1,maxSide,`frame ${i} height`);
    const bytes=f?.data;
    if(!(bytes instanceof Uint8Array)&&!(bytes instanceof Uint8ClampedArray)&&!Array.isArray(bytes))throw new TypeError(`frame ${i} requires actual RGBA pixels`);
    if(bytes.length!==f.width*f.height*4)throw new RangeError(`frame ${i} RGBA byte length mismatch`);
    return {width:f.width,height:f.height,data:bytes,id:typeof f.id==='string'?f.id:`frame-${i}`,durationMs:f.durationMs??100};
  });
  if(new Set(normalized.map(x=>x.id)).size!==normalized.length)throw new Error('duplicate sprite frame ID');
  const cellWidth=Math.max(...normalized.map(f=>f.width));const cellHeight=Math.max(...normalized.map(f=>f.height));
  const cols=Math.min(columns,normalized.length),rows=Math.ceil(normalized.length/cols);
  const width=cols*(cellWidth+padding)+padding,height=rows*(cellHeight+padding)+padding;
  if(width>maxSide||height>maxSide)throw new RangeError('sprite atlas exceeds bounded output size');
  const data=new Uint8Array(width*height*4),entries=[];
  normalized.forEach((f,i)=>{
    const x=padding+(i%cols)*(cellWidth+padding),y=padding+Math.floor(i/cols)*(cellHeight+padding);
    for(let row=0;row<f.height;row++)data.set(f.data.subarray?f.data.subarray(row*f.width*4,(row+1)*f.width*4):f.data.slice(row*f.width*4,(row+1)*f.width*4),((y+row)*width+x)*4);
    integer(f.durationMs,1,60000,`frame ${i} durationMs`);
    entries.push({id:f.id,index:i,x,y,width:f.width,height:f.height,durationMs:f.durationMs});
  });
  return {schema:'synthia.sprite-atlas.v1',width,height,channels:4,format:'RGBA8',data,frames:entries,padding,columns:cols,rows,
    sourcePolicy:'real-input-pixels-only; alpha preserved; no generated skin fidelity implied'};
}
export function frameFromAtlas(atlas,id){
  const f=atlas.frames.find(x=>x.id===id);if(!f)throw new RangeError('unknown frame id');
  const data=new Uint8Array(f.width*f.height*4);
  for(let y=0;y<f.height;y++)data.set(atlas.data.subarray(((f.y+y)*atlas.width+f.x)*4,((f.y+y)*atlas.width+f.x+f.width)*4),y*f.width*4);
  return {width:f.width,height:f.height,data,durationMs:f.durationMs,id:f.id};
}
