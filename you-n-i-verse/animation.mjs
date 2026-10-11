export function sheetGrid(width,height,cols,rows){
 if(![width,height,cols,rows].every(Number.isInteger)||width<1||height<1||cols<1||rows<1||cols>16||rows>16||width%cols||height%rows)throw Error('Use a grid that divides the sheet into equal-sized frames.');
 const frameWidth=width/cols,frameHeight=height/rows;
 if(frameWidth*frameHeight*cols*rows>16000000)throw Error('Sheet is too large. Use a sheet under 16 million pixels.');
 return {cols,rows,frameWidth,frameHeight,count:cols*rows};
}
export async function personalizeFrames(photo,sheet,grid,{morph,onProgress=()=>{},cancelled=()=>false}={}){
 const out=document.createElement('canvas');out.width=sheet.width;out.height=sheet.height;const ctx=out.getContext('2d');const receipts=[];
 for(let i=0;i<grid.count;i++){
  if(cancelled())throw Error('Animation preparation cancelled.');
  const cell=document.createElement('canvas');cell.width=grid.frameWidth;cell.height=grid.frameHeight;
  cell.getContext('2d').drawImage(sheet,(i%grid.cols)*cell.width,Math.floor(i/grid.cols)*cell.height,cell.width,cell.height,0,0,cell.width,cell.height);
  let result;try{result=await morph(photo,cell)}catch(error){throw Error('Frame '+(i+1)+': '+error.message+' No partial sheet was applied.');}
  if(cancelled())throw Error('Animation preparation cancelled.');
  ctx.drawImage(result.canvas,0,0,result.canvas.width,result.canvas.height,(i%grid.cols)*cell.width,Math.floor(i/grid.cols)*cell.height,cell.width,cell.height);
  receipts.push(result.receipt);onProgress(i+1,grid.count);
  await new Promise(resolve=>setTimeout(resolve,0));
 }
 return {canvas:out,receipt:{method:'per-frame-local-face-morph',frames:grid.count,grid,receipts,network:false}};
}
export function actionKeyposes(reference){
 // Four detectable key poses from the user's board, not a synthesized walk cycle.
 const sheet=document.createElement('canvas');sheet.width=100*4;sheet.height=180;
 const indices=[0,1,2,11];
 indices.forEach((j,i)=>sheet.getContext('2d').drawImage(reference,600+(j%6)*112,170+Math.floor(j/6)*215,100,180,i*100,0,100,180));return sheet;
}
