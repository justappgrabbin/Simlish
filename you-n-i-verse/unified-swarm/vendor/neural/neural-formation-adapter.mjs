/** Additive adapter: lowers a layered dense network to the EXISTING Synthia primitive VM.
 * No new execution engine, backend, worker, or model dependency.
 * An address may be supplied by the caller and is preserved in the formation metadata.
 */
export function composeDenseFormation({layers, address=null, id='neural-formation'}={}) {
  if (!Array.isArray(layers)||!layers.length) throw new TypeError('layers required');
  const nodes=[];
  let previous='input';
  let width=null;
  layers.forEach((layer,index)=>{
    const weights=layer.weights;
    if (!Array.isArray(weights)||!weights.length||!weights.every(row=>Array.isArray(row)&&row.length>0)) throw new TypeError('nonempty rectangular weight matrix required');
    const inputWidth=weights.length, outputWidth=weights[0].length;
    if (!weights.every(row=>row.length===outputWidth&&row.every(Number.isFinite))) throw new TypeError('weights must be finite rectangular matrix');
    if (width!==null&&width!==inputWidth) throw new Error('adjacent layer shape mismatch');
    if (!Array.isArray(layer.bias)||layer.bias.length!==outputWidth||!layer.bias.every(Number.isFinite)) throw new TypeError('bias shape mismatch');
    const activation=layer.activation??'identity';
    if (!['identity','relu','sigmoid','tanh','softmax'].includes(activation)) throw new Error('unsupported activation');
    nodes.push({id:`w${index}`,op:'const',value:weights});
    nodes.push({id:`b${index}`,op:'const',value:[layer.bias]});
    nodes.push({id:`z${index}`,op:'matmul',inputs:[previous,`w${index}`]});
    nodes.push({id:`a${index}`,op:'add',inputs:[`z${index}`,`b${index}`]});
    nodes.push({id:`h${index}`,op:activation,inputs:[`a${index}`]});
    previous=`h${index}`;width=outputWidth;
  });
  return {id,address,program:{nodes,outputs:[previous]},output:previous,inputWidth:layers[0].weights.length,outputWidth:width};
}
export async function executeNeuralFormation(vm,formation,input) {
  if (!Array.isArray(input)||input.length!==formation.inputWidth||!input.every(Number.isFinite)) throw new TypeError('input vector shape mismatch');
  const result=await vm.run(formation.program,{inputs:{input:[input]}});
  return {formationId:formation.id,address:formation.address,output:result.returnValue[formation.output][0],trace:result.trace};
}
