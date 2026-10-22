"""Evaluate any checkpoint against labeled JSONL, with per-head confusion matrices."""
import argparse
import json
import numpy as np
from neural.cli import records
from neural.sequence_model import SequenceContactModel


def evaluate(model, data):
    matrices={head:np.zeros((len(labels),len(labels)),dtype=int) for head,labels in model.heads.items()}
    for record in data:
        predictions=model.forward(record['events'])[0]
        for head,label in record.get('targets',{}).items():
            if head not in predictions: raise ValueError('Missing evidence for evaluation target '+head)
            matrices[head][model.heads[head].index(label),predictions[head].argmax()]+=1
    result={}
    for head,matrix in matrices.items():
        count=int(matrix.sum())
        if count:
            f1=[2*matrix[i,i]/max(1,matrix[i].sum()+matrix[:,i].sum()) for i in range(len(matrix))]
            result[head]={'examples':count,'accuracy':float(np.trace(matrix)/count),'macro_f1':float(np.mean(f1)),
                          'labels':model.heads[head],'confusion_matrix':matrix.tolist()}
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True); parser.add_argument('--data',required=True); parser.add_argument('--output')
    args=parser.parse_args()
    result=json.dumps(evaluate(SequenceContactModel.load(args.model),records(args.data)),indent=2)+'\n'
    if args.output:
        with open(args.output,'w') as output: output.write(result)
    print(result)
