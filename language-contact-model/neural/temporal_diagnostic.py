"""Synthetic check that prior events can change interpretation of the same signal."""
import json
import numpy as np
from neural.sequence_model import SequenceContactModel
from neural.gesture_benchmark import ROOT


def run():
    rng=np.random.default_rng(42)
    def dataset(n):
        result=[]
        for _ in range(n):
            history=rng.choice([-1.,1.]); signal=rng.choice([-1.,1.])
            result.append({'events':[{'observations':{'history':[history]}},
                                     {'observations':{'body':[signal]}}],
                           'targets':{'meaning':'accept' if history*signal>0 else 'decline'}})
        return result
    train,test=dataset(256),dataset(128)
    model=SequenceContactModel({'history':1,'body':1},hidden=16,heads={'meaning':['accept','decline']})
    for _ in range(120):
        order=rng.permutation(len(train))
        for start in range(0,len(train),32):
            model.train_batch([train[i] for i in order[start:start+32]],.1,prediction_weight=0)
    def accuracy(last=False):
        return float(np.mean([model.heads['meaning'][model.forward(r['events'][-1:] if last else r['events'])[0]['meaning'].argmax()]
                              ==r['targets']['meaning'] for r in test]))
    report={'dataset':'synthetic history-dependent interpretation diagnostic',
            'sequence_accuracy':accuracy(),'last_event_only_accuracy':accuracy(True),
            'meaning':'Identical final observations require different interpretations according to prior context.'}
    (ROOT/'artifacts'/'temporal-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    model.save(ROOT/'artifacts'/'temporal-diagnostic.npz')
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__': run()
