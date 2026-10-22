"""Train one shared recurrent model across real gesture, speech and sign trajectories.

Datasets are independent, not fabricated aligned conversations. Each objective has
its own expandable head; encoders, recurrent dynamics and decoders are shared.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from neural.sequence_model import SequenceContactModel
from neural.gesture_benchmark import ROOT, LABELS


def load_ts(path):
    rows=[]; started=False
    for line in Path(path).read_text().splitlines():
        line=line.strip()
        if line.lower()=='@data': started=True; continue
        if not started or not line or line.startswith('#'): continue
        parts=line.split(':'); label=parts[-1].strip()
        features=[np.fromstring(part,sep=',') for part in parts[:-1]]
        if len(features)!=12 or len(set(map(len,features)))!=1:
            raise ValueError('Invalid 12-channel speech sequence')
        rows.append((np.column_stack(features),label))
    return rows


def prepare():
    speech_dir=ROOT/'data'/'speech'
    train=load_ts(speech_dir/'JapaneseVowels_TRAIN.ts'); test=load_ts(speech_dir/'JapaneseVowels_TEST.ts')
    raw=np.concatenate([x for x,y in train]); mean=raw.mean(0); std=raw.std(0); std[std<1e-6]=1
    def speech_records(rows):
        return [{'events':[{'observations':{'speech':row.tolist()}} for row in (x-mean)/std],
                 'targets':{'speaker':y},'source':'JapaneseVowels'} for x,y in rows]
    speech_train=speech_records(train); speech_test=speech_records(test)
    raw_sign=np.loadtxt(ROOT/'data'/'sign'/'movement_libras.data',delimiter=',')
    if raw_sign.shape!=(360,91): raise ValueError('Expected complete LIBRAS dataset')
    rng=np.random.default_rng(7); train_indices=[]; test_indices=[]
    # Split groups of identical trajectories together; retain labels as opaque IDs.
    for label in sorted(set(raw_sign[:,-1])):
        groups={}
        for index in np.where(raw_sign[:,-1]==label)[0]:
            key=hashlib.sha256(raw_sign[index,:-1].tobytes()).hexdigest()
            groups.setdefault(key,[]).append(int(index))
        keys=list(groups); rng.shuffle(keys); boundary=max(1,int(len(keys)*2/3))
        for key in keys[:boundary]: train_indices.extend(groups[key])
        for key in keys[boundary:]: test_indices.extend(groups[key])
    sign_x=[row[:-1].reshape(45,2) for row in raw_sign]
    sign_values=np.concatenate([sign_x[i] for i in train_indices])
    sign_mean=sign_values.mean(0); sign_std=sign_values.std(0); sign_std[sign_std<1e-6]=1
    def sign_records(indices):
        result=[]
        for i in indices:
            x=(sign_x[i]-sign_mean)/sign_std
            motion=np.vstack([np.zeros(2),np.diff(x,axis=0)])
            result.append({'events':[{'observations':{'sign':row.tolist(),'sign_motion':speed.tolist()}}
                                     for row,speed in zip(x,motion)],
                           'targets':{'movement':str(int(raw_sign[i,-1]))},'source':'LIBRAS','row':i})
        return result
    sign_train,sign_test=sign_records(train_indices),sign_records(test_indices)
    out=ROOT/'artifacts'
    normalization={'speech':{'mean':mean.tolist(),'std':std.tolist()},
                   'sign':{'mean':sign_mean.tolist(),'std':sign_std.tolist()},
                   'sign_train_rows':train_indices,'sign_test_rows':test_indices}
    (out/'multimodal-normalization.json').write_text(json.dumps(normalization,indent=2)+'\n')
    from neural.gesture_benchmark import prepare_gesture
    gesture_train,gesture_test,_,_=prepare_gesture()
    return {'phase':(gesture_train,gesture_test),
            'speaker':(speech_train,speech_test),'movement':(sign_train,sign_test)}


def score(model,records,head):
    labels=model.heads[head]; confusion=np.zeros((len(labels),len(labels)),dtype=int)
    for r in records:
        p=model.forward(r['events'])[0][head]
        confusion[labels.index(r['targets'][head]),p.argmax()]+=1
    f1=[2*confusion[i,i]/max(1,confusion[i].sum()+confusion[:,i].sum()) for i in range(len(labels))]
    return {'accuracy':float(np.trace(confusion)/len(records)),'macro_f1':float(np.mean(f1)),
            'confusion_matrix':confusion.tolist()}


def run(epochs=40):
    tasks=prepare()
    model=SequenceContactModel.load(ROOT/'artifacts'/'gesture-sequence-model.npz')
    model.add_channel('speech',12); model.add_channel('sign',2); model.add_channel('sign_motion',2)
    model.add_head('speaker',[str(i) for i in range(1,10)])
    model.add_head('movement',[str(i) for i in range(1,16)])
    model.head_requirements={'phase':['left_hand','right_hand'],'speaker':['speech'],'movement':['sign']}
    initial={head:score(model,test,head) for head,(train,test) in tasks.items()}
    rng=np.random.default_rng(27)
    all_train=[r for train,test in tasks.values() for r in train]
    for epoch in range(epochs):
        order=rng.permutation(len(all_train)); losses=[]
        for start in range(0,len(all_train),16):
            losses.append(model.train_batch([all_train[i] for i in order[start:start+16]],.04,prediction_weight=.015))
        if epoch%5==0 or epoch==epochs-1:
            print(json.dumps({'epoch':epoch+1,'joint_train_loss':float(np.mean(losses))}),flush=True)
    report={'model':'shared multimodal recurrent neural contact model','epochs':epochs,
            'initialization':'trained gesture checkpoint, new speech/sign encoders and heads',
            'optimization_seed':27,'tasks':{}}
    for head,(train,test) in tasks.items():
        labels=[r['targets'][head] for r in train]; majority=max(set(labels),key=labels.count)
        report['tasks'][head]={'train_sequences':len(train),'test_sequences':len(test),
            'initial':initial[head],'final':score(model,test,head),
            'majority_baseline':float(np.mean([r['targets'][head]==majority for r in test]))}
    report['limitations']=['Independent datasets, not aligned cross-modal conversations.',
        'Speech task identifies known speakers from acoustic features; it does not transcribe words.',
        'LIBRAS task identifies 15 movement types from hand trajectories; it does not translate full signed language.',
        'LIBRAS split is stratified and duplicate-grouped, not signer-held-out: identity metadata unavailable.',
        'Gesture recording holdout is not a participant-held-out test.',
        'No claim of general human understanding or production readiness.']
    model.save(ROOT/'artifacts'/'multimodal-contact-model.npz')
    (ROOT/'artifacts'/'multimodal-benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
    config={'dimensions':model.dimensions,'hidden':model.hidden,'heads':model.heads,'head_requirements':model.head_requirements}
    (ROOT/'data'/'multimodal-config.json').write_text(json.dumps(config,indent=2)+'\n')
    with open(ROOT/'data'/'multimodal-train.jsonl','w') as f:
        for r in all_train: f.write(json.dumps(r)+'\n')
    with open(ROOT/'data'/'multimodal-test.jsonl','w') as f:
        for train,test in tasks.values():
            for r in test: f.write(json.dumps(r)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report

if __name__=='__main__': run()
