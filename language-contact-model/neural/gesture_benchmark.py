"""Reproducible whole-recording holdout on real UCI gesture observations.

Fit normalization on training recordings only. No random frame split. Class labels
are gesture phases, not emotions or meanings. This is a component evaluation.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from neural.sequence_model import SequenceContactModel

ROOT=Path(__file__).resolve().parents[1]
LABELS=['Rest','Preparation','Stroke','Hold','Retraction']
DIMENSIONS={'left_hand':3,'right_hand':3,'posture':6,'wrists':6,'motion':18,'timing':1}


def read(path):
    with open(path,newline='') as source:
        rows=list(csv.DictReader(source))
    names=list(rows[0])[:18]
    raw=np.array([[float(row[k]) for k in names] for row in rows])
    t=np.array([float(row['timestamp']) for row in rows])
    labels=[row['phase'].strip() for row in rows]
    if set(labels)-set(LABELS): raise ValueError('Unrecognized gesture phases')
    # Body-relative coordinates remove recording-position offsets.
    relative=(raw.reshape(-1,6,3)-raw[:,9:12,None].transpose(0,2,1)).reshape(-1,18)
    motion=np.vstack([np.zeros(18),np.diff(relative,axis=0)])
    dt=np.r_[0,np.diff(t)]
    if (dt<0).any(): raise ValueError('Out-of-order recording timestamps')
    dt=dt/np.median(dt[dt>0])
    return np.column_stack([relative,motion,dt]), labels


def windows(x, labels, recording, mean, std, length=12, stride=6):
    x=(x-mean)/std
    result=[]
    for end in range(length,len(x)+1,stride):
        events=[]
        for row in x[end-length:end]:
            events.append({'observations':{'left_hand':row[:3].tolist(),'right_hand':row[3:6].tolist(),
                'posture':row[6:12].tolist(),'wrists':row[12:18].tolist(),
                'motion':row[18:36].tolist(),'timing':row[36:].tolist()}})
        result.append({'events':events,'targets':{'phase':labels[end-1]},'recording':recording})
    return result


def metrics(model, data, last_only=False):
    predicted=[]; actual=[]
    for r in data:
        events=r['events'][-1:] if last_only else r['events']
        p=model.forward(events)[0]['phase']
        predicted.append(int(p.argmax())); actual.append(LABELS.index(r['targets']['phase']))
    confusion=np.zeros((len(LABELS),len(LABELS)),dtype=int)
    for a,p in zip(actual,predicted): confusion[a,p]+=1
    f1=[]
    for i in range(len(LABELS)):
        f1.append(2*confusion[i,i]/max(1,confusion[i,:].sum()+confusion[:,i].sum()))
    return {'accuracy':float(np.mean(np.array(actual)==predicted)), 'macro_f1':float(np.mean(f1)),
            'confusion_matrix':confusion.tolist()}


def prepare_gesture():
    directory=ROOT/'data'/'gesture'
    recordings={name:read(directory/(name+'_raw.csv')) for name in ['a1','a2','a3']}
    features=np.concatenate([recordings[n][0] for n in ['a1','a2']])
    mean=features.mean(0); std=features.std(0); std[std<1e-6]=1
    train=[]
    for name in ['a1','a2']:
        x,y=recordings[name]; train+=windows(x,y,name,mean,std)
    test=windows(*recordings['a3'],'a3',mean,std)
    return train, test, mean, std


def run(epochs=45):
    directory=ROOT/'data'/'gesture'
    train,test,mean,std=prepare_gesture()
    model=SequenceContactModel(DIMENSIONS,hidden=24,heads={'phase':LABELS})
    initial=metrics(model,test)
    rng=np.random.default_rng(7)
    for epoch in range(epochs):
        order=rng.permutation(len(train)); losses=[]
        for start in range(0,len(train),16):
            batch=[train[i] for i in order[start:start+16]]
            losses.append(model.train_batch(batch,learning_rate=.035,prediction_weight=.03))
        if epoch%5==0 or epoch==epochs-1:
            print(json.dumps({'epoch':epoch+1,'train_loss':float(np.mean(losses))}),flush=True)
    # Final evaluation only; no checkpoint selection using held-out recording.
    final=metrics(model,test); ablation=metrics(model,test,last_only=True)
    labels=[r['targets']['phase'] for r in train]
    majority=max(LABELS,key=labels.count)
    baseline=np.mean([r['targets']['phase']==majority for r in test])
    out=ROOT/'artifacts'; out.mkdir(exist_ok=True)
    model.save(out/'gesture-sequence-model.npz')
    normalization={'mean':mean.tolist(),'std':std.tolist(),'feature_order':list(DIMENSIONS),
                   'window_length':12,'stride':6,'train_recordings':['a1','a2'],'test_recordings':['a3']}
    (out/'gesture-normalization.json').write_text(json.dumps(normalization,indent=2)+'\n')
    report={'dataset':'UCI Gesture Phase Segmentation, mirrored raw recordings',
            'source':'https://doi.org/10.24432/C5Z32C','license':'CC BY 4.0',
            'seed':7,'epochs':epochs,'train_windows':len(train),'test_windows':len(test),
            'train_recordings':['a1','a2'],'test_recordings':['a3'],'labels':LABELS,
            'initial':initial,'final':final,'last_frame_only_ablation':ablation,
            'training_majority_label':majority,'majority_baseline_accuracy':float(baseline),
            'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob('*.csv')},
            'limitations':['Three recordings only; not participant-independent evaluation.',
                          'Gesture phase identification is not semantic gesture understanding.',
                          'Overlapping windows within each split; no recording appears in both splits.',
                          'No raw video, speech, or sign-language training in this checkpoint.']}
    (out/'gesture-benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
    with open(directory/'train.jsonl','w') as f:
        for r in train: f.write(json.dumps(r)+'\n')
    with open(directory/'test.jsonl','w') as f:
        for r in test: f.write(json.dumps(r)+'\n')
    (directory/'config.json').write_text(json.dumps({'dimensions':DIMENSIONS,'hidden':24,'heads':{'phase':LABELS}},indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    return report

if __name__=='__main__': run()
