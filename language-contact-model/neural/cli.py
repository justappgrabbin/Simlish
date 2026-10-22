"""Local CLI for configurable training, inference and feedback. JSON/JSONL only."""
import argparse
import json
from pathlib import Path
import numpy as np
from neural.sequence_model import SequenceContactModel
from neural.memory import ContactMemory
from neural.runtime import ContactRuntime


def records(path):
    with open(path) as source:
        return [json.loads(line) for line in source if line.strip()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    train=sub.add_parser('train'); train.add_argument('--config',required=True); train.add_argument('--data',required=True)
    train.add_argument('--output',required=True); train.add_argument('--epochs',type=int,default=50)
    train.add_argument('--batch-size',type=int,default=16); train.add_argument('--resume'); train.add_argument('--seed',type=int,default=7)
    train.add_argument('--learning-rate',type=float,default=.03)
    predict=sub.add_parser('predict'); predict.add_argument('--model',required=True); predict.add_argument('--events',required=True)
    session=sub.add_parser('session'); session.add_argument('--db',required=True); session.add_argument('--participant',required=True)
    for name in ['observe','feedback','adapt']:
        p=sub.add_parser(name); p.add_argument('--model',required=True); p.add_argument('--db',required=True)
        p.add_argument('--session',required=True); p.add_argument('--participant',required=True)
        if name=='observe': p.add_argument('--event',required=True)
        if name=='feedback':
            p.add_argument('--event-id',required=True); p.add_argument('--head',required=True); p.add_argument('--label',required=True)
        if name=='adapt':
            p.add_argument('--output',required=True); p.add_argument('--steps',type=int,default=20); p.add_argument('--replay')
    args=parser.parse_args()
    if args.command=='train':
        if args.epochs<1 or args.batch_size<1: parser.error('Positive epochs and batch size required')
        config=json.loads(Path(args.config).read_text())
        model=SequenceContactModel.load(args.resume) if args.resume else SequenceContactModel(**config)
        data=records(args.data)
        if not data: parser.error('Training data is empty')
        rng=np.random.default_rng(args.seed)
        for epoch in range(args.epochs):
            indices=rng.permutation(len(data)); losses=[]
            for start in range(0,len(data),args.batch_size):
                losses.append(model.train_batch([data[i] for i in indices[start:start+args.batch_size]],args.learning_rate))
            print(json.dumps({'epoch':epoch+1,'loss':float(np.mean(losses))}),flush=True)
        model.save(args.output)
    elif args.command=='predict':
        model=SequenceContactModel.load(args.model)
        p,future,state,_=model.forward(records(args.events))
        print(json.dumps({'outputs':{name:dict(zip(model.heads[name],map(float,value))) for name,value in p.items()},
                          'next_observations':{k:v.tolist() for k,v in future.items()},'state':state.tolist()}))
    else:
        memory=ContactMemory(args.db)
        try:
            if args.command=='session': print(memory.create_session(args.participant)); return
            model=SequenceContactModel.load(args.model); runtime=ContactRuntime(model,memory)
            if args.command=='observe':
                print(json.dumps(runtime.observe(args.session,args.participant,json.loads(Path(args.event).read_text()))))
            elif args.command=='feedback':
                runtime.feedback(args.session,args.participant,args.event_id,args.head,args.label); print('{"saved":true}')
            else:
                result=runtime.adapt(args.session,args.participant,records(args.replay) if args.replay else (),args.steps)
                model.save(args.output); print(json.dumps(result))
        finally: memory.close()

if __name__=='__main__': main()
