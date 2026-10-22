"""Persistent stateful inference and user-correction training with replay."""
import hashlib
import json
import numpy as np
from neural.memory import ContactMemory


def model_version(model):
    digest=hashlib.sha256(json.dumps({'dimensions':model.dimensions,'heads':model.heads,'requirements':model.head_requirements},sort_keys=True).encode())
    for key in sorted(model.p):
        digest.update(key.encode()); digest.update(model.p[key].tobytes())
    return digest.hexdigest()


class ContactRuntime:
    def __init__(self, model, memory):
        self.model, self.memory = model, memory

    def observe(self, session, participant, event, threshold=.7, related_sessions=()):
        if not 0 <= threshold <= 1:
            raise ValueError('Threshold must be between zero and one')
        version=model_version(self.model)
        history=self.memory.history(session,participant)
        # Replay history against current weights after learning, rather than reuse stale state.
        initial=None
        if history:
            if history[-1]['version']==version:
                initial=history[-1]['state']
            else:
                initial=self.model.forward([r['event'] for r in history])[2]
        event=dict(event)
        if related_sessions:
            states=[]
            for related in related_sessions:
                # Context access is restricted to this participant's own sessions.
                related_history=self.memory.history(related,participant)
                if related_history:
                    states.append(self.model.forward([r['event'] for r in related_history])[2])
            if states: event['context']=np.mean(states,axis=0).tolist()
        probabilities,future,state,_=self.model.forward([event],initial)
        event_id=self.memory.append(session,participant,event,state,version)
        output={}
        for head,p in probabilities.items():
            index=int(np.argmax(p)); score=float(p[index])
            output[head]={'candidate':self.model.heads[head][index], 'score':score,
                          'needs_clarification':score<threshold,
                          'probabilities':dict(zip(self.model.heads[head],map(float,p)))}
        return {'event_id':event_id,'model_version':version,'outputs':output,
                'next_observations':{k:v.tolist() for k,v in future.items()},
                'observed_channels':[k for k,v in event.get('observations',{}).items() if v is not None],
                'missing_channels':[k for k in self.model.dimensions if event.get('observations',{}).get(k) is None],
                'state':state.tolist(), 'score_status':'uncalibrated'}

    def feedback(self,session,participant,event_id,head,label):
        if head not in self.model.heads or label not in self.model.heads[head]:
            raise ValueError('Register this output head/label before providing feedback')
        self.memory.correct(session,participant,event_id,head,label)

    def adapt(self,session,participant,replay_records=(),steps=20,learning_rate=.03):
        if not isinstance(steps,int) or steps < 1:
            raise ValueError('Positive training step count required')
        records=self.memory.training_records(session,participant)
        if not records:
            raise ValueError('No verified feedback available for training')
        records=records+list(replay_records)
        losses=[self.model.train_batch(records,learning_rate) for _ in range(steps)]
        return {'training_steps':steps,'first_loss':losses[0],'last_loss':losses[-1],
                'model_version':model_version(self.model),'replay_records':len(replay_records)}
