"""Multimodal recurrent neural model with learned relational messages.

All encoders, recurrent dynamics, message transforms, prediction decoders and named
output heads are trainable with backpropagation through time. No neural framework
or remote API is required. Numeric sequences retain time, missingness and context.
"""
import json
import numpy as np


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


class SequenceContactModel:
    def __init__(self, dimensions, hidden=32, heads=None, seed=7, head_requirements=None):
        if not dimensions or hidden < 1 or any(not isinstance(n, int) or n < 1 for n in dimensions.values()):
            raise ValueError('Positive channel dimensions and hidden size required')
        self.dimensions = dict(dimensions)
        self.hidden = hidden
        self.heads = {}
        self.head_requirements = dict(head_requirements or {})
        self.steps = 0
        self.rng = np.random.default_rng(seed)
        self.p = {}
        for name, size in self.dimensions.items():
            self._channel(name, size)
        self.p['recurrent'] = np.eye(hidden)*.6 + self.rng.normal(0,.02,(hidden,hidden))
        self.p['bias'] = np.zeros(hidden)
        self.p['message'] = self.rng.normal(0,1/np.sqrt(hidden),(hidden,hidden))
        for name, labels in (heads or {}).items():
            self.add_head(name, labels)

    def _channel(self, name, size):
        self.p['encode:'+name] = self.rng.normal(0,1/np.sqrt(size),(size,self.hidden))
        self.p['present:'+name] = np.zeros(self.hidden)
        self.p['decode:'+name] = self.rng.normal(0,.1,(self.hidden,size))
        self.p['decode_bias:'+name] = np.zeros(size)

    def add_channel(self, name, size):
        if name in self.dimensions or not isinstance(size,int) or size < 1:
            raise ValueError('New channel name and positive integer size required')
        self.dimensions[name] = size
        self._channel(name,size)

    def add_head(self, name, labels):
        labels = list(labels)
        if not labels or len(set(labels)) != len(labels) or not all(isinstance(v,str) for v in labels):
            raise ValueError('Unique nonempty string labels required')
        if name in self.heads:
            raise ValueError('Use expand_head to add labels')
        self.heads[name] = labels
        self.p['head:'+name] = self.rng.normal(0,.1,(self.hidden,len(labels)))
        self.p['head_bias:'+name] = np.zeros(len(labels))

    def expand_head(self, name, labels):
        new = [v for v in labels if v not in self.heads[name]]
        if len(set(new)) != len(new) or not all(isinstance(v,str) for v in new):
            raise ValueError('New labels must be unique strings')
        self.heads[name].extend(new)
        self.p['head:'+name] = np.concatenate([self.p['head:'+name], self.rng.normal(0,.1,(self.hidden,len(new)))],axis=1)
        self.p['head_bias:'+name] = np.concatenate([self.p['head_bias:'+name],np.zeros(len(new))])

    def _observations(self, event):
        values = event.get('observations',{})
        unknown = set(values)-set(self.dimensions)
        if unknown:
            raise ValueError('Unregistered channels: '+', '.join(sorted(unknown)))
        result={}
        for name, value in values.items():
            if value is None:
                continue
            x=np.asarray(value,dtype=float)
            if x.shape != (self.dimensions[name],) or not np.isfinite(x).all():
                raise ValueError('Invalid features: '+name)
            result[name]=x
        return result

    def forward(self, events, initial=None):
        if not events:
            raise ValueError('At least one contact event required')
        h=np.zeros(self.hidden) if initial is None else np.asarray(initial,dtype=float).copy()
        if h.shape != (self.hidden,) or not np.isfinite(h).all():
            raise ValueError('Invalid initial state')
        cache=[]
        for event in events:
            x=self._observations(event)
            context=np.asarray(event.get('context',np.zeros(self.hidden)),dtype=float)
            if context.shape != (self.hidden,) or not np.isfinite(context).all():
                raise ValueError('Invalid relational context')
            z=h@self.p['recurrent']+self.p['bias']+context@self.p['message']
            for name, value in x.items():
                z += value@self.p['encode:'+name]+self.p['present:'+name]
            next_h=np.tanh(z)
            cache.append((x,h,context,next_h))
            h=next_h
        observed = set().union(*(set(item[0]) for item in cache))
        predictions={name:softmax(h@self.p['head:'+name]+self.p['head_bias:'+name])
                     for name in self.heads if set(self.head_requirements.get(name, ())) <= observed}
        future={name:h@self.p['decode:'+name]+self.p['decode_bias:'+name] for name in self.dimensions}
        return predictions, future, h.copy(), cache

    def loss_and_grad(self, events, targets=None, prediction_weight=.1):
        targets=targets or {}
        pred, _, _, cache=self.forward(events)
        grads={k:np.zeros_like(v) for k,v in self.p.items()}
        dhs=[np.zeros(self.hidden) for _ in events]
        loss=0.
        for name,label in targets.items():
            if name not in self.heads or label not in self.heads[name]:
                raise ValueError('Unknown target head/label')
            if name not in pred:
                raise ValueError('Missing required evidence for output head '+name)
            index=self.heads[name].index(label)
            loss-=np.log(max(pred[name][index],1e-12))
            d=pred[name].copy(); d[index]-=1
            grads['head:'+name]+=np.outer(cache[-1][3],d)
            grads['head_bias:'+name]+=d
            dhs[-1]+=d@self.p['head:'+name].T
        pairs=[(t,name,value) for t in range(len(events)-1) for name,value in cache[t+1][0].items()]
        for t,name,value in pairs:
            estimate=cache[t][3]@self.p['decode:'+name]+self.p['decode_bias:'+name]
            scale=prediction_weight/max(1,len(pairs))/len(value)
            delta=(estimate-value)*scale
            loss+=.5*scale*np.sum((estimate-value)**2)
            grads['decode:'+name]+=np.outer(cache[t][3],delta)
            grads['decode_bias:'+name]+=delta
            dhs[t]+=delta@self.p['decode:'+name].T
        carry=np.zeros(self.hidden)
        for t in reversed(range(len(cache))):
            x,previous,context,h=cache[t]
            dz=(dhs[t]+carry)*(1-h*h)
            grads['recurrent']+=np.outer(previous,dz)
            grads['message']+=np.outer(context,dz)
            grads['bias']+=dz
            for name,value in x.items():
                grads['encode:'+name]+=np.outer(value,dz)
                grads['present:'+name]+=dz
            carry=dz@self.p['recurrent'].T
        return float(loss),grads

    def train_batch(self, records, learning_rate=.03, prediction_weight=.1):
        if not records or not 0 < learning_rate <= 1 or prediction_weight < 0:
            raise ValueError('Nonempty batch and valid training settings required')
        gradients={k:np.zeros_like(v) for k,v in self.p.items()}
        loss=0.
        for record in records:
            value,g=self.loss_and_grad(record['events'],record.get('targets'),prediction_weight)
            loss+=value
            for k in g: gradients[k]+=g[k]/len(records)
        norm=np.sqrt(sum(np.sum(g*g) for g in gradients.values()))
        scale=min(1.,5./max(norm,1e-12))
        for k in gradients: self.p[k]-=learning_rate*scale*gradients[k]
        self.steps+=1
        return loss/len(records)

    def save(self,path):
        metadata=json.dumps({'dimensions':self.dimensions,'hidden':self.hidden,'heads':self.heads,'steps':self.steps,'head_requirements':self.head_requirements})
        np.savez_compressed(path,metadata=metadata,**self.p)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as archive:
            meta=json.loads(str(archive['metadata']))
            model=cls(meta['dimensions'],meta['hidden'],meta['heads'],head_requirements=meta.get('head_requirements'))
            for k,expected in model.p.items():
                value=archive[k]
                if value.shape != expected.shape or not np.isfinite(value).all():
                    raise ValueError('Invalid checkpoint parameter '+k)
                model.p[k]=value.copy()
            model.steps=meta['steps']
        return model
