"""Trainable multimodal contact prototype; NumPy CPU implementation.
Inputs are measured features, not raw-video/sign-language recognition.
"""
from pathlib import Path
import numpy as np


class NeuralContactModel:
    def __init__(self, dimensions=None, hidden=16, classes=3, seed=7):
        self.dimensions = dimensions or {'speech': 4, 'body': 4, 'sign': 4, 'history': 4}
        self.hidden, self.classes = hidden, classes
        rng = np.random.default_rng(seed)
        self.weights = {}
        for name, size in self.dimensions.items():
            self.weights[name + '_W'] = rng.normal(0, .3, (size, hidden))
            self.weights[name + '_b'] = np.zeros(hidden)
        size = len(self.dimensions) * (hidden + 1)
        self.weights['fusion_W'] = rng.normal(0, .2, (size, hidden))
        self.weights['fusion_b'] = np.zeros(hidden)
        self.weights['output_W'] = rng.normal(0, .2, (hidden, classes))
        self.weights['output_b'] = np.zeros(classes)

    def add_channel(self, name, size, seed=0):
        """Expand input space without changing existing-channel predictions.

        New channel starts disconnected at fusion; subsequent training learns its use.
        """
        if name in self.dimensions or not name or size < 1:
            raise ValueError('Channel must have a new name and positive feature size')
        rng = np.random.default_rng(seed)
        self.dimensions[name] = size
        self.weights[name+'_W'] = rng.normal(0, .3, (size, self.hidden))
        self.weights[name+'_b'] = np.zeros(self.hidden)
        self.weights['fusion_W'] = np.concatenate([
            self.weights['fusion_W'], np.zeros((self.hidden+1, self.hidden))], axis=0)

    def encode(self, observations, masks):
        """Continuous learned contact representation, independent of output labels."""
        return self.forward(observations, masks)[1][2].copy()

    def forward(self, observations, masks):
        if set(observations) != set(self.dimensions) or set(masks) != set(self.dimensions):
            raise ValueError('Supply all channels with explicit presence masks')
        encodings, parts = {}, []
        batch = None
        for name, size in self.dimensions.items():
            x = np.asarray(observations[name], dtype=float)
            m = np.asarray(masks[name], dtype=float).reshape(-1, 1)
            if x.ndim != 2 or x.shape[1] != size or len(m) != len(x):
                raise ValueError('Invalid feature or mask shape: ' + name)
            if batch is not None and batch != len(x):
                raise ValueError('Channel batch sizes differ')
            batch = len(x)
            if not np.isin(m, [0, 1]).all():
                raise ValueError('Presence masks must be binary')
            x = np.where(m.astype(bool), x, 0.)
            if not np.isfinite(x).all():
                raise ValueError('Observed features must be finite')
            h = np.tanh(x @ self.weights[name+'_W'] + self.weights[name+'_b'])
            encodings[name] = (x, m, h)
            parts.extend([h * m, m])
        joined = np.concatenate(parts, axis=1)
        fused = np.tanh(joined @ self.weights['fusion_W'] + self.weights['fusion_b'])
        logits = fused @ self.weights['output_W'] + self.weights['output_b']
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities = exp / exp.sum(axis=1, keepdims=True)
        return probabilities, (encodings, joined, fused)

    def train_step(self, observations, masks, labels, learning_rate=.1):
        p, (encodings, joined, fused) = self.forward(observations, masks)
        labels = np.asarray(labels)
        if labels.shape != (len(p),) or not np.issubdtype(labels.dtype, np.integer) or ((labels < 0) | (labels >= self.classes)).any():
            raise ValueError('Invalid class labels')
        loss = -np.log(p[np.arange(len(p)), labels] + 1e-12).mean()
        delta = p.copy()
        delta[np.arange(len(p)), labels] -= 1
        delta /= len(p)
        gradients = {'output_W': fused.T @ delta, 'output_b': delta.sum(axis=0)}
        df = (delta @ self.weights['output_W'].T) * (1 - fused*fused)
        gradients.update(fusion_W=joined.T @ df, fusion_b=df.sum(axis=0))
        dj = df @ self.weights['fusion_W'].T
        offset = 0
        for name in self.dimensions:
            x, m, h = encodings[name]
            dh = dj[:, offset:offset+self.hidden] * m * (1-h*h)
            gradients[name+'_W'] = x.T @ dh
            gradients[name+'_b'] = dh.sum(axis=0)
            offset += self.hidden + 1
        for key in self.weights:
            self.weights[key] -= learning_rate * gradients[key]
        return float(loss)

    def predict(self, observations, masks, threshold=.7):
        p, _ = self.forward(observations, masks)
        return np.where(p.max(axis=1) >= threshold, p.argmax(axis=1), -1), p

    def save(self, path):
        np.savez(path, **self.weights, names=np.array(list(self.dimensions)),
                 sizes=np.array(list(self.dimensions.values())), hidden=self.hidden, classes=self.classes)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            model = cls(dict(zip(data['names'].tolist(), data['sizes'].tolist())),
                        int(data['hidden']), int(data['classes']))
            model.weights = {key: data[key].copy() for key in model.weights}
        return model
