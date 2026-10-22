"""Synthetic context-sensitive learning benchmark, not human validation."""
import json
from pathlib import Path
import numpy as np
from neural.contact_model import NeuralContactModel


def dataset(n, seed):
    rng = np.random.default_rng(seed)
    x = {name: rng.normal(size=(n, 4)) for name in ('speech', 'body', 'sign', 'history')}
    masks = {name: (rng.random(n) > .2).astype(float) for name in x}
    # Same speech can imply different targets given body, signing and history.
    score = sum(x[name][:, 0] * masks[name] * weight for name, weight in
                [('speech', 1.), ('body', 1.5), ('sign', 1.5), ('history', 2.)])
    y = np.digitize(score, [-.8, .8])
    return x, masks, y


def run():
    model = NeuralContactModel()
    x, masks, y = dataset(1600, 12)
    tx, tm, ty = dataset(600, 93)
    before = model.forward(tx, tm)[0]
    initial_weights = {k: v.copy() for k, v in model.weights.items()}
    for _ in range(500):
        loss = model.train_step(x, masks, y, .15)
    p = model.forward(tx, tm)[0]
    speech_masks = {k: v if k == 'speech' else np.zeros_like(v) for k, v in tm.items()}
    output = Path('artifacts'); output.mkdir(exist_ok=True)
    model.save(output / 'contact-model.npz')
    report = {'dataset': 'synthetic feature-level contact task', 'seed': 7,
              'initial_accuracy': float((before.argmax(1) == ty).mean()),
              'held_out_accuracy': float((p.argmax(1) == ty).mean()),
              'majority_baseline': float(np.bincount(ty).max()/len(ty)),
              'speech_only_ablation_accuracy': float((model.forward(tx, speech_masks)[0].argmax(1) == ty).mean()),
              'final_training_loss': loss,
              'changed_parameter_arrays': sum(not np.array_equal(v, initial_weights[k]) for k,v in model.weights.items()),
              'limitation': 'No raw perception, real human evaluation, or calibrated confidence established.'}
    (output / 'benchmark.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return report

if __name__ == '__main__': run()
