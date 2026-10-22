import tempfile
import unittest
import numpy as np
from neural.contact_model import NeuralContactModel
from neural.demo import dataset

class ContactTests(unittest.TestCase):
    def test_missing_channel_and_checkpoint(self):
        model = NeuralContactModel()
        x, masks, _ = dataset(8, 2)
        masks['body'][:] = 0
        expected = model.forward(x, masks)[0]
        x['body'][:] = np.nan
        np.testing.assert_array_equal(expected, model.forward(x, masks)[0])
        with tempfile.TemporaryDirectory() as directory:
            model.save(directory+'/model.npz')
            restored = NeuralContactModel.load(directory+'/model.npz')
            np.testing.assert_array_equal(expected, restored.forward(x, masks)[0])

    def test_expand_channel_preserves_learned_function(self):
        model = NeuralContactModel()
        x, masks, _ = dataset(8, 9)
        expected = model.forward(x, masks)[0]
        model.add_channel('environment', 7)
        x['environment'] = np.ones((8,7))
        masks['environment'] = np.ones(8)
        np.testing.assert_allclose(expected, model.forward(x,masks)[0], atol=1e-15)
        self.assertEqual(model.encode(x,masks).shape, (8,16))

    def test_gradients(self):
        model = NeuralContactModel(hidden=3)
        x, masks, y = dataset(6, 4)
        old = {k:v.copy() for k,v in model.weights.items()}
        model.train_step(x, masks, y, .01)
        gradients = {k:(old[k]-v)/.01 for k,v in model.weights.items()}
        model.weights = old
        for key in old:
            index = (0,) * old[key].ndim
            value = old[key][index]
            losses=[]
            for change in [1e-5, -1e-5]:
                old[key][index] = value+change
                p = model.forward(x,masks)[0]
                losses.append(-np.log(p[np.arange(len(y)),y]).mean())
            old[key][index] = value
            self.assertAlmostEqual(gradients[key][index], (losses[0]-losses[1])/2e-5, places=6)

if __name__ == '__main__': unittest.main()
