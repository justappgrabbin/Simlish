import tempfile
import unittest
from pathlib import Path
import numpy as np
from neural.sequence_model import SequenceContactModel
from neural.memory import ContactMemory
from neural.runtime import ContactRuntime
from neural.adapters import text_events, landmark_events


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.model=SequenceContactModel({'body':2,'sign':2},hidden=4,heads={'intent':['a','b']})
        self.events=[{'observations':{'body':[.2,-.3],'sign':[.7,.2]},'context':[.1,.3,-.2,.4]},
                     {'observations':{'sign':[-.1,.8]}}]

    def test_all_gradients_through_time(self):
        loss,g=self.model.loss_and_grad(self.events,{'intent':'a'})
        for key,value in self.model.p.items():
            for index in np.ndindex(value.shape):
                old=value[index]; losses=[]
                for change in [1e-5,-1e-5]:
                    value[index]=old+change
                    losses.append(self.model.loss_and_grad(self.events,{'intent':'a'})[0])
                value[index]=old
                self.assertAlmostEqual(g[key][index],(losses[0]-losses[1])/2e-5,places=6,msg=key)

    def test_checkpoint_streaming_and_channel_expansion(self):
        self.model.train_batch([{'events':self.events,'targets':{'intent':'b'}}])
        full=self.model.forward(self.events)
        initial=self.model.forward(self.events[:1])[2]
        np.testing.assert_allclose(full[2],self.model.forward(self.events[1:],initial)[2])
        self.model.add_channel('touch',5)
        np.testing.assert_allclose(full[2],self.model.forward(self.events)[2])
        self.model.expand_head('intent',['c'])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.npz'; self.model.save(path)
            loaded=SequenceContactModel.load(path)
            self.assertEqual(loaded.heads['intent'],['a','b','c'])
            for key in self.model.p: np.testing.assert_array_equal(self.model.p[key],loaded.p[key])

    def test_memory_restart_isolation_and_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'memory.db'
            memory=ContactMemory(path); session=memory.create_session('alice')
            runtime=ContactRuntime(self.model,memory)
            first=runtime.observe(session,'alice',self.events[0]); memory.close()
            memory=ContactMemory(path); runtime=ContactRuntime(self.model,memory)
            second=runtime.observe(session,'alice',self.events[1])
            np.testing.assert_allclose(second['state'],self.model.forward(self.events)[2])
            with self.assertRaises(ValueError): runtime.observe(session,'bob',self.events[1])
            runtime.feedback(session,'alice',second['event_id'],'intent','b')
            old=self.model.p['head:intent'].copy()
            report=runtime.adapt(session,'alice',steps=4)
            self.assertFalse(np.array_equal(old,self.model.p['head:intent']))
            self.assertLess(report['last_loss'],report['first_loss'])
            # Updated weights trigger replay of historical state.
            third=runtime.observe(session,'alice',self.events[0])
            np.testing.assert_allclose(third['state'],self.model.forward(self.events+[self.events[0]])[2])
            memory.close()

    def test_evidence_gates_and_forecasting_learning(self):
        self.model.head_requirements={'intent':['sign']}
        missing=[{'observations':{'body':[.2,.1]}}]
        self.assertNotIn('intent',self.model.forward(missing)[0])
        with self.assertRaises(ValueError): self.model.loss_and_grad(missing,{'intent':'a'})
        record={'events':self.events,'targets':{}}
        before=self.model.loss_and_grad(self.events)[0]
        for _ in range(20): self.model.train_batch([record],learning_rate=.1)
        self.assertLess(self.model.loss_and_grad(self.events)[0],before)

    def test_unknown_vs_zero_and_validation(self):
        self.model.p['present:body'][:]=.5
        missing=self.model.forward([{'observations':{}}])[2]
        observed=self.model.forward([{'observations':{'body':[0,0]}}])[2]
        self.assertFalse(np.array_equal(missing,observed))
        for event in [{'observations':{'mystery':[1]}},{'observations':{'body':[1]}},{'observations':{'body':[float('nan'),0]}}]:
            with self.assertRaises(ValueError): self.model.forward([event])
        self.assertEqual(len(text_events('é')),2)
        self.assertEqual(len(landmark_events(np.ones((3,2,3)))),3)

if __name__=='__main__': unittest.main()
