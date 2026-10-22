"""Run the saved shared model on one held-out example from each real dataset."""
import json
import tempfile
from pathlib import Path
from neural.multimodal_benchmark import prepare, ROOT
from neural.sequence_model import SequenceContactModel
from neural.memory import ContactMemory
from neural.runtime import ContactRuntime


def run():
    model=SequenceContactModel.load(ROOT/'artifacts'/'multimodal-contact-model.npz')
    tasks=prepare(); result={}
    with tempfile.TemporaryDirectory() as directory:
        memory=ContactMemory(Path(directory)/'demo.db')
        runtime=ContactRuntime(model,memory)
        for head,(train,test) in tasks.items():
            record=test[0]; session=memory.create_session('demo-'+head)
            for event in record['events']:
                response=runtime.observe(session,'demo-'+head,event)
            result[head]={'expected':record['targets'][head],'prediction':response['outputs'].get(head),
                          'persisted_events':len(memory.history(session,'demo-'+head))}
        memory.close()
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__': run()
