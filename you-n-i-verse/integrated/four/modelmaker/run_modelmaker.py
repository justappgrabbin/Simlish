# Box-side harness (builder's). Imports her UNMODIFIED ModelMaker app.py (justappgrabbin/Stellarproximology-lab
# de9d32a, sha256 d2109b98...) and calls its train_model() on her Drive docs. No Gradio server, no HF push.
# Output: the trained BPE vocabulary + merges become ModelMaker's particles (semantic units it learned).
import sys, json, zipfile, shutil, hashlib, torch
from pathlib import Path
sys.path.insert(0, "/workspace/chassis-work/src/Stellarproximology-lab/Modelmaker-repair")
import app
torch.manual_seed(0)
here = Path(__file__).parent; docs = Path(sys.argv[1])
names = ["Synthia_Semantic_Swarm_Master_Spec_v3.docx.txt", "REASSEMBLY_MASTER_BUILD.md", "Cynthia_core.pdf.pdftotext.txt"]
corpus = "\n\n".join((docs / n).read_text() for n in names)
summary, zp = app.train_model(model_name="swarm-docs", pasted_text=corpus, files=None, vocab_size=1024, context_length=32,
    layers=2, heads=4, embedding_size=64, epochs=2, batch_size=16, learning_rate=0.003, push_to_hub=False, progress=lambda *a, **k: None)
print(summary)
out = Path("/workspace/scratch/four-mm"); shutil.rmtree(out, ignore_errors=True); zipfile.ZipFile(zp).extractall(out)
vocab = json.loads((out / "vocab.json").read_text()); merges = (out / "merges.txt").read_text().splitlines()[1:]
build = json.loads((out / "modelmaker_build.json").read_text())
toks = sorted(vocab.items(), key=lambda kv: kv[1])
json.dump({"source": "ModelMaker app.train_model() (unmodified) on 3 Drive docs", "corpusSha256": hashlib.sha256(corpus.encode()).hexdigest(),
  "summary": summary, "build": build, "vocabSize": len(vocab), "merges": len(merges),
  "learnedTokens": [{"id": i, "token": t} for t, i in toks if i >= 256 and not t.startswith("<")]}, open(here / "modelmaker-run.json", "w"), indent=1)
