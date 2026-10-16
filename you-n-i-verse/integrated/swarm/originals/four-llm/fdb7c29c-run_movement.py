# Harness only: imports the unmodified Modelmaker app.py from the git clone and calls its train_model().
# No Gradio server is launched; no Hugging Face push (push_to_hub=False, no HF_TOKEN).
import sys, json, shutil, zipfile, torch
from pathlib import Path
sys.path.insert(0, "/workspace/chassis-work/src/Stellarproximology-lab/Modelmaker-repair")
import app  # Modelmaker
torch.manual_seed(0)
here = Path(__file__).parent
corpus = (here / "movement-corpus.txt").read_text()
summary, zip_path = app.train_model(
    model_name="movement", pasted_text=corpus, files=None,
    vocab_size=512, context_length=32, layers=2, heads=4, embedding_size=64,
    epochs=20, batch_size=4, learning_rate=0.003, push_to_hub=False,
    progress=lambda *a, **k: None,
)
print("=== Modelmaker train_model() summary ===")
print(summary)
if not zip_path:
    sys.exit(1)
out = here / "movement-model"
shutil.rmtree(out, ignore_errors=True)
zipfile.ZipFile(zip_path).extractall(out)
shutil.copy(zip_path, here / "movement.zip")
print("=== files produced ===")
for p in sorted(out.rglob("*")):
    if p.is_file(): print(p.relative_to(out), p.stat().st_size)
# Load the built model back (what Modelmaker's save_pretrained output is for) and generate.
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
tok = GPT2TokenizerFast.from_pretrained(out); model = GPT2LMHeadModel.from_pretrained(out).eval()
for prompt in ["Movement", "naming is"]:
    ids = tok(prompt, return_tensors="pt").input_ids
    g = model.generate(ids, max_new_tokens=25, do_sample=False, pad_token_id=tok.pad_token_id)
    print(f"=== generate (greedy) prompt={prompt!r} ===")
    print(repr(tok.decode(g[0], skip_special_tokens=True)))
