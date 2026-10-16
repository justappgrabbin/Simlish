# Modelmaker Repair

Non-destructive repair package for the Hugging Face Space `stellarproximology/Modelmaker`.

## Install

1. Copy `app.py` and `requirements.txt` into the **root** of the existing Hugging Face Modelmaker Space.
2. Do not delete existing Modelmaker files.
3. If Hugging Face still reports **No application file**, use the YAML in `README_FRONTMATTER.txt` at the very top of the Space's existing `README.md`.
4. Optional: add a Hugging Face write token as a Space secret named `HF_TOKEN` to allow Modelmaker to publish trained models.

The app can ingest training text/files, train a compact GPT-style causal language model, export it in Hugging Face `save_pretrained()` format, and return the built model as a ZIP.
