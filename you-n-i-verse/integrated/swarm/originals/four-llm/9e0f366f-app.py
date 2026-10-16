from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import traceback
import zipfile
from pathlib import Path
from typing import Iterable

import gradio as gr
import torch
from huggingface_hub import HfApi
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

APP_TITLE = "Modelmaker"
DEFAULT_NAMESPACE = "stellarproximology"


def _read_uploads(files: Iterable | None) -> str:
    chunks: list[str] = []
    if not files:
        return ""

    for item in files:
        path = getattr(item, "name", None) or str(item)
        p = Path(path)
        if not p.exists() or not p.is_file():
            continue

        suffix = p.suffix.lower()
        try:
            if suffix in {".txt", ".md", ".py", ".js", ".ts", ".html", ".css", ".csv"}:
                chunks.append(p.read_text(encoding="utf-8", errors="ignore"))
            elif suffix == ".json":
                obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                chunks.append(json.dumps(obj, ensure_ascii=False, indent=2))
            elif suffix == ".jsonl":
                lines = []
                for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        lines.append(json.dumps(json.loads(line), ensure_ascii=False))
                    except json.JSONDecodeError:
                        lines.append(line)
                chunks.append("\n".join(lines))
        except Exception as exc:
            chunks.append(f"\n[Skipped {p.name}: {exc}]\n")

    return "\n\n".join(chunks)


def _dataset_text(pasted_text: str, files: Iterable | None) -> str:
    pieces = []
    if pasted_text and pasted_text.strip():
        pieces.append(pasted_text.strip())
    upload_text = _read_uploads(files)
    if upload_text.strip():
        pieces.append(upload_text.strip())
    return "\n\n".join(pieces)


def _safe_name(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_." else "-" for c in (name or "").strip())
    cleaned = cleaned.strip("-._")
    return cleaned or "modelmaker-model"


def _build_tokenizer(text_path: Path, output_dir: Path, vocab_size: int) -> GPT2TokenizerFast:
    tokenizer_dir = output_dir / "tokenizer_build"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    trainer = ByteLevelBPETokenizer()
    trainer.train(
        files=[str(text_path)],
        vocab_size=max(256, int(vocab_size)),
        min_frequency=1,
        special_tokens=["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"],
    )
    trainer.save_model(str(tokenizer_dir))

    # Load from the saved vocab.json/merges.txt via from_pretrained. transformers 5.x
    # no longer accepts vocab_file=/merges_file= in the constructor (they are silently
    # ignored, leaving an empty vocabulary); from_pretrained works on 4.x and 5.x.
    tokenizer = GPT2TokenizerFast.from_pretrained(
        str(tokenizer_dir),
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        unk_token="<|unk|>",
        pad_token="<|pad|>",
    )
    return tokenizer


def _make_sequences(token_ids: list[int], context_length: int) -> torch.Tensor:
    context_length = max(16, int(context_length))
    usable = (len(token_ids) // (context_length + 1)) * (context_length + 1)
    if usable < context_length + 1:
        raise ValueError(
            f"Not enough training text after tokenization. Need at least {context_length + 1} tokens; got {len(token_ids)}."
        )
    data = torch.tensor(token_ids[:usable], dtype=torch.long)
    return data.view(-1, context_length + 1)


def _zip_dir(directory: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in directory.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(directory))


def _push_model(model_dir: Path, model_name: str) -> str:
    token = os.getenv("HF_TOKEN", "").strip()
    if not token:
        return "HF_TOKEN is not set, so the model was built locally but not pushed."

    repo_id = model_name if "/" in model_name else f"{DEFAULT_NAMESPACE}/{model_name}"
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=str(model_dir), repo_id=repo_id, repo_type="model")
    return f"Pushed to https://huggingface.co/{repo_id}"


def train_model(
    model_name: str,
    pasted_text: str,
    files,
    vocab_size: int,
    context_length: int,
    layers: int,
    heads: int,
    embedding_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    push_to_hub: bool,
    progress=gr.Progress(track_tqdm=False),
):
    work_root = Path(tempfile.mkdtemp(prefix="modelmaker-"))
    try:
        name = _safe_name(model_name)
        model_dir = work_root / name
        model_dir.mkdir(parents=True, exist_ok=True)

        text = _dataset_text(pasted_text, files)
        if len(text.strip()) < 200:
            raise ValueError("Add more training material. Modelmaker needs at least about 200 characters to begin.")

        text_path = work_root / "corpus.txt"
        text_path.write_text(text, encoding="utf-8")

        progress(0.08, desc="Training tokenizer")
        tokenizer = _build_tokenizer(text_path, model_dir, int(vocab_size))
        tokenizer.save_pretrained(model_dir)

        ids = tokenizer.encode(text, add_special_tokens=False)
        sequences = _make_sequences(ids, int(context_length))

        if int(embedding_size) % int(heads) != 0:
            raise ValueError("Embedding size must divide evenly by the number of attention heads.")

        config = GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=int(context_length),
            n_ctx=int(context_length),
            n_embd=int(embedding_size),
            n_layer=int(layers),
            n_head=int(heads),
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = GPT2LMHeadModel(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))

        total_params = sum(p.numel() for p in model.parameters())
        num_batches = max(1, math.ceil(len(sequences) / int(batch_size)))
        losses: list[float] = []

        progress(0.16, desc=f"Training {total_params:,} parameters on {device.type}")
        model.train()
        for epoch in range(int(epochs)):
            permutation = torch.randperm(len(sequences))
            epoch_loss = 0.0
            epoch_steps = 0

            for start in range(0, len(sequences), int(batch_size)):
                idx = permutation[start : start + int(batch_size)]
                batch = sequences[idx].to(device)
                x = batch[:, :-1]
                y = batch[:, 1:]

                optimizer.zero_grad(set_to_none=True)
                out = model(input_ids=x, labels=y)
                loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += float(loss.detach().cpu())
                epoch_steps += 1

                completed = epoch * num_batches + epoch_steps
                total_steps = max(1, int(epochs) * num_batches)
                progress(0.16 + 0.68 * (completed / total_steps), desc=f"Epoch {epoch + 1}/{int(epochs)}")

            losses.append(epoch_loss / max(1, epoch_steps))

        progress(0.88, desc="Saving Hugging Face model")
        model.save_pretrained(model_dir, safe_serialization=True)
        tokenizer.save_pretrained(model_dir)

        model_card = f"""# {name}\n\nBuilt with **stellarproximology/Modelmaker**.\n\n- Architecture: GPT-2 style causal language model\n- Parameters: {total_params:,}\n- Context length: {int(context_length)}\n- Layers: {int(layers)}\n- Attention heads: {int(heads)}\n- Embedding size: {int(embedding_size)}\n- Vocabulary size: {len(tokenizer)}\n- Epochs: {int(epochs)}\n- Final training loss: {losses[-1]:.6f}\n\nThis model was trained on user-supplied material. Review its outputs before production use.\n"""
        (model_dir / "README.md").write_text(model_card, encoding="utf-8")

        build_meta = {
            "name": name,
            "parameters": total_params,
            "device": device.type,
            "context_length": int(context_length),
            "layers": int(layers),
            "heads": int(heads),
            "embedding_size": int(embedding_size),
            "vocab_size": len(tokenizer),
            "epochs": int(epochs),
            "loss_history": losses,
        }
        (model_dir / "modelmaker_build.json").write_text(json.dumps(build_meta, indent=2), encoding="utf-8")

        push_message = "Push disabled."
        if push_to_hub:
            progress(0.93, desc="Pushing to Hugging Face")
            push_message = _push_model(model_dir, name)

        zip_path = work_root / f"{name}.zip"
        _zip_dir(model_dir, zip_path)
        progress(1.0, desc="Done")

        summary = (
            f"### Built {name}\n"
            f"- Parameters: **{total_params:,}**\n"
            f"- Training sequences: **{len(sequences):,}**\n"
            f"- Final loss: **{losses[-1]:.6f}**\n"
            f"- Device: **{device.type}**\n"
            f"- {push_message}"
        )
        return summary, str(zip_path)

    except Exception as exc:
        detail = traceback.format_exc(limit=5)
        return f"### Build failed\n**{type(exc).__name__}:** {exc}\n\n```text\n{detail}\n```", None


def build_ui():
    with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🧬 Modelmaker\n"
            "Build a compact causal LLM from your own material. The result is saved in Hugging Face `save_pretrained()` format."
        )

        with gr.Row():
            with gr.Column(scale=3):
                model_name = gr.Textbox(label="Model name", value="synthia-seed")
                pasted_text = gr.Textbox(
                    label="Training text",
                    lines=14,
                    placeholder="Paste training material here, or upload files below.",
                )
                files = gr.File(
                    label="Training files",
                    file_count="multiple",
                    file_types=[".txt", ".md", ".json", ".jsonl", ".py", ".js", ".ts", ".html", ".css", ".csv"],
                )

            with gr.Column(scale=2):
                gr.Markdown("### Model shape")
                vocab_size = gr.Slider(512, 16000, value=4096, step=256, label="Vocabulary")
                context_length = gr.Slider(32, 512, value=128, step=32, label="Context length")
                layers = gr.Slider(1, 12, value=2, step=1, label="Transformer layers")
                heads = gr.Dropdown([1, 2, 4, 8], value=4, label="Attention heads")
                embedding_size = gr.Dropdown([64, 128, 256, 512], value=256, label="Embedding size")

                gr.Markdown("### Training")
                epochs = gr.Slider(1, 20, value=2, step=1, label="Epochs")
                batch_size = gr.Dropdown([1, 2, 4, 8, 16], value=4, label="Batch size")
                learning_rate = gr.Number(value=0.0003, label="Learning rate")
                push_to_hub = gr.Checkbox(
                    value=False,
                    label="Push model to Hugging Face after training (requires HF_TOKEN Space secret)",
                )

        build = gr.Button("Build model", variant="primary")
        status = gr.Markdown()
        artifact = gr.File(label="Built model package")

        build.click(
            fn=train_model,
            inputs=[
                model_name,
                pasted_text,
                files,
                vocab_size,
                context_length,
                layers,
                heads,
                embedding_size,
                epochs,
                batch_size,
                learning_rate,
                push_to_hub,
            ],
            outputs=[status, artifact],
        )

        gr.Markdown(
            "Small models are intentional here: Modelmaker should create testable specialized brains first, then scale only when the data and role justify it."
        )

    return demo


if __name__ == "__main__":
    build_ui().launch()
