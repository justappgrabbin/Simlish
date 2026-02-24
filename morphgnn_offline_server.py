"""
MorphingGNN — ONNX + Offline HTTP Server + HF Hub
--------------------------------------------------
Step 1 (torch needed, once):   python morphgnn_offline_server.py --export
Step 2 (offline forever):      python morphgnn_offline_server.py --serve
Step 3 (push to Hub once):     python morphgnn_offline_server.py --push --repo USERNAME/morphing-gnn

Runtime deps:
  export:  pip install torch onnx
  serve:   pip install onnxruntime          ← no torch needed at runtime
  hub:     pip install huggingface_hub
"""

import json
import argparse
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from pathlib import Path

MODEL_PATH = Path("morphgnn.onnx")
HUG_PATH   = Path("hug_types.json")
MODES      = ["spatial", "temporal", "attention", "diffusion", "hierarchical"]

# ── HUG METADATA ──────────────────────────────────────────────────────────────

HUG_TYPES = {
    "warm":      {"edge_weight": 0.6, "mode_bias": [0.0, 0.0, 1.5, 0.0, 0.0], "desc": "Gentle, steady presence"},
    "electric":  {"edge_weight": 1.0, "mode_bias": [0.5, 0.0, 2.0, 0.0, 0.0], "desc": "High-energy celebration"},
    "grounding": {"edge_weight": 0.4, "mode_bias": [1.0, 0.0, 0.5, 0.5, 0.0], "desc": "Stabilizing body-field support"},
    "healing":   {"edge_weight": 0.5, "mode_bias": [0.0, 0.5, 1.5, 0.5, 0.0], "desc": "Soft recovery after hard moments"},
    "cosmic":    {"edge_weight": 1.5, "mode_bias": [0.0, 0.0, 2.5, 0.5, 1.0], "desc": "Full field merge — rare, powerful"},
}

# ── STEP 1: EXPORT TO ONNX ────────────────────────────────────────────────────

def export_onnx():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class MorphingGNN(nn.Module):
        def __init__(self):
            super().__init__()
            H = 32
            self.controller    = nn.Sequential(nn.Linear(8 + H, 128), nn.ReLU(), nn.Linear(128, 5))
            self.mode_bias     = nn.Parameter(torch.zeros(5))
            self.att_mlp       = nn.Sequential(nn.Linear(H * 2, 64), nn.ReLU(), nn.Linear(64, 1))
            self.heat_kernels  = nn.Parameter(torch.linspace(0.1, 2.0, 5))
            self.time_gate     = nn.Linear(H + 1, H)
            self.scale_weights = nn.Parameter(torch.ones(3))
            self.s_layers = nn.ModuleList([nn.Linear(16 if i==0 else H, H) for i in range(2)])
            self.a_layers = nn.ModuleList([nn.Linear(16 if i==0 else H, H) for i in range(2)])
            self.output   = nn.Sequential(nn.Linear(H, H//2), nn.ReLU(), nn.Linear(H//2, 8))
            self.register_buffer('prev_emb', torch.zeros(1, H))

        def agg(self, x, src, dst, ew):
            out = torch.zeros_like(x)
            out.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.size(1)), x[src] * ew.unsqueeze(1))
            deg = torch.zeros(x.size(0)).scatter_add_(0, dst, ew).clamp(min=1).view(-1, 1)
            return out / deg

        def forward(self, x, src, dst, ew, hug_bias):
            n, e = x.size(0), src.size(0)
            deg  = torch.zeros(n).scatter_add_(0, dst, torch.ones(e))
            stats = torch.stack([
                torch.tensor(n/1000.), torch.tensor(float(e)/max(n,1)),
                deg.std(), (deg==0).float().mean(),
                x.mean(), x.std(), ew.mean(),
                torch.tensor(float(e)/max(n*n,1))
            ])
            logits = self.controller(torch.cat([stats, self.prev_emb.squeeze(0)])) + self.mode_bias + hug_bias
            probs  = torch.softmax(logits, dim=0)

            # spatial path
            hx = x
            for lin in self.s_layers:
                hx = F.relu(self.agg(lin(hx), src, dst, ew))

            # attention path
            ax = x
            for lin in self.a_layers:
                ax   = lin(ax)
                attn = F.softmax(self.att_mlp(torch.cat([ax[dst], ax[src]], -1)).squeeze(-1), dim=0)
                out  = torch.zeros_like(ax)
                out.scatter_add_(0, dst.unsqueeze(1).expand(-1, ax.size(1)), ax[src] * attn.unsqueeze(1))
                ax   = F.relu(out)

            fused = probs[0] * hx + probs[2] * ax
            self.prev_emb = fused.detach().mean(0, keepdim=True)
            return self.output(fused), probs

    model = MorphingGNN(); model.eval()
    N, E  = 20, 60
    dummy = (
        torch.randn(N, 16),
        torch.randint(0, N, (E,)),
        torch.randint(0, N, (E,)),
        torch.ones(E),
        torch.zeros(5),
    )
    torch.onnx.export(
        model, dummy, str(MODEL_PATH),
        input_names=["node_features", "src", "dst", "edge_weights", "hug_bias"],
        output_names=["output", "mode_probs"],
        dynamic_axes={"node_features": {0: "N"}, "src": {0: "E"}, "dst": {0: "E"}, "edge_weights": {0: "E"}},
        opset_version=14,
    )
    HUG_PATH.write_text(json.dumps(HUG_TYPES, indent=2))
    print(f"✅ ONNX exported → {MODEL_PATH}  ({MODEL_PATH.stat().st_size // 1024} KB)")
    print(f"✅ Hug types     → {HUG_PATH}")


# ── STEP 2: OFFLINE HTTP SERVER (onnxruntime only) ────────────────────────────

SESSION = None

def get_session():
    global SESSION
    if SESSION is None:
        import onnxruntime as ort
        assert MODEL_PATH.exists(), f"No model at {MODEL_PATH} — run --export first"
        SESSION = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
        print(f"✅ ONNX session ready")
    return SESSION

def run_infer(body):
    sess = get_session()
    x    = np.array(body["node_features"], dtype=np.float32)
    src  = np.array(body["src"],           dtype=np.int64)
    dst  = np.array(body["dst"],           dtype=np.int64)
    ew   = np.array(body.get("edge_weights", [1.0] * len(src)), dtype=np.float32)

    hug_bias = np.zeros(5, dtype=np.float32)
    hug_type = body.get("hug_type")
    if hug_type and hug_type in HUG_TYPES:
        meta     = HUG_TYPES[hug_type]
        hug_bias = np.array(meta["mode_bias"], dtype=np.float32)
        boost    = meta["edge_weight"]
        sn, rn   = body.get("sender_node"), body.get("receiver_node")
        if sn is not None and rn is not None:
            mask     = ((src == sn) & (dst == rn)) | ((src == rn) & (dst == sn))
            ew[mask] = boost

    out, probs = sess.run(None, {
        "node_features": x, "src": src, "dst": dst,
        "edge_weights": ew, "hug_bias": hug_bias,
    })
    mode_idx = int(np.argmax(probs))
    return {
        "output":      out.tolist(),
        "active_mode": MODES[mode_idx],
        "mode_probs":  dict(zip(MODES, [round(float(p), 4) for p in probs])),
        "hug_applied": hug_type,
        "nodes": int(x.shape[0]), "edges": int(len(src)),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): print(f"  {fmt % args}")

    def send_json(self, data, status=200):
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def read_body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n)) if n else {}

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", ""):
            self.send_json({"system": "YOU-N-I-VERSE", "model": "MorphingGNN-ONNX", "status": "offline"})
        elif path == "/hug-types":
            self.send_json(HUG_TYPES)
        elif path == "/health":
            self.send_json({"ok": True, "model_exists": MODEL_PATH.exists()})
        else:
            self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self.read_body()
            if path == "/infer":
                self.send_json(run_infer(body))
            elif path == "/hug":
                h    = body.get("hug_type", "warm")
                meta = HUG_TYPES.get(h, HUG_TYPES["warm"])
                self.send_json({
                    "event": "hug_sent", "from": body.get("sender_node"),
                    "to": body.get("receiver_node"), "hug_type": h,
                    "edge_boost": meta["edge_weight"], "desc": meta["desc"],
                })
            else:
                self.send_json({"error": "not found"}, 404)
        except Exception as ex:
            self.send_json({"error": str(ex)}, 500)


def serve(port=8000):
    assert MODEL_PATH.exists(), f"No model at {MODEL_PATH} — run --export first"
    print(f"\n🌌 YOU-N-I-VERSE offline  →  http://localhost:{port}")
    print(f"   GET  /hug-types        — all 5 hug resonances")
    print(f"   POST /infer            — run GNN inference")
    print(f"   POST /hug              — emit hug event")
    print(f"   GET  /health\n")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()


# ── STEP 3: PUSH TO HF HUB ────────────────────────────────────────────────────

def push_hub(repo_id, private=False):
    from huggingface_hub import HfApi, create_repo
    assert MODEL_PATH.exists(), "Run --export first"

    readme = f"""---
license: apache-2.0
tags: [onnx, graph-neural-network, consciousness, human-design, offline, resonance]
---
# MorphingGNN — YOU-N-I-VERSE (ONNX, offline-ready)

Self-morphing resonance network. No PyTorch at runtime — just `onnxruntime`.

## Quick Start (offline)
```bash
pip install onnxruntime huggingface_hub
huggingface-cli download {repo_id} morphgnn.onnx hug_types.json
python morphgnn_offline_server.py --serve
```

## Hug Types
| Type | Edge Boost | Field |
|------|-----------|-------|
| warm | 0.6 | Gentle presence |
| electric | 1.0 | Celebration |
| grounding | 0.4 | Body-field anchor |
| healing | 0.5 | Soft recovery |
| cosmic | 1.5 | Full field merge |

## Inputs
`node_features [N,16]` · `src [E]` · `dst [E]` · `edge_weights [E]` · `hug_bias [5]`
"""
    Path("README.md").write_text(readme)
    api = HfApi()
    create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    for f in [MODEL_PATH, HUG_PATH, Path("README.md"), Path(__file__)]:
        if f.exists():
            api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name, repo_id=repo_id, repo_type="model")
            print(f"  ↑ {f.name}")
    print(f"\n🚀 https://huggingface.co/{repo_id}")
    print(f"   Pull:  huggingface-cli download {repo_id} morphgnn.onnx hug_types.json")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MorphingGNN ONNX server")
    p.add_argument("--export",  action="store_true", help="Export to ONNX (needs torch+onnx)")
    p.add_argument("--serve",   action="store_true", help="Run offline HTTP server (needs onnxruntime)")
    p.add_argument("--push",    action="store_true", help="Push to HuggingFace Hub")
    p.add_argument("--repo",    default="",          help="HF repo e.g. username/morphing-gnn")
    p.add_argument("--port",    type=int, default=8000)
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    if   args.export: export_onnx()
    elif args.serve:  serve(args.port)
    elif args.push:   push_hub(args.repo, args.private)
    else:             p.print_help()
