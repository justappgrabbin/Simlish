"""
MorphingGNN HTTP Server
FastAPI endpoint for consciousness field inference + hug mechanic
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import uvicorn

# ── ENUMS ──────────────────────────────────────────────────────────────────────

class MorphMode(Enum):
    SPATIAL      = "spatial"
    TEMPORAL     = "temporal"
    ATTENTION    = "attention"
    DIFFUSION    = "diffusion"
    HIERARCHICAL = "hierarchical"

class HugType(str, Enum):
    WARM      = "warm"        # gentle, steady presence
    ELECTRIC  = "electric"    # high-energy celebration
    GROUNDING = "grounding"   # stabilizing, body-field support
    HEALING   = "healing"     # soft recovery
    COSMIC    = "cosmic"      # full field merge — rare, powerful

HUG_EDGE_WEIGHT = {
    HugType.WARM:      0.6,
    HugType.ELECTRIC:  1.0,
    HugType.GROUNDING: 0.4,
    HugType.HEALING:   0.5,
    HugType.COSMIC:    1.5,
}

# Bias toward ATTENTION mode for all hugs (relational gesture)
HUG_MODE_BIAS = {
    HugType.WARM:      [0.0, 0.0, 1.5, 0.0, 0.0],
    HugType.ELECTRIC:  [0.5, 0.0, 2.0, 0.0, 0.0],
    HugType.GROUNDING: [1.0, 0.0, 0.5, 0.5, 0.0],
    HugType.HEALING:   [0.0, 0.5, 1.5, 0.5, 0.0],
    HugType.COSMIC:    [0.0, 0.0, 2.5, 0.5, 1.0],
}

# ── CONFIG & MODEL ─────────────────────────────────────────────────────────────

@dataclass
class MorphConfig:
    in_channels:     int   = 16
    hidden_channels: int   = 32
    out_channels:    int   = 8
    num_layers:      int   = 2
    temperature:     float = 0.5
    dropout:         float = 0.2

class MorphingGNN(nn.Module):
    def __init__(self, config: MorphConfig):
        super().__init__()
        self.config = config

        self.controller = nn.Sequential(
            nn.Linear(8 + config.hidden_channels, 128),
            nn.ReLU(), nn.Dropout(config.dropout), nn.Linear(128, 5)
        )
        self.mode_bias    = nn.Parameter(torch.zeros(5))
        self.att_mlp      = nn.Sequential(nn.Linear(config.hidden_channels * 2, 64), nn.ReLU(), nn.Linear(64, 1))
        self.heat_kernels = nn.Parameter(torch.linspace(0.1, 2.0, 5))
        self.time_gate    = nn.Linear(config.hidden_channels + 1, config.hidden_channels)
        self.scale_weights = nn.Parameter(torch.ones(3))

        self.spatial_layers     = self._make_layers()
        self.temporal_layers    = self._make_layers()
        self.attention_layers   = self._make_layers()
        self.diffusion_layers   = self._make_layers()
        self.hierarchical_layers = self._make_layers()

        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_channels, config.hidden_channels // 2),
            nn.ReLU(), nn.Dropout(config.dropout),
            nn.Linear(config.hidden_channels // 2, config.out_channels)
        )
        self.register_buffer('prev_embedding', torch.zeros(1, config.hidden_channels))

    def _make_layers(self):
        return nn.ModuleList([
            nn.Linear(
                self.config.in_channels if i == 0 else self.config.hidden_channels,
                self.config.hidden_channels
            ) for i in range(self.config.num_layers)
        ])

    def compute_stats(self, x, edge_index, edge_weight=None):
        n, e = x.size(0), edge_index.size(1)
        row, col = edge_index
        deg = torch.zeros(n, device=x.device).scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
        return torch.tensor([
            n / 1000.0, e / max(n, 1), deg.std().item() if n > 0 else 0,
            (deg == 0).float().mean().item(), x.mean().item(), x.std().item(),
            edge_weight.mean().item() if edge_weight is not None else 1.0,
            (e / (n * n)) if n > 0 else 0,
        ], device=x.device, dtype=torch.float)

    def aggregate(self, x, edge_index, edge_weight=None):
        row, col = edge_index
        if edge_weight is None:
            edge_weight = torch.ones(row.size(0), device=x.device)
        out = torch.zeros_like(x)
        out.scatter_add_(0, row.unsqueeze(1).expand(-1, x.size(1)), x[col] * edge_weight.unsqueeze(1))
        deg = torch.zeros(x.size(0), device=x.device).scatter_add_(0, row, edge_weight).clamp(min=1).view(-1, 1)
        return out / deg

    def spatial_pass(self, x, ei, layers, ew):
        for lin in layers:
            x = F.relu(self.aggregate(lin(x), ei, ew))
        return x

    def attention_pass(self, x, ei, layers):
        for lin in layers:
            x = lin(x)
            row, col = ei
            attn = F.softmax(self.att_mlp(torch.cat([x[row], x[col]], dim=-1)).squeeze(-1), dim=0)
            out  = torch.zeros_like(x)
            out.scatter_add_(0, row.unsqueeze(1).expand(-1, x.size(1)), x[col] * attn.unsqueeze(1))
            x = F.relu(out)
        return x

    def diffusion_pass(self, x, ei, layers):
        row, col = ei
        deg = torch.zeros(x.size(0), device=x.device).scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
        deg_inv_sqrt = deg.pow(-0.5).clamp(min=1e-8)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        for lin in layers:
            x = lin(x); outputs = []
            for t in self.heat_kernels:
                h = torch.zeros_like(x)
                h.scatter_add_(0, row.unsqueeze(1).expand(-1, x.size(1)), x[col] * norm.unsqueeze(1))
                x = (1 - t) * x + t * h; outputs.append(x)
            x = F.relu(torch.stack(outputs).mean(dim=0))
        return x

    def hierarchical_pass(self, x, ei, layers):
        for lin in layers:
            x  = lin(x)
            h1 = self.aggregate(x, ei)
            h2 = self.aggregate(h1, ei)
            h3 = self.aggregate(h2, ei)
            w  = F.softmax(self.scale_weights, dim=0)
            x  = F.relu(w[0]*h1 + w[1]*h2 + w[2]*h3)
        return x

    def temporal_pass(self, x, ei, layers, ts):
        if ts is None:
            ts = torch.zeros(x.size(0), 1, device=x.device)
        for lin in layers:
            x    = lin(x)
            gate = torch.sigmoid(self.time_gate(torch.cat([x, ts], dim=-1)))
            agg  = self.aggregate(x, ei)
            x    = F.relu(gate * x + (1 - gate) * agg)
        return x

    def forward(self, x, edge_index, timestamps=None, edge_weight=None,
                return_info=False, hug_bias=None):
        stats   = self.compute_stats(x, edge_index, edge_weight)
        quality = self.prev_embedding.mean(dim=0)
        logits  = self.controller(torch.cat([stats, quality], dim=0)) + self.mode_bias

        if hug_bias is not None:
            logits = logits + torch.tensor(hug_bias, dtype=torch.float, device=x.device)

        probs = F.gumbel_softmax(logits, tau=self.config.temperature, hard=False)
        modes = [MorphMode.SPATIAL, MorphMode.TEMPORAL, MorphMode.ATTENTION,
                 MorphMode.DIFFUSION, MorphMode.HIERARCHICAL]
        lsets = [self.spatial_layers, self.temporal_layers, self.attention_layers,
                 self.diffusion_layers, self.hierarchical_layers]

        outputs = []
        for i, (mode, layers) in enumerate(zip(modes, lsets)):
            if probs[i] > 0.001:
                if mode == MorphMode.SPATIAL:
                    out = self.spatial_pass(x, edge_index, layers, edge_weight)
                elif mode == MorphMode.TEMPORAL:
                    out = self.temporal_pass(x, edge_index, layers, timestamps)
                elif mode == MorphMode.ATTENTION:
                    out = self.attention_pass(x, edge_index, layers)
                elif mode == MorphMode.DIFFUSION:
                    out = self.diffusion_pass(x, edge_index, layers)
                else:
                    out = self.hierarchical_pass(x, edge_index, layers)
                outputs.append(probs[i] * out)

        if not outputs:
            outputs.append(self.spatial_pass(x, edge_index, self.spatial_layers, edge_weight))

        x = torch.stack(outputs).sum(dim=0) if len(outputs) > 1 else outputs[0]
        self.prev_embedding = x.detach().mean(dim=0, keepdim=True)
        out = self.output_proj(x)

        if return_info:
            selected = probs.argmax().item()
            return out, {'mode_name': modes[selected].value, 'mode_probs': probs.detach().tolist()}
        return out


# ── FASTAPI APP ────────────────────────────────────────────────────────────────

app    = FastAPI(title="YOU-N-I-VERSE MorphingGNN", version="1.0.0")
config = MorphConfig()
model  = MorphingGNN(config)
model.eval()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── REQUEST SCHEMAS ────────────────────────────────────────────────────────────

class InferRequest(BaseModel):
    node_features:  List[List[float]]          # [N, 16]
    edge_index:     List[List[int]]            # [[src...], [dst...]]
    edge_weights:   Optional[List[float]] = None
    timestamps:     Optional[List[float]] = None
    hug_type:       Optional[HugType]     = None
    sender_node:    Optional[int]         = None   # node sending the hug
    receiver_node:  Optional[int]         = None   # node receiving the hug

class HugRequest(BaseModel):
    sender_node:   int
    receiver_node: int
    hug_type:      HugType
    message:       Optional[str] = None

# ── ROUTES ─────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "system": "YOU-N-I-VERSE",
        "model":  "MorphingGNN v1",
        "status": "online",
        "hug_types": [h.value for h in HugType],
    }

@app.get("/hug-types")
def get_hug_types():
    return {
        h.value: {
            "edge_weight": HUG_EDGE_WEIGHT[h],
            "mode_bias":   HUG_MODE_BIAS[h],
            "description": {
                HugType.WARM:      "Gentle, steady presence",
                HugType.ELECTRIC:  "High-energy celebration",
                HugType.GROUNDING: "Stabilizing body-field support",
                HugType.HEALING:   "Soft recovery after hard moments",
                HugType.COSMIC:    "Full field merge — rare, powerful",
            }[h]
        }
        for h in HugType
    }

@app.post("/infer")
def infer(req: InferRequest):
    try:
        x          = torch.tensor(req.node_features, dtype=torch.float)
        edge_index = torch.tensor(req.edge_index, dtype=torch.long)
        ew         = torch.tensor(req.edge_weights, dtype=torch.float) if req.edge_weights else None
        ts         = torch.tensor([[t] for t in req.timestamps], dtype=torch.float) if req.timestamps else None

        hug_bias   = None
        boosted_ew = ew

        if req.hug_type and req.sender_node is not None and req.receiver_node is not None:
            hug_bias   = HUG_MODE_BIAS[req.hug_type]
            boost      = HUG_EDGE_WEIGHT[req.hug_type]
            row, col   = edge_index
            # boost edge weights on the hug pair
            mask = ((row == req.sender_node) & (col == req.receiver_node)) | \
                   ((row == req.receiver_node) & (col == req.sender_node))
            if ew is None:
                ew = torch.ones(edge_index.size(1))
            boosted_ew = ew.clone()
            boosted_ew[mask] = boost

        with torch.no_grad():
            out, info = model(x, edge_index, timestamps=ts, edge_weight=boosted_ew,
                              return_info=True, hug_bias=hug_bias)

        return {
            "output":       out.tolist(),
            "active_mode":  info["mode_name"],
            "mode_probs":   dict(zip([m.value for m in MorphMode], info["mode_probs"])),
            "hug_applied":  req.hug_type.value if req.hug_type else None,
            "nodes":        x.size(0),
            "edges":        edge_index.size(1),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/hug")
def send_hug(req: HugRequest):
    """Standalone hug event — logs and returns field impact metadata."""
    weight = HUG_EDGE_WEIGHT[req.hug_type]
    bias   = HUG_MODE_BIAS[req.hug_type]
    return {
        "event":         "hug_sent",
        "from":          req.sender_node,
        "to":            req.receiver_node,
        "hug_type":      req.hug_type.value,
        "edge_boost":    weight,
        "mode_bias":     bias,
        "message":       req.message,
        "field_note":    f"{req.hug_type.value.title()} hug propagates through ATTENTION field.",
    }

@app.get("/health")
def health():
    params = sum(p.numel() for p in model.parameters())
    return {"status": "healthy", "params": params, "device": "cpu"}


# ── ENTRY ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
