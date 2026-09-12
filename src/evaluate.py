import argparse
from pathlib import Path
import json
import torch
import pandas as pd

from .config import load_config
from .data import MultiSourceDataset, make_loader
from .augmentations import build_transforms
from .models import ReliabilityAwareHybridFusion
from .metrics import classification_metrics
from .utils import get_device, save_json

@torch.no_grad()
def collect(model, loader):
    model.eval()
    ys, ps, ids, rel, att = [], [], [], [], []
    for batch in loader:
        out = model(batch["sources"], batch["source_masks"])
        ys.append(batch["label"].cpu())
        ps.append(out["final_probs"].cpu())
        ids.extend(batch["sample_id"])
        rel.append(out["reliability_weights"].cpu())
        att.append(out["attention_weights"].cpu())
    return torch.cat(ys).numpy(), torch.cat(ps).numpy(), ids, torch.cat(rel).numpy(), torch.cat(att).numpy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = get_device()
    S = int(cfg["dataset"]["num_sources"])
    model = ReliabilityAwareHybridFusion(
        S, int(cfg["dataset"]["num_classes"]),
        int(cfg["model"]["embedding_dim"]),
        int(cfg["model"]["attention_hidden_dim"]),
        pretrained=False,
        dropout=float(cfg["model"].get("dropout", 0.0)),
        temperature=float(cfg["fusion"]["temperature"]),
        epsilon=float(cfg["fusion"]["reliability_epsilon"]),
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    ds = MultiSourceDataset(
        cfg["dataset"]["manifest"], S, build_transforms(cfg, False),
        cfg.get("evaluation", {}).get("split", "test")
    )
    loader = make_loader(ds, int(cfg["training"]["batch_size"]), False, int(cfg["training"]["num_workers"]))
    y, p, ids, rel, att = collect(model, loader)
    metrics = classification_metrics(y, p)
    outdir = Path(cfg["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, outdir / "evaluation_metrics.json")
    pred = pd.DataFrame({"sample_id": ids, "true_label": y, "predicted_label": p.argmax(1)})
    for c in range(p.shape[1]):
        pred[f"prob_{c}"] = p[:, c]
    for s in range(rel.shape[1]):
        pred[f"reliability_weight_{s}"] = rel[:, s]
        pred[f"attention_weight_{s}"] = att[:, s]
    pred.to_csv(outdir / "predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
