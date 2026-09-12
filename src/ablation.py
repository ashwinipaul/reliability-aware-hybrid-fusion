import argparse
from pathlib import Path
import torch
import pandas as pd

from .config import load_config
from .data import MultiSourceDataset, make_loader
from .augmentations import build_transforms
from .models import ReliabilityAwareHybridFusion
from .metrics import classification_metrics
from .utils import get_device

@torch.no_grad()
def run(model, loader, mode):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        out = model(batch["sources"], batch["source_masks"])
        if mode == "feature":
            p = torch.softmax(out["fused_logits"], dim=-1)
        elif mode == "decision":
            probs = out["source_probs"]
            mask = out["source_masks"].float().unsqueeze(-1)
            p = (probs * mask).sum(1) / mask.sum(1).clamp_min(1)
        elif mode == "hybrid_no_reliability":
            probs = out["source_probs"]
            mask = out["source_masks"].float().unsqueeze(-1)
            p = (probs * mask).sum(1) / mask.sum(1).clamp_min(1)
        elif mode == "full":
            p = out["final_probs"]
        else:
            raise ValueError(mode)
        ys.append(batch["label"])
        ps.append(p.cpu())
    return classification_metrics(torch.cat(ys).numpy(), torch.cat(ps).numpy())

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
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])
    ds = MultiSourceDataset(
        cfg["dataset"]["manifest"], S, build_transforms(cfg, False),
        cfg.get("evaluation", {}).get("split", "test")
    )
    loader = make_loader(ds, int(cfg["training"]["batch_size"]), False, int(cfg["training"]["num_workers"]))
    rows = [{"configuration": mode, **run(model, loader, mode)}
            for mode in ["feature", "decision", "hybrid_no_reliability", "full"]]
    out = pd.DataFrame(rows)
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(cfg["output_dir"]) / "ablation_results.csv", index=False)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
