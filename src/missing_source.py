import argparse
import itertools
from pathlib import Path
import pandas as pd
import torch

from .config import load_config
from .data import MultiSourceDataset, make_loader
from .augmentations import build_transforms
from .models import ReliabilityAwareHybridFusion
from .metrics import classification_metrics
from .utils import get_device

@torch.no_grad()
def evaluate_mask(model, loader, missing):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        sources = list(batch["sources"])
        masks = [m.clone() for m in batch["source_masks"]]
        for s in missing:
            sources[s] = None
            masks[s].zero_()
        out = model(sources, masks)
        ys.append(batch["label"].cpu())
        ps.append(out["final_probs"].cpu())
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

    cases = [()]
    if S >= 2:
        cases += list(itertools.combinations(range(S), 1))
    if S >= 3:
        cases += list(itertools.combinations(range(S), 2))
    rows = []
    for missing in cases:
        met = evaluate_mask(model, loader, tuple(missing))
        rows.append({
            "missing_sources": "none" if not missing else ",".join(map(str, missing)),
            "num_missing": len(missing),
            **met,
        })
    out = pd.DataFrame(rows)
    Path(cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(cfg["output_dir"]) / "missing_source_results.csv", index=False)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
