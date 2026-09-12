import argparse
from pathlib import Path
import time
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .config import load_config, save_config
from .data import MultiSourceDataset, make_loader
from .augmentations import build_transforms
from .losses import hybrid_loss
from .metrics import classification_metrics
from .models import ReliabilityAwareHybridFusion
from .utils import seed_everything, get_device, EarlyStopping, save_json

def run_epoch(model, loader, optimizer, cfg, train=True):
    model.train(train)
    total_loss, n = 0.0, 0
    all_y, all_p = [], []
    for batch in tqdm(loader, leave=False):
        labels = batch["label"].to(next(model.parameters()).device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        out = model(batch["sources"], batch["source_masks"])
        loss, _ = hybrid_loss(
            out, labels,
            lambda_source=float(cfg["training"]["lambda_source"]),
            lambda_reg=float(cfg["training"]["lambda_reg"]),
        )
        if train:
            loss.backward()
            optimizer.step()
        bs = labels.size(0)
        total_loss += float(loss.detach().cpu()) * bs
        n += bs
        all_y.append(labels.detach().cpu())
        all_p.append(out["final_probs"].detach().cpu())
    y, p = torch.cat(all_y).numpy(), torch.cat(all_p).numpy()
    metrics = classification_metrics(y, p)
    metrics["loss"] = total_loss / max(n, 1)
    return metrics

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        labels = batch["label"].to(next(model.parameters()).device)
        out = model(batch["sources"], batch["source_masks"])
        ys.append(labels.cpu())
        ps.append(out["final_probs"].cpu())
    return classification_metrics(torch.cat(ys).numpy(), torch.cat(ps).numpy())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("evaluation", {}).get("seed", 42))
    seed_everything(seed)

    device = get_device()
    manifest = cfg["dataset"]["manifest"]
    S = int(cfg["dataset"]["num_sources"])
    train_ds = MultiSourceDataset(manifest, S, build_transforms(cfg, True), "train")
    val_ds = MultiSourceDataset(manifest, S, build_transforms(cfg, False), "val")
    test_ds = MultiSourceDataset(manifest, S, build_transforms(cfg, False), "test")
    tc = cfg["training"]
    train_loader = make_loader(train_ds, int(tc["batch_size"]), True, int(tc["num_workers"]))
    val_loader = make_loader(val_ds, int(tc["batch_size"]), False, int(tc["num_workers"]))
    test_loader = make_loader(test_ds, int(tc["batch_size"]), False, int(tc["num_workers"]))

    model = ReliabilityAwareHybridFusion(
        num_sources=S,
        num_classes=int(cfg["dataset"]["num_classes"]),
        embedding_dim=int(cfg["model"]["embedding_dim"]),
        attention_hidden_dim=int(cfg["model"]["attention_hidden_dim"]),
        pretrained=bool(cfg["model"]["pretrained"]),
        dropout=float(cfg["model"].get("dropout", 0.0)),
        temperature=float(cfg["fusion"]["temperature"]),
        epsilon=float(cfg["fusion"]["reliability_epsilon"]),
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=float(tc["learning_rate"]), weight_decay=float(tc["weight_decay"]))
    scheduler = CosineAnnealingLR(optimizer, T_max=int(tc["epochs"]))
    stopper = EarlyStopping(int(tc["early_stopping_patience"]), mode="max")
    outdir = Path(cfg["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, outdir / "config.yaml")

    history = []
    for epoch in range(1, int(tc["epochs"]) + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, optimizer, cfg, True)
        va = run_epoch(model, val_loader, optimizer, cfg, False)
        scheduler.step()
        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in tr.items()},
            **{f"val_{k}": v for k, v in va.items()},
            "seconds": time.time() - t0,
        }
        history.append(row)
        score = va.get("auc", va["f1"])
        if stopper.step(score):
            torch.save(
                {"model": model.state_dict(), "config": cfg, "seed": seed, "epoch": epoch},
                outdir / "best.pt",
            )
        print(
            f"Epoch {epoch:03d} | train loss {tr['loss']:.4f} | "
            f"val AUC {va.get('auc', float('nan')):.4f} | val F1 {va['f1']:.4f}"
        )
        if stopper.should_stop:
            print("Early stopping.")
            break

    pd.DataFrame(history).to_csv(outdir / "history.csv", index=False)
    ckpt = torch.load(outdir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, test_loader)
    save_json(test_metrics, outdir / "test_metrics.json")
    print("Test:", test_metrics)

if __name__ == "__main__":
    main()
