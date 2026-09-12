from pathlib import Path
import tempfile
import pandas as pd
import numpy as np
from PIL import Image

def make_data(root, n=8, sources=3):
    rows = []
    for i in range(n):
        paths = []
        for s in range(sources):
            p = root / f"s{s}"
            p.mkdir(exist_ok=True)
            fn = p / f"{i}.png"
            arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(arr).save(fn)
            paths.append(str(fn))
        rows.append({
            "sample_id": f"x{i}",
            "patient_id": f"p{i}",
            "label": i % 2,
            **{f"source_{s}": paths[s] for s in range(sources)},
        })
    df = pd.DataFrame(rows)
    df["split"] = ["train"] * 4 + ["val"] * 2 + ["test"] * 2
    manifest = root / "manifest.csv"
    df.to_csv(manifest, index=False)
    return manifest

if __name__ == "__main__":
    import torch
    from .data import MultiSourceDataset, make_loader
    from .augmentations import build_transforms
    from .models import ReliabilityAwareHybridFusion

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        manifest = make_data(root)
        cfg = {
            "dataset": {
                "manifest": str(manifest),
                "num_sources": 3,
                "num_classes": 2,
                "image_size": 224,
            },
            "augmentation": {},
            "model": {
                "embedding_dim": 1024,
                "attention_hidden_dim": 256,
                "pretrained": False,
                "dropout": 0.0,
            },
            "fusion": {"temperature": 1.0, "reliability_epsilon": 1e-8},
            "training": {"batch_size": 2, "num_workers": 0},
        }
        ds = MultiSourceDataset(str(manifest), 3, build_transforms(cfg, False), "test")
        loader = make_loader(ds, 2, False, 0)
        model = ReliabilityAwareHybridFusion(3, 2, 1024, 256, pretrained=False)
        batch = next(iter(loader))
        out = model(batch["sources"], batch["source_masks"])
        assert out["final_probs"].shape == (2, 2)
        assert torch.allclose(out["final_probs"].sum(1), torch.ones(2), atol=1e-5)
        print("Smoke test passed.")
