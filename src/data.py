from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class MultiSourceDataset(Dataset):
    def __init__(self, manifest, num_sources, transform=None, split=None):
        self.df = pd.read_csv(manifest)
        if split is not None and "split" in self.df.columns:
            self.df = self.df[self.df["split"].astype(str) == str(split)].reset_index(drop=True)
        self.num_sources = int(num_sources)
        self.transform = transform
        required = {"sample_id", "label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Manifest missing columns: {sorted(missing)}")
        for s in range(self.num_sources):
            col = f"source_{s}"
            if col not in self.df.columns:
                raise ValueError(f"Manifest missing {col}")

    def __len__(self):
        return len(self.df)

    def _load(self, value):
        if pd.isna(value) or str(value).strip() == "":
            return None
        p = Path(str(value))
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        img = Image.open(p).convert("RGB")
        return self.transform(img) if self.transform else img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sources = [self._load(row[f"source_{s}"]) for s in range(self.num_sources)]
        return {
            "sample_id": str(row["sample_id"]),
            "patient_id": str(row["patient_id"]) if "patient_id" in row else str(row["sample_id"]),
            "sources": sources,
            "label": int(row["label"]),
        }

def collate_multisource(batch):
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    n = len(batch[0]["sources"])
    out = {
        "sample_id": [b["sample_id"] for b in batch],
        "patient_id": [b["patient_id"] for b in batch],
        "label": labels,
        "sources": [],
        "source_masks": [],
    }
    for s in range(n):
        imgs = [b["sources"][s] for b in batch]
        valid = torch.tensor([x is not None for x in imgs], dtype=torch.bool)
        out["source_masks"].append(valid)
        if valid.all():
            out["sources"].append(torch.stack(imgs))
        elif (~valid).all():
            out["sources"].append(None)
        else:
            shape = next(x.shape for x in imgs if x is not None)
            tensor = torch.zeros((len(imgs), *shape), dtype=imgs[0].dtype if imgs[0] is not None else torch.float32)
            for i, x in enumerate(imgs):
                if x is not None:
                    tensor[i] = x
            out["sources"].append(tensor)
    return out

def make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_multisource,
    )
