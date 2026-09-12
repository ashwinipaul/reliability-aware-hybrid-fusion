import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def _patient_table(df):
    if "patient_id" not in df.columns:
        df = df.copy()
        df["patient_id"] = df["sample_id"]
    rows = []
    for pid, g in df.groupby("patient_id", sort=False):
        label = int(g["label"].mode().iloc[0])
        rows.append((pid, label))
    return pd.DataFrame(rows, columns=["patient_id", "label"])

def patient_split(df, train_ratio=0.70, val_ratio=0.10, seed=42):
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")
    pt = _patient_table(df)
    train_p, temp_p = train_test_split(
        pt["patient_id"].values,
        test_size=1.0 - train_ratio,
        random_state=seed,
        stratify=pt["label"].values,
    )
    temp = pt.set_index("patient_id").loc[temp_p].reset_index()
    val_fraction = val_ratio / (1.0 - train_ratio)
    val_p, test_p = train_test_split(
        temp["patient_id"].values,
        test_size=1.0 - val_fraction,
        random_state=seed,
        stratify=temp["label"].values,
    )
    split_map = {p: "train" for p in train_p}
    split_map.update({p: "val" for p in val_p})
    split_map.update({p: "test" for p in test_p})
    out = df.copy()
    if "patient_id" not in out.columns:
        out["patient_id"] = out["sample_id"]
    out["split"] = out["patient_id"].map(split_map)
    if out["split"].isna().any():
        raise RuntimeError("Some rows could not be assigned to a split.")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--train-ratio", type=float, default=0.70)
    ap.add_argument("--val-ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    df = pd.read_csv(args.manifest)
    out = patient_split(df, args.train_ratio, args.val_ratio, args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out["split"].value_counts().to_string())

if __name__ == "__main__":
    main()
