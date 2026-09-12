import argparse
from pathlib import Path
import csv
import numpy as np
from PIL import Image
import openslide

def tissue_fraction(rgb):
    arr = np.asarray(rgb).astype(np.float32) / 255.0
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return float((gray < 0.90).mean())

def extract_nonoverlap(wsi_path, output_dir, patch_size=224, level=0, threshold=0.50):
    slide = openslide.OpenSlide(str(wsi_path))
    width, height = slide.level_dimensions[level]
    downsample = slide.level_downsamples[level]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            patch = slide.read_region(
                (int(x * downsample), int(y * downsample)),
                level,
                (patch_size, patch_size),
            ).convert("RGB")
            frac = tissue_fraction(patch)
            if frac >= threshold:
                name = f"x{x}_y{y}.png"
                patch.save(out / name)
                records.append({"patch": str(out / name), "x": x, "y": y, "tissue_fraction": frac})
    with open(out / "patch_index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["patch", "x", "y", "tissue_fraction"])
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} tissue patches to {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--patch-size", type=int, default=224)
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--tissue-threshold", type=float, default=0.50)
    args = ap.parse_args()
    extract_nonoverlap(args.wsi, args.output_dir, args.patch_size, args.level, args.tissue_threshold)

if __name__ == "__main__":
    main()
