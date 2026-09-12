import argparse
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

def paired_test(proposed, baseline):
    proposed = np.asarray(proposed, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if proposed.shape != baseline.shape:
        raise ValueError("Paired arrays must have the same shape.")
    stat, p = ttest_rel(proposed, baseline)
    return float(stat), float(p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--proposed", required=True)
    ap.add_argument("--baseline", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    stat, p = paired_test(df[args.proposed], df[args.baseline])
    print(f"paired t-test statistic={stat:.6f}, p={p:.6g}, significant={p < 0.05}")

if __name__ == "__main__":
    main()
