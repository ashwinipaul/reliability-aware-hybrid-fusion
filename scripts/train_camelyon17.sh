#!/usr/bin/env bash
set -e
python -m src.train --config configs/camelyon17.yaml --seed 42
python -m src.evaluate --config configs/camelyon17.yaml --checkpoint outputs/camelyon17/best.pt
python -m src.missing_source --config configs/camelyon17.yaml --checkpoint outputs/camelyon17/best.pt
