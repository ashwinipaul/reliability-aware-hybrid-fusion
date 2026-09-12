#!/usr/bin/env bash
set -e
python -m src.train --config configs/camelyon16.yaml --seed 42
python -m src.evaluate --config configs/camelyon16.yaml --checkpoint outputs/camelyon16/best.pt
python -m src.missing_source --config configs/camelyon16.yaml --checkpoint outputs/camelyon16/best.pt
