#!/usr/bin/env bash
set -e
python -m src.train --config configs/breakhis.yaml --seed 42
python -m src.evaluate --config configs/breakhis.yaml --checkpoint outputs/breakhis/best.pt
python -m src.missing_source --config configs/breakhis.yaml --checkpoint outputs/breakhis/best.pt
