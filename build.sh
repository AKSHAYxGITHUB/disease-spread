#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# The trimmed dataset (~2 years) is committed to the repo and small enough to
# ship directly, so there is no need to regenerate it at build time. To rebuild
# the full historical dataset locally, run: python data_generator.py
