#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install dependencies
pip install -r requirements.txt

# Generate the dataset (since it's too large to be reliably checked into Git)
python data_generator.py
