#!/usr/bin/env bash
# Download raw datasets for AHF experiments
set -e

DATA_DIR="$(dirname "$0")"

echo "=== Downloading ML100K ==="
wget -q -O "$DATA_DIR/ml-100k.zip" \
  "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
unzip -q "$DATA_DIR/ml-100k.zip" -d "$DATA_DIR/"
echo "ML100K done."

echo "=== Downloading ML1M ==="
wget -q -O "$DATA_DIR/ml-1m.zip" \
  "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
unzip -q "$DATA_DIR/ml-1m.zip" -d "$DATA_DIR/"
echo "ML1M done."

echo "=== Yelp ==="
echo "Yelp data requires manual download from https://www.yelp.com/dataset"
echo "Place 'yelp_academic_dataset_review.json' and 'yelp_academic_dataset_business.json'"
echo "in $DATA_DIR/yelp_raw/"

echo "=== All downloads complete ==="
