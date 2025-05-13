#!/bin/bash

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <num-articles>"
    exit 1
fi

rm -rf data/wikipedia

python scripts/download_wikipedia_articles.py --num-articles $1
python scripts/embed_documents.py
python scripts/generate_queries.py
python scripts/generate_inputs.py
