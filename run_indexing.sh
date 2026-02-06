#!/bin/bash
cd /Users/meghvyas/Desktop/Multi-Modal-RAG-engine
source .venv/bin/activate
python -m indexing.index_images --image-dir ./data/food-101/images --batch-size 64 2>&1 | tee /tmp/indexing.log
