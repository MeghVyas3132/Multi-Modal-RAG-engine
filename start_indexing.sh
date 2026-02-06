#!/bin/bash
# Launch indexing as a fully detached process
cd /Users/meghvyas/Desktop/Multi-Modal-RAG-engine
source .venv/bin/activate
nohup python -m indexing.index_images --image-dir ./data/food-101/images --batch-size 64 >> /tmp/indexing.log 2>&1 &
INDEXPID=$!
echo "$INDEXPID" > /tmp/indexing.pid
echo "Indexing started with PID=$INDEXPID"
echo "Monitor with: tail -f /tmp/indexing.log"
echo "Check progress: curl -s http://localhost:6333/collections/image_vectors | python3 -m json.tool"
