# ================================================================
# Makefile — Development tasks for Multi-Modal RAG Image Search
# ================================================================

.PHONY: help build up down logs install simulate index serve serve-dev \
        test test-search health stats benchmark clean \
        onnx-convert onnx-benchmark \
        k8s-apply k8s-delete k8s-staging k8s-prod lint \
        migrate web-index graph-stats cache-stats

SHELL := /bin/zsh
PYTHON := python3

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Infrastructure ──────────────────────────────────────────

build: ## Build all Docker images
	docker compose build

up: ## Start full stack (app + qdrant + redis)
	docker compose up -d
	@echo "Waiting for Qdrant..."
	@until curl -s http://localhost:6333/healthz > /dev/null 2>&1; do sleep 1; done
	@echo "Qdrant ready on :6333 (gRPC :6334)"
	@echo "Redis ready on :6379"
	@echo "API ready on :8000"

down: ## Stop and remove all containers and volumes
	docker compose down -v

logs: ## Tail all service logs
	docker compose logs -f

# ── Python Environment ──────────────────────────────────────

install: ## Install Python dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# ── Data Pipeline ───────────────────────────────────────────

simulate: ## Generate 1M simulated vectors into Qdrant
	$(PYTHON) -m scripts.simulate_dataset --num-vectors 1000000 --batch-size 10000

index: ## Index real images from ./data/images into Qdrant
	$(PYTHON) -m indexing.index_images --image-dir ./data/images --batch-size 256

# ── Server ──────────────────────────────────────────────────

serve: ## Start FastAPI server locally (outside Docker)
	$(PYTHON) -m uvicorn services.api_gateway.app:app \
		--host 0.0.0.0 \
		--port 8000 \
		--workers 1 \
		--log-level info

serve-dev: ## Start FastAPI server with auto-reload (dev mode)
	$(PYTHON) -m uvicorn services.api_gateway.app:app \
		--host 0.0.0.0 \
		--port 8000 \
		--reload \
		--log-level debug

# ── Testing ─────────────────────────────────────────────────

test: ## Run test suite
	$(PYTHON) -m pytest tests/ -v --tb=short

test-search: ## Quick manual search test
	curl -s -X POST http://localhost:8000/search \
		-H "Content-Type: application/json" \
		-d '{"query": "a dog playing in the park", "top_k": 5}' | python3 -m json.tool

health: ## Check service health
	curl -s http://localhost:8000/health | python3 -m json.tool

stats: ## Get runtime performance stats
	curl -s http://localhost:8000/stats | python3 -m json.tool

# ── Benchmarks ──────────────────────────────────────────────

benchmark: ## Run search latency benchmark
	$(PYTHON) -m scripts.benchmark --num-requests 100 --top-k 10

# ── V2: Migration & Management ──────────────────────────────

migrate: ## Migrate legacy collections to unified V2 collection
	$(PYTHON) -m scripts.migrate_to_unified

migrate-dry: ## Preview migration without changes
	$(PYTHON) -m scripts.migrate_to_unified --dry-run

web-index: ## Index a URL (usage: make web-index URL=https://example.com)
	curl -s -X POST http://localhost:8000/web/index \
		-H "Content-Type: application/json" \
		-d '{"url": "$(URL)"}' | python3 -m json.tool

graph-stats: ## Show knowledge graph statistics
	curl -s http://localhost:8000/graph/stats | python3 -m json.tool

cache-stats: ## Show cache statistics (L1/L2/L3)
	curl -s http://localhost:8000/stats | python3 -m json.tool | grep -A 20 cache

# ── ONNX Runtime ────────────────────────────────────────────

onnx-convert: ## Export CLIP text encoder to ONNX format
	$(PYTHON) -m scripts.convert_to_onnx \
		--model ViT-H-14 --pretrained laion2b_s32b_b79k \
		--output-dir models/onnx --validate

onnx-benchmark: ## Compare ONNX vs PyTorch inference speed
	$(PYTHON) -m scripts.benchmark_onnx_vs_pytorch \
		--iterations 200 --output results/onnx_benchmark.csv

# ── Kubernetes ──────────────────────────────────────────────

k8s-apply: ## Deploy to Kubernetes (default: dev overlay)
	kubectl apply -k k8s/overlays/dev

k8s-delete: ## Delete Kubernetes deployment
	kubectl delete -k k8s/overlays/dev

k8s-staging: ## Deploy staging overlay
	kubectl apply -k k8s/overlays/staging

k8s-prod: ## Deploy production overlay
	kubectl apply -k k8s/overlays/prod

# ── Code Quality ────────────────────────────────────────────

lint: ## Run linters
	$(PYTHON) -m ruff check .
	$(PYTHON) -m mypy services/ configs/ utils/ --ignore-missing-imports

# ── Cleanup ─────────────────────────────────────────────────

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .mypy_cache .ruff_cache dist build *.egg-info
