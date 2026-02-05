# ================================================================# ================================================================# ================================================================# ================================================================

# Makefile -- Development tasks for Multi-Modal RAG Image Search

# ================================================================# Makefile -- Common development tasks



.PHONY: help build up down logs install simulate index serve serve-dev \# ================================================================# Makefile -- Common development tasks# Makefile — Common development tasks

        test health stats benchmark clean onnx-convert onnx-benchmark \

        k8s-apply k8s-delete lint



SHELL := /bin/zsh.PHONY: help build up down logs install simulate index serve serve-dev test test-search health stats benchmark clean# ================================================================# ================================================================

PYTHON := python3



help: ## Show this help

	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \SHELL := /bin/zsh

		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

PYTHON := python3

# ── Infrastructure ──────────────────────────────────────────

.PHONY: help build up down logs install simulate index serve serve-dev test test-search health stats benchmark clean.PHONY: help infra infra-down install index serve test clean simulate

build: ## Build all Docker images

	docker compose buildhelp: ## Show this help



up: ## Start full stack (app + qdrant + redis)	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \

	docker compose up -d

	@echo "Waiting for Qdrant..."		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

	@until curl -s http://localhost:6333/healthz > /dev/null 2>&1; do sleep 1; done

	@echo "Qdrant ready on :6333 (gRPC :6334)"SHELL := /bin/zshSHELL := /bin/zsh

	@echo "Redis ready on :6379"

	@echo "API ready on :8000"# -- Infrastructure -----------------------------------------------



down: ## Stop and remove all containers and volumesPYTHON := python3PROJECT_DIR := $(shell pwd)

	docker compose down -v

build: ## Build all Docker images

logs: ## Tail all service logs

	docker compose logs -f	docker compose buildPYTHON := python3



# ── Python Environment ──────────────────────────────────────



install: ## Install Python dependenciesup: ## Start full stack (app + qdrant + redis)help: ## Show this help

	$(PYTHON) -m pip install --upgrade pip

	$(PYTHON) -m pip install -r requirements.txt	docker compose up -d



# ── Data Pipeline ───────────────────────────────────────────	@echo "Waiting for Qdrant..."	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \help: ## Show this help



simulate: ## Generate 1M simulated vectors into Qdrant	@until curl -s http://localhost:6333/healthz > /dev/null 2>&1; do sleep 1; done

	$(PYTHON) -m scripts.simulate_dataset --num-vectors 1000000 --batch-size 10000

	@echo "Qdrant ready on :6333 (gRPC :6334)"		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \

index: ## Index real images from ./data/images into Qdrant

	$(PYTHON) -m indexing.index_images --image-dir ./data/images --batch-size 256	@echo "Redis ready on :6379"



# ── Server ──────────────────────────────────────────────────	@echo "API ready on :8000"		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'



serve: ## Start FastAPI server locally (outside Docker)

	$(PYTHON) -m uvicorn services.api_gateway.app:app \

		--host 0.0.0.0 --port 8000 --workers 1 --log-level infodown: ## Stop and remove all containers and volumes# -- Infrastructure -----------------------------------------------



serve-dev: ## Start FastAPI server with auto-reload (dev mode)	docker compose down -v

	$(PYTHON) -m uvicorn services.api_gateway.app:app \

		--host 0.0.0.0 --port 8000 --reload --log-level debug# ── Infrastructure ──────────────────────────────────────────



# ── Testing ─────────────────────────────────────────────────logs: ## Tail all service logs



test: ## Run test suite	docker compose logs -fbuild: ## Build all Docker images

	$(PYTHON) -m pytest tests/ -v --tb=short



test-search: ## Quick manual search test

	curl -s -X POST http://localhost:8000/search \# -- Python Environment -------------------------------------------	docker compose buildinfra: ## Start Qdrant + Redis via Docker Compose

		-H "Content-Type: application/json" \

		-d '{"query": "a dog playing in the park", "top_k": 5}' | python3 -m json.tool



health: ## Check service healthinstall: ## Install Python dependencies	docker compose -f infra/docker-compose.yml up -d

	curl -s http://localhost:8000/health | python3 -m json.tool

	$(PYTHON) -m pip install --upgrade pip

stats: ## Get runtime performance stats

	curl -s http://localhost:8000/stats | python3 -m json.tool	$(PYTHON) -m pip install -r requirements.txtup: ## Start full stack (app + qdrant + redis)	@echo "Waiting for Qdrant..."



# ── Benchmarks ──────────────────────────────────────────────



benchmark: ## Run search latency benchmark# -- Indexing ------------------------------------------------------	docker compose up -d	@until curl -s http://localhost:6333/healthz > /dev/null 2>&1; do sleep 1; done

	$(PYTHON) -m scripts.benchmark --iterations 1000 --concurrency 10



# ── ONNX Runtime ────────────────────────────────────────────

simulate: ## Generate 1M simulated vectors into Qdrant	@echo "Waiting for Qdrant..."	@echo "Qdrant ready on :6333 (gRPC :6334)"

onnx-convert: ## Export CLIP text encoder to ONNX format

	$(PYTHON) -m scripts.convert_to_onnx \	$(PYTHON) -m scripts.simulate_dataset --num-vectors 1000000 --batch-size 10000

		--model ViT-H-14 --pretrained laion2b_s32b_b79k \

		--output-dir models/onnx --validate	@until curl -s http://localhost:6333/healthz > /dev/null 2>&1; do sleep 1; done	@echo "Redis ready on :6379"



onnx-benchmark: ## Compare ONNX vs PyTorch inference speedindex: ## Index real images from ./data/images into Qdrant

	$(PYTHON) -m scripts.benchmark_onnx_vs_pytorch \

		--iterations 200 --output results/onnx_benchmark.csv	$(PYTHON) -m indexing.index_images --image-dir ./data/images --batch-size 256	@echo "Qdrant ready on :6333 (gRPC :6334)"



# ── Kubernetes ──────────────────────────────────────────────



k8s-apply: ## Deploy to Kubernetes (default: dev overlay)# -- Server --------------------------------------------------------	@echo "Redis ready on :6379"infra-down: ## Stop and remove infra containers

	kubectl apply -k k8s/overlays/dev



k8s-delete: ## Delete Kubernetes deployment

	kubectl delete -k k8s/overlays/devserve: ## Start FastAPI server locally (outside Docker)	@echo "API ready on :8000"	docker compose -f infra/docker-compose.yml down -v



k8s-staging: ## Deploy staging overlay	$(PYTHON) -m uvicorn services.api_gateway.app:app \

	kubectl apply -k k8s/overlays/staging

		--host 0.0.0.0 \

k8s-prod: ## Deploy production overlay

	kubectl apply -k k8s/overlays/prod		--port 8000 \



# ── Code Quality ────────────────────────────────────────────		--workers 1 \down: ## Stop and remove all containers and volumesinfra-logs: ## Tail infra logs



lint: ## Run linters		--log-level info

	$(PYTHON) -m ruff check .

	$(PYTHON) -m mypy services/ configs/ utils/ --ignore-missing-imports	docker compose down -v	docker compose -f infra/docker-compose.yml logs -f



# ── Cleanup ─────────────────────────────────────────────────serve-dev: ## Start with auto-reload (development only)



clean: ## Remove caches and build artifacts	$(PYTHON) -m uvicorn services.api_gateway.app:app \

	find . -type d -name __pycache__ -exec rm -rf {} +

	find . -type d -name .pytest_cache -exec rm -rf {} +		--host 0.0.0.0 \

	find . -name "*.pyc" -delete

	rm -rf .mypy_cache .ruff_cache dist build *.egg-info		--port 8000 \logs: ## Tail all service logs# ── Python Environment ──────────────────────────────────────


		--reload \

		--log-level debug	docker compose logs -f



# -- Testing -------------------------------------------------------install: ## Install Python dependencies



test: ## Run test suite# -- Python Environment -------------------------------------------	$(PYTHON) -m pip install --upgrade pip

	$(PYTHON) -m pytest tests/ -v --tb=short

	$(PYTHON) -m pip install -r requirements.txt

test-search: ## Smoke test: search for "a cat"

	curl -s -X POST http://localhost:8000/search \install: ## Install Python dependencies

		-H "Content-Type: application/json" \

		-d '{"query": "a cat sitting on a couch", "top_k": 5}' | python3 -m json.tool	$(PYTHON) -m pip install --upgrade pip# ── Indexing ────────────────────────────────────────────────



health: ## Check service health	$(PYTHON) -m pip install -r requirements.txt

	curl -s http://localhost:8000/health | python3 -m json.tool

simulate: ## Generate 1M simulated vectors and index them into Qdrant

stats: ## Get runtime statistics

	curl -s http://localhost:8000/stats | python3 -m json.tool# -- Indexing ------------------------------------------------------	$(PYTHON) -m scripts.simulate_dataset --num-vectors 1000000 --batch-size 10000



benchmark: ## Run latency benchmark (100 requests)

	$(PYTHON) -m scripts.benchmark --num-requests 100 --top-k 10

simulate: ## Generate 1M simulated vectors into Qdrantindex: ## Index real images from ./data/images into Qdrant

# -- Cleanup -------------------------------------------------------

	$(PYTHON) -m scripts.simulate_dataset --num-vectors 1000000 --batch-size 10000	$(PYTHON) -m indexing.index_images --image-dir ./data/images --batch-size 256

clean: ## Remove generated files and caches

	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

	find . -type f -name "*.pyc" -delete 2>/dev/null || true

	rm -rf .pytest_cache/ data/index_manifest.jsonindex: ## Index real images from ./data/images into Qdrant# ── Server ──────────────────────────────────────────────────


	$(PYTHON) -m indexing.index_images --image-dir ./data/images --batch-size 256

serve: ## Start the FastAPI server

# -- Server --------------------------------------------------------	$(PYTHON) -m uvicorn services.api_gateway.app:app \

		--host 0.0.0.0 \

serve: ## Start FastAPI server locally (outside Docker)		--port 8000 \

	$(PYTHON) -m uvicorn services.api_gateway.app:app \		--workers 1 \

		--host 0.0.0.0 \		--log-level info

		--port 8000 \

		--workers 1 \serve-dev: ## Start with auto-reload (development only)

		--log-level info	$(PYTHON) -m uvicorn services.api_gateway.app:app \

		--host 0.0.0.0 \

serve-dev: ## Start with auto-reload (development only)		--port 8000 \

	$(PYTHON) -m uvicorn services.api_gateway.app:app \		--reload \

		--host 0.0.0.0 \		--log-level debug

		--port 8000 \

		--reload \# ── Testing ─────────────────────────────────────────────────

		--log-level debug

test: ## Run tests

# -- Testing -------------------------------------------------------	$(PYTHON) -m pytest tests/ -v --tb=short



test: ## Run test suitetest-search: ## Quick smoke test: search for "a cat"

	$(PYTHON) -m pytest tests/ -v --tb=short	curl -s -X POST http://localhost:8000/search \

		-H "Content-Type: application/json" \

test-search: ## Smoke test: search for "a cat"		-d '{"query": "a cat sitting on a couch", "top_k": 5}' | python3 -m json.tool

	curl -s -X POST http://localhost:8000/search \

		-H "Content-Type: application/json" \health: ## Check service health

		-d '{"query": "a cat sitting on a couch", "top_k": 5}' | python3 -m json.tool	curl -s http://localhost:8000/health | python3 -m json.tool



health: ## Check service healthstats: ## Get runtime statistics

	curl -s http://localhost:8000/health | python3 -m json.tool	curl -s http://localhost:8000/stats | python3 -m json.tool



stats: ## Get runtime statistics# ── Cleanup ─────────────────────────────────────────────────

	curl -s http://localhost:8000/stats | python3 -m json.tool

clean: ## Remove generated files and caches

benchmark: ## Run latency benchmark (100 requests)	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

	$(PYTHON) -m scripts.benchmark --num-requests 100 --top-k 10	find . -type f -name "*.pyc" -delete 2>/dev/null || true

	rm -rf .pytest_cache/ data/index_manifest.json

# -- Cleanup -------------------------------------------------------

clean: ## Remove generated files and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ data/index_manifest.json
