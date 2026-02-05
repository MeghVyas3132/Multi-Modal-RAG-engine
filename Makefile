# ================================================================# ================================================================

# Makefile -- Common development tasks# Makefile — Common development tasks

# ================================================================# ================================================================



.PHONY: help build up down logs install simulate index serve serve-dev test test-search health stats benchmark clean.PHONY: help infra infra-down install index serve test clean simulate



SHELL := /bin/zshSHELL := /bin/zsh

PYTHON := python3PROJECT_DIR := $(shell pwd)

PYTHON := python3

help: ## Show this help

	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \help: ## Show this help

		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \

		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# -- Infrastructure -----------------------------------------------

# ── Infrastructure ──────────────────────────────────────────

build: ## Build all Docker images

	docker compose buildinfra: ## Start Qdrant + Redis via Docker Compose

	docker compose -f infra/docker-compose.yml up -d

up: ## Start full stack (app + qdrant + redis)	@echo "Waiting for Qdrant..."

	docker compose up -d	@until curl -s http://localhost:6333/healthz > /dev/null 2>&1; do sleep 1; done

	@echo "Waiting for Qdrant..."	@echo "Qdrant ready on :6333 (gRPC :6334)"

	@until curl -s http://localhost:6333/healthz > /dev/null 2>&1; do sleep 1; done	@echo "Redis ready on :6379"

	@echo "Qdrant ready on :6333 (gRPC :6334)"

	@echo "Redis ready on :6379"infra-down: ## Stop and remove infra containers

	@echo "API ready on :8000"	docker compose -f infra/docker-compose.yml down -v



down: ## Stop and remove all containers and volumesinfra-logs: ## Tail infra logs

	docker compose down -v	docker compose -f infra/docker-compose.yml logs -f



logs: ## Tail all service logs# ── Python Environment ──────────────────────────────────────

	docker compose logs -f

install: ## Install Python dependencies

# -- Python Environment -------------------------------------------	$(PYTHON) -m pip install --upgrade pip

	$(PYTHON) -m pip install -r requirements.txt

install: ## Install Python dependencies

	$(PYTHON) -m pip install --upgrade pip# ── Indexing ────────────────────────────────────────────────

	$(PYTHON) -m pip install -r requirements.txt

simulate: ## Generate 1M simulated vectors and index them into Qdrant

# -- Indexing ------------------------------------------------------	$(PYTHON) -m scripts.simulate_dataset --num-vectors 1000000 --batch-size 10000



simulate: ## Generate 1M simulated vectors into Qdrantindex: ## Index real images from ./data/images into Qdrant

	$(PYTHON) -m scripts.simulate_dataset --num-vectors 1000000 --batch-size 10000	$(PYTHON) -m indexing.index_images --image-dir ./data/images --batch-size 256



index: ## Index real images from ./data/images into Qdrant# ── Server ──────────────────────────────────────────────────

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
