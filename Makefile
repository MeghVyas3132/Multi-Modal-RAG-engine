# ================================================================# ================================================================# ================================================================

# Makefile -- Common development tasks

# ================================================================# Makefile -- Common development tasks# Makefile — Common development tasks



.PHONY: help build up down logs install simulate index serve serve-dev test test-search health stats benchmark clean# ================================================================# ================================================================



SHELL := /bin/zsh

PYTHON := python3

.PHONY: help build up down logs install simulate index serve serve-dev test test-search health stats benchmark clean.PHONY: help infra infra-down install index serve test clean simulate

help: ## Show this help

	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \

		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

SHELL := /bin/zshSHELL := /bin/zsh

# -- Infrastructure -----------------------------------------------

PYTHON := python3PROJECT_DIR := $(shell pwd)

build: ## Build all Docker images

	docker compose buildPYTHON := python3



up: ## Start full stack (app + qdrant + redis)help: ## Show this help

	docker compose up -d

	@echo "Waiting for Qdrant..."	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \help: ## Show this help

	@until curl -s http://localhost:6333/healthz > /dev/null 2>&1; do sleep 1; done

	@echo "Qdrant ready on :6333 (gRPC :6334)"		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \

	@echo "Redis ready on :6379"

	@echo "API ready on :8000"		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'



down: ## Stop and remove all containers and volumes# -- Infrastructure -----------------------------------------------

	docker compose down -v

# ── Infrastructure ──────────────────────────────────────────

logs: ## Tail all service logs

	docker compose logs -fbuild: ## Build all Docker images



# -- Python Environment -------------------------------------------	docker compose buildinfra: ## Start Qdrant + Redis via Docker Compose



install: ## Install Python dependencies	docker compose -f infra/docker-compose.yml up -d

	$(PYTHON) -m pip install --upgrade pip

	$(PYTHON) -m pip install -r requirements.txtup: ## Start full stack (app + qdrant + redis)	@echo "Waiting for Qdrant..."



# -- Indexing ------------------------------------------------------	docker compose up -d	@until curl -s http://localhost:6333/healthz > /dev/null 2>&1; do sleep 1; done



simulate: ## Generate 1M simulated vectors into Qdrant	@echo "Waiting for Qdrant..."	@echo "Qdrant ready on :6333 (gRPC :6334)"

	$(PYTHON) -m scripts.simulate_dataset --num-vectors 1000000 --batch-size 10000

	@until curl -s http://localhost:6333/healthz > /dev/null 2>&1; do sleep 1; done	@echo "Redis ready on :6379"

index: ## Index real images from ./data/images into Qdrant

	$(PYTHON) -m indexing.index_images --image-dir ./data/images --batch-size 256	@echo "Qdrant ready on :6333 (gRPC :6334)"



# -- Server --------------------------------------------------------	@echo "Redis ready on :6379"infra-down: ## Stop and remove infra containers



serve: ## Start FastAPI server locally (outside Docker)	@echo "API ready on :8000"	docker compose -f infra/docker-compose.yml down -v

	$(PYTHON) -m uvicorn services.api_gateway.app:app \

		--host 0.0.0.0 \

		--port 8000 \

		--workers 1 \down: ## Stop and remove all containers and volumesinfra-logs: ## Tail infra logs

		--log-level info

	docker compose down -v	docker compose -f infra/docker-compose.yml logs -f

serve-dev: ## Start with auto-reload (development only)

	$(PYTHON) -m uvicorn services.api_gateway.app:app \

		--host 0.0.0.0 \

		--port 8000 \logs: ## Tail all service logs# ── Python Environment ──────────────────────────────────────

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
