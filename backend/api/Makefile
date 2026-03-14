.PHONY: help build dev run stop clean test test-all

.DEFAULT_GOAL := help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Build the API Docker image
	docker build -f Dockerfile -t mad-api:latest ..

dev: ## Run API locally with uvicorn (port 8001)
	cd .. && set -a; source ../.env; uv run uvicorn app:app --port 8001

run: ## Run API in Docker (port 8000)
	docker run --rm -p 8000:8000 --env-file ../../.env --name mad-api mad-api:latest

stop: ## Stop API container
	-docker stop mad-api 2>/dev/null

clean: ## Remove API Docker image
	-docker rmi mad-api:latest 2>/dev/null

test: ## Run core tests
	cd .. && uv run python -m pytest tests/test_opencode_types.py tests/test_opencode_service.py -v

test-all: ## Run all tests
	cd .. && uv run python -m pytest tests/ -v
