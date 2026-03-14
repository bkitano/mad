.PHONY: help build push run run-dry run-idle stop clean deploy-modal

.DEFAULT_GOAL := help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

build: ## Build the worker Docker image
	docker build -f worker/Dockerfile -t mad-worker:latest ..

push: build ## Build and push worker image to GHCR
	docker tag mad-worker:latest ghcr.io/bkitano/mad-worker:latest
	docker push ghcr.io/bkitano/mad-worker:latest

run: ## Run worker in Docker
	docker run --rm --env-file ../../.env --name mad-worker mad-worker:latest

run-dry: ## Run worker once in dry-run mode
	docker run --rm --env-file ../../.env --name mad-worker mad-worker:latest --dry-run --once

run-idle: ## Run worker in idle mode (port 4096)
	docker run --rm --env-file ../../.env -p 4096:4096 --name mad-worker mad-worker:latest --idle

stop: ## Stop worker container
	-docker stop mad-worker 2>/dev/null

clean: ## Remove worker Docker image
	-docker rmi mad-worker:latest 2>/dev/null

deploy-modal: ## Deploy worker to Modal
	cd .. && uv run python -m modal deploy worker.modal_worker
