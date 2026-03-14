.PHONY: help build-all stop clean

.DEFAULT_GOAL := help

help: ## Show this help
	@echo "Root targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "API targets (make -C api):"
	@$(MAKE) -C api help --no-print-directory
	@echo ""
	@echo "Worker targets (make -C worker):"
	@$(MAKE) -C worker help --no-print-directory

build-all: ## Build all Docker images
	$(MAKE) -C api build
	$(MAKE) -C worker build

stop: ## Stop all running containers
	-docker stop mad-api mad-worker 2>/dev/null

clean: ## Remove all Docker images
	-docker rmi mad-api:latest mad-worker:latest 2>/dev/null
