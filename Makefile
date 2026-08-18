# DARTsim — build, test, and reproduce the paper's evaluation.

CARGO   ?= cargo
PYTHON  ?= python3
FIGDIR  ?= figures

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk -F':.*?## ' '{printf "  \033[1m%-12s\033[0m %s\n", $$1, $$2}'

.PHONY: build
build: ## Build the simulator in release mode
	$(CARGO) build --release

.PHONY: test
test: ## Run the test suite
	$(CARGO) test --release

.PHONY: lint
lint: ## Check formatting and run clippy
	$(CARGO) fmt --check
	$(CARGO) clippy --all-targets -- -D warnings

.PHONY: figures
figures: build ## Reproduce every figure in the paper
	$(PYTHON) -m reproduce --output $(FIGDIR)

.PHONY: list
list: ## List the figures that can be built
	@$(PYTHON) -m reproduce --list

.PHONY: clean
clean: ## Remove build output
	$(CARGO) clean

.PHONY: clean-cache
clean-cache: ## Drop cached simulation results, forcing a full recomputation
	rm -rf .cache
