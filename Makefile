.PHONY: clear
clear: ## Clear the workspace
	rm -rf ./bin

.PHONY: lint
lint: ## Run linter
	golangci-lint run

.PHONY: fix
fix: ## Fix lint violations
	golangci-lint run --fix

.PHONY: test
test: ## Run all tests
	mkdir -p bin
	go test  -count=1 `go list ./... | grep --invert-match regressiontest` -race -covermode=atomic -coverprofile=bin/coverage.out ./...

.PHONY: coverage
coverage: test ## Run all tests and open covergae as a html
	go tool cover -html=bin/coverage.out


# https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
.PHONY: help
.DEFAULT_GOAL := help
help:
	@grep -h -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
