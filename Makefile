SHELL :=/bin/bash

POSTS_DIR=hugo/content/posts
IPYNB_FILES=$(wildcard notebooks/*.ipynb)
MARKDOWN_FILES_TO_BUILD=$(patsubst notebooks/%.ipynb, $(POSTS_DIR)/%.md, $(IPYNB_FILES))
RUN=poetry run

# Rule to build markdown files from jupyter notebooks
$(POSTS_DIR)/%.md: notebooks/%.ipynb
	@echo Building '$<' in $@
	$(RUN) nb2hugo --site-dir hugo --section posts $<

.PHONY: bootstrap
bootstrap:
	$(shell curl -fsSL https://raw.githubusercontent.com/Linuxbrew/install/master/install.sh)
	brew install hugo
	pipx install nb2hugo

build-notebooks: $(MARKDOWN_FILES_TO_BUILD)

.PHONY: build-notebooks
build: build-notebooks
	@echo "Building the final website."
	hugo --source hugo

.PHONY: serve
serve: build-notebooks
	hugo serve --buildDrafts --source hugo

.PHONY: publish
publish: build
	$(shell git --git-dir=hugo/public/.git add -A)
	$(shell git --git-dir=hugo/public/.git diff-index --quiet HEAD || git --git-dir=hugo/public/.git commit -m "Update site")
	$(shell git --git-dir=hugo/public/.git push)

.DEFAULT_GOAL := default
default: build
	@echo "Use:"
	@echo "- 'make serve' to view the contents"
	@echo "- 'make publish' to publish the contents"
