%.ipynb:
	nb2hugo --site-dir hugo --section posts $@

.PHONY: bootstrap
bootstrap:
	$(shell curl -fsSL https://raw.githubusercontent.com/Linuxbrew/install/master/install.sh)
	brew install hugo
	pipx install nb2hugo

build-notebooks: notebooks/*.ipynb

.PHONY: build
build: build-notebooks
	@echo "Building"
	hugo --source hugo

.PHONY: serve
serve: build-notebooks
	hugo serve --buildDrafts --source hugo

.PHONY: publish
publish: build
	cd hugo/public
	git add .
	git commit
	git push

.DEFAULT_GOAL := default
default: build
	@echo "Default"
