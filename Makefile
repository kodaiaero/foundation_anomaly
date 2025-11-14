ifneq (,$(wildcard .env))
	include .env
	export $(shell sed 's/=.*//' .env)
endif

REGION        ?= noto
VERSION       ?= v1
SEED          ?= 42
PER_GROUP     ?= 50
SELECTED_FILE ?= auto
SCRIPT        := scripts.freeze_training_set_region
PYTHON        := python

.PHONY: freeze-dry freeze show-env

# dry-runで動作確認
freeze-dry:
	$(PYTHON) -m $(SCRIPT) \
		--region $(REGION) \
		--selected $(SELECTED_FILE) \
		--per-group $(PER_GROUP) \
		--seed $(SEED) \
		--dry-run

# 実際に実行してtrainingを確定
freeze:
	$(PYTHON) -m $(SCRIPT) \
		--region $(REGION) \
		--selected $(SELECTED_FILE) \
		--per-group $(PER_GROUP) \
		--seed $(SEED)

show-env:
	@echo "REGION        = $(REGION)"
	@echo "VERSION       = $(VERSION)"
	@echo "SEED          = $(SEED)"
	@echo "PER_GROUP     = $(PER_GROUP)"
	@echo "SELECTED_FILE = $(SELECTED_FILE)"
