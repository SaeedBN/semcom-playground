.PHONY: test train

test:
	pytest

train:
	python scripts/train.py