.PHONY: test train train-dbg

test:
	pytest

train-dbg:
	python scripts/train.py

train:
	python scripts/train.py -c $(CP)
