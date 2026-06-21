.PHONY: test train train-dbg lint format docker-build docker-test docker-train docker-shell

test:
	pytest

train-dbg:
	python scripts/train.py

train:
	python scripts/train.py -c $(CP)

lint:
	ruff check src tests scripts

format:
	ruff format src tests scripts

docker-build:
	docker compose build

docker-test:
	docker compose run --rm semcom make test

docker-train:
	docker compose run --rm semcom make train CP=$(CP)

docker-shell:
	docker compose run --rm semcom bash
