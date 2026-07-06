.PHONY: test train train-dbg lint format docker-build docker-test docker-train docker-shell prepare-europarl run-deepsc-text docker-run-deepsc-text

test:
	pytest

train-dbg:
	python scripts/train.py

train:
	python scripts/train.py -c $(CP)

lint:
	ruff check --fix src tests scripts

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

prepare-europarl:
	python src/semcom/data/prepare_europarl.py --input-dir $(IN)

run-deepsc-text:
	python scripts/run_deepsc_text.py -c $(CP)

docker-run-deepsc-text:
	docker compose run --rm semcom make run-deepsc-text CP=$(CP)
