# semcom-playground

An experimental playground for AI-native wireless communication through
semantic transmission. The repository currently focuses on text semantic
communication with PyTorch models, configurable wireless channels, and
reproducible DeepSC-style experiments.

> **Status:** This project is under active development. APIs, configurations,
> and experiment results may change.

## Features

- Text semantic autoencoder and DeepSC-style Transformer models
- AWGN, Rayleigh, and Rician channel simulations
- Europarl preprocessing, vocabulary generation, and token encoding
- Config-driven training and evaluation
- BLEU, sentence-similarity, token-accuracy, and sequence-accuracy metrics
- Training and evaluation plots with sample sentence reconstructions
- Local and Docker-based workflows
- Tests, formatting, and linting through `pytest` and Ruff

## Requirements

- Python 3.11 or newer
- `make`
- Docker with Docker Compose (optional)

## Installation

Create and activate a virtual environment, then install the project:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

## Dataset

The DeepSC experiments use English text files from the Europarl dataset.
Prepare the dataset with:

```bash
make prepare-europarl IN=dataset/raw/europarl/
```

`IN` can be any directory containing the dataset's `.txt` files. The script
searches that directory recursively.

The default output directory is:

```text
dataset/processed/europarl/
```

To use a different output directory, run the preparation script directly:

```bash
python src/semcom/data/prepare_europarl.py \
  --input-dir dataset/raw/europarl/ \
  --output-path path/to/output/
```

If you change the default output directory, update `dataset.text_path` in the
experiment's `config.yaml` to use the same path.

The prepared directory contains:

- `encoded_data.pkl`: tokenized and encoded sentences
- `vocab.json`: token-to-index vocabulary
- `sentences.txt`: normalized sentences

## Experiments

| Experiment | Model and channel | Dataset |
|---|---|---|
| [001 — Text AWGN Debug](experiments/001_text_awgn_debug/) | Semantic autoencoder, AWGN | Built-in toy text |
| [002 — DeepSC AWGN](experiments/002_deepsc_paper_awgn/) | DeepSC-style model, AWGN | Europarl |
| [003 — DeepSC Rayleigh](experiments/003_deepsc_paper_rayleigh/) | DeepSC-style model, Rayleigh | Europarl |
| [004 — DeepSC Rician](experiments/004_deepsc_paper_rician/) | DeepSC-style model, Rician | Europarl |

Run the debug experiment:

```bash
make train CP=experiments/001_text_awgn_debug/config.yaml
```

Run a DeepSC experiment:

```bash
make run-deepsc-text CP=experiments/002_deepsc_paper_awgn/config.yaml
```

Replace `CP` with the configuration path for experiment 003 or 004 to evaluate
another channel. Each experiment README contains its specific commands,
outputs, plots, and sample reconstructions.

## Configuration

Experiment YAML files control the model architecture, channel, dataset,
training, evaluation, tracking, and output directory. The reusable debug
configuration groups live under `configs/`.

Common settings include:

- `channel.name`: `awgn`, `rayleigh`, or `rician`
- `dataset.text_path`: prepared dataset directory
- `training.device`: training device, such as `cpu` or `cuda`
- `training.snr_db_min` and `training.snr_db_max`: training SNR range
- `evaluation.snr_values_db`: SNR values used during evaluation
- `experiment.output_dir`: result directory

## Outputs

Runs write artifacts to the configured `experiment.output_dir`, typically
under `results/`. DeepSC runs produce metrics, plots, and reconstruction
examples; the experiment directories also contain the corresponding
configuration and documented results.

## Docker

Build the development image:

```bash
make docker-build
```

Run tests in Docker:

```bash
make docker-test
```

Run a DeepSC experiment in Docker:

```bash
make docker-run-deepsc-text \
  CP=experiments/002_deepsc_paper_awgn/config.yaml
```

Open a shell in the container:

```bash
make docker-shell
```

## Development

Run the test suite:

```bash
make test
```

Lint and automatically fix supported issues:

```bash
make lint
```

Format the source, tests, and scripts:

```bash
make format
```

## Project Structure

```text
configs/       Reusable configuration groups
dataset/       Raw and processed local datasets
experiments/   Experiment configurations, documentation, and artifacts
results/       Generated run outputs
scripts/       Training and DeepSC experiment entry points
src/semcom/    Channels, datasets, models, evaluation, and utilities
tests/         Unit tests
```

## Reference

The DeepSC experiments are based on:

Xie et al., [Deep Learning Enabled Semantic Communication
Systems](https://arxiv.org/abs/2006.10685).
