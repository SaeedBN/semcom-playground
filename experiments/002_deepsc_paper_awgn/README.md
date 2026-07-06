# Experiment 002 — DeepSC over AWGN

This experiment implements a version of the DeepSC text semantic communication system from:

> H. Xie, Z. Qin, G. Y. Li, and B.-H. Juang,
> "Deep Learning Enabled Semantic Communication Systems."

## Alignment

This experiment follows the paper in three main aspects:

1. Dataset: English Europarl proceedings text.
2. Preprocessing: keep sentences with 4 to 30 words.
3. Channel/model: AWGN channel with a DeepSC-style Transformer semantic encoder/decoder and dense channel encoder/decoder.

## Pipeline

```text
sentence
→ tokenization
→ semantic encoder
→ channel encoder
→ AWGN channel
→ channel decoder
→ semantic decoder
→ reconstructed sentence
```

## Dataset

Expected processed dataset path:

```text
dataset/processed/europarl/europarl_en.txt
```

Prepare dataset:

```bash
make prepare-europarl IN=dataset/raw/europarl/
```

Or any directory you have downloaded the files into.

## Run

```bash
make run-deepsc-text EXP=experiments/002_deepsc_paper_awgn/config.yaml
```
