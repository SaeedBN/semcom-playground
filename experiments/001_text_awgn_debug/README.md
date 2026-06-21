# Experiment 001 — Text Semantic Communication over AWGN

This experiment trains a simple Transformer-based text semantic autoencoder over an AWGN channel. It is mainly for debugging and setting up the project.

## Goal

Verifying the pipeline:

token IDs
→ semantic encoder
→ latent representation
→ AWGN channel
→ semantic decoder
→ reconstructed token IDs

## Metircs

- Reconstruction loss (Cross Entropy)
- Token Accuracy
- Sequence Accuracy

## Run

python scripts/train.py --config-path experiments/001_text_awgn_debug/config.yaml
