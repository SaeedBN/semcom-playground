# 005 Text Embedding Retrieval OFDM

Digital text embedding retrieval over an OFDM link with an AWGN propagation model.

Current PHY assumption:
- SISO only (`num_tx=1`, `num_streams_per_tx=1`, single receive antenna)

Run:

```bash
make run-text-embed-eval CP=experiments/005_text_embedding_retrieval_ofdm/config.yaml
```

Config:

```text
experiments/005_text_embedding_retrieval_ofdm/config.yaml
```
