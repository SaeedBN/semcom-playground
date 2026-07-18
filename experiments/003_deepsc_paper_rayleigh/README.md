# Experiment 003 — DeepSC Rayleigh

DeepSC-based text semantic communication over a Rayleigh fading channel.

Paper: <https://arxiv.org/abs/2006.10685>

## Config

```text
experiments/003_deepsc_paper_rayleigh/config.yaml
```

## Dataset

```text
dataset/processed/europarl/
```

## Run

```bash
make run-deepsc-text CP=experiments/003_deepsc_paper_rayleigh/config.yaml
```

## Outputs

```text
results/003_deepsc_paper_rayleigh/
```

## Plots

![Training metrics](plots/training_metrics.png)

![Evaluation metrics](plots/evaluation_metrics.png)

## Sample Reconstructions

| SNR (dB) | Original | Reconstructed |
|---:|---|---|
| 0 | i approve of the proposed amendments which once again highlight galileo s importance as a strictly civilian project and reject any possibility of using space for military purposes . | we would having voted to accept which those but we strongly believe cannot do a non border and support for any of such as you . |
| 3 | i approve of the proposed amendments which once again highlight galileo s importance as a strictly civilian project and reject any possibility of using space for military purposes . | i have received that amendments nos which my many has negotiated by commissioner as a substantial solution and regulation on any event of setting up as regards people . |
| 6 | i approve of the proposed amendments which once again highlight galileo s importance as a strictly civilian project and reject any possibility of using space for military purposes . | i suggest the amendments proposed by the given she would be implemented as a rather than and oppose discrimination without any event of a military service in april . |
| 9 | i approve of the proposed amendments which once again highlight galileo s importance as a strictly civilian project and reject any possibility of using space for military purposes . | i voted of the proposed amendments which given this particularly own initiative has been a rather effective and and monitoring any event of any basis for coming from coming . |
| 12 | i approve of the proposed amendments which once again highlight galileo s importance as a strictly civilian project and reject any possibility of using space for military purposes . | i voted of the proposed amendments which once again just s project has as a rather than cross border protection and any provision of service for urgent subjects . |
| 15 | i approve of the proposed amendments which once again highlight galileo s importance as a strictly civilian project and reject any possibility of using space for military purposes . | i voted of the proposed amendments which given their aspects of the importance as a rather and cross border regulation and any event of any similar assistance . |
| 18 | i approve of the proposed amendments which once again highlight galileo s importance as a strictly civilian project and reject any possibility of using space for military purposes . | i believe of the amendments which contains of once particularly cross border importance as a rather than cross border and regulation ec of electronic equipment for military purposes . |
