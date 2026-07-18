# Experiment 004 — DeepSC Rician

DeepSC-style text semantic communication over a Rician fading channel.

Paper: <https://arxiv.org/abs/2006.10685>

## Config

```text
experiments/004_deepsc_paper_rician/config.yaml
```

## Dataset

```text
dataset/processed/europarl/
```

## Run

```bash
make run-deepsc-text CP=experiments/004_deepsc_paper_rician/config.yaml
```

## Outputs

```text
results/004_deepsc_paper_rician/
```

## Plots

![Training metrics](plots/training_metrics.png)

![Evaluation metrics](plots/evaluation_metrics.png)

## Sample Reconstructions

| SNR (dB) | Original | Reconstructed |
|---:|---|---|
| 0 | i approve of the proposed amendments which once again highlight galileo s importance as a strictly civilian project and reject any possibility of using space for military purposes . | regarding this step up amendment which we have voted by mr brok rural principles on a lack of decisions which must take of european consumer protection in their rights . |
| 3 | i approve of the proposed amendments which once again highlight galileo s importance as a strictly civilian project and reject any possibility of using space for military purposes . | i am convinced that additional amendments which their own project directive with a few ones s proposals and set out any means of fisheries policy with which have been . |
| 6 | i approve of the proposed amendments which once again highlight galileo s importance as a strictly civilian project and reject any possibility of using space for military purposes . | i understand the amendments the proposed amendments which again stated by mr langen s delay a proper mechanism and reject any series of consumer legislation for current rights . |
| 9 | i approve of the proposed amendments which once again highlight galileo s importance as a strictly civilian project and reject any possibility of using space for military purposes . | i feel that the proposed amendments which again their directive s shared importance on a maximum form and implementation provides any possibility of using common standards for food . |
| 12 | i approve of the proposed amendments which once again highlight galileo s importance as a strictly civilian project and reject any possibility of using space for military purposes . | i approve of the proposed amendments which again their own practices report as a remain a limited project and put any point of liberalisation of military purposes . |
| 15 | i approve of the proposed amendments which once again highlight galileo s importance as a strictly civilian project and reject any possibility of using space for military purposes . | i approve of the proposed amendments which their directive which promote s importance as a stronger project generally and reject any possibility of using military purposes . |
| 18 | i approve of the proposed amendments which once again highlight galileo s importance as a strictly civilian project and reject any possibility of using space for military purposes . | i recommend the number of amendments which their aspects again firmly s importance as a limited nature and project reject any possibility of using common duty for peace . |
