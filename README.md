# CNN loss-surface manuscript reproduction

The repository contains materials supporting reproduction of the experimental
design, CNN training procedures, empirical loss-surface curvature calculations,
and statistical analyses reported in this manuscript.

## Manuscript process

The study uses nine CNN architecture and training factors in a
`2^(9-3)` Resolution-IV fractional-factorial design. The design contains 64
factorial configurations and 16 center configurations, for 80 settings in
total. Each setting is evaluated on CIFAR-10 and CIFAR-100 with SGD,
SGD-Nesterov, Adam, AdamW, and RMSProp across 10 seed replicates. This produces
4,000 configuration-optimizer-seed observations per dataset and 8,000
observations overall.

For each run, a CNN is constructed from the assigned factor settings with
LeakyReLU fixed throughout the architecture. The network is trained for 10
epochs and evaluated on the corresponding test set. Starting from the trained
parameter vector, a Gaussian random path samples the surrounding local loss
surface. Each full-network Gaussian direction is normalized and scaled to the
fixed step size. A path contains at most 500 steps and terminates early if its
loss exceeds the loss measured at initialization. The maximum empirical
gradient-Lipschitz ratio along the path is retained as the curvature response.

The statistical analysis models Gaussian Lipschitz curvature and 10-epoch test
accuracy. It fits 20 primary Box-Cox OLS models containing the nine main
effects, estimable two-factor interaction terms or alias classes, seed blocks,
and one joint-center departure. The analysis also evaluates missing Gaussian
responses, VIFs, retrospective omnibus power, Gaussian-curvature/accuracy
correlations, and one optimizer-selection meta-model per dataset.

## Run the experiment process

From `experiment/`, generate the 80-setting design and run each dataset with 10
seed replicates:

```bash
python generate_matrix.py
python main.py --dataset-coded CIFAR10 --num-seeds 10
python main.py --dataset-coded CIFAR100 --num-seeds 10
```

## Run the manuscript analysis

From this directory:

```bash
python analysis/run_all.py
```

The analysis reads `data/cifar10_nine_factor.jsonl` and
`data/cifar100_nine_factor.jsonl` and writes the regenerated manuscript
artifacts under `outputs/`.
