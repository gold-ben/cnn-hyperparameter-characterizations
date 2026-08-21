# Experimental process

The manuscript uses nine CNN architecture and training factors: batch size,
dropout, batch normalization, max pooling, initialization, convolutional width,
convolutional depth, fully-connected width, and fully-connected depth.

The `2^(9-3)` Resolution-IV design contains 64 factorial configurations. Sixteen
additional configurations set all five quantitative factors to their coded
midpoints while retaining every combination of the four two-level factors.
Together these form the 80 rows in `experiment_list.csv`.

Each setting is evaluated on CIFAR-10 and CIFAR-100 using SGD, SGD-Nesterov,
Adam, AdamW, and RMSProp with 10 seed replicates. Each seed supplies a distinct
network initialization and dataset shuffle. LeakyReLU is fixed throughout all
CNN architectures.

For each experimental run, the code:

1. constructs and initializes the assigned CNN;
2. trains it for 10 epochs and records the initialization and final training
   losses;
3. evaluates 10-epoch test accuracy; and
4. generates a Gaussian random path from the trained parameter vector.

At each path step, a 1,024-image training sample is used to estimate loss and
gradient. A full-network Gaussian direction is normalized to unit Euclidean
norm and scaled by the fixed step size. The path is limited to 500 steps and
terminates early when its loss exceeds the initialization loss. The maximum
empirical gradient-Lipschitz ratio observed along the path is the curvature
response for the run.

Generate the design and run the two datasets with 10 seed replicates:

```bash
python generate_matrix.py
python main.py --dataset-coded CIFAR10 --num-seeds 10
python main.py --dataset-coded CIFAR100 --num-seeds 10
```
