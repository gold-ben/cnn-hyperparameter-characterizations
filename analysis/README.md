# Statistical analysis process

The manuscript analysis uses 8,000 observations: 4,000 from CIFAR-10 and 4,000
from CIFAR-100. Each observation identifies one of 80 design settings, one of
five optimization routines, and one of 10 seed replicates, together with the
nine coded factors, maximum Gaussian empirical Lipschitz estimate, and
10-epoch test accuracy.

The analysis performs the following steps:

1. For each dataset, optimizer, and response, apply a Box-Cox transformation
   and fit an OLS model containing nine main effects, the estimable two-factor
   interaction terms or alias classes, seed blocks, and one joint-center
   departure. This produces 20 primary models.
2. Apply a within-model Bonferroni correction across the 41 tested treatment
   contrasts.
3. Report joint-center departures, missing Gaussian responses, variance
   inflation factors, and retrospective omnibus power.
4. For each dataset, use the five optimizer-specific test-accuracy models to
   predict accuracy on the original scale and select the optimizer with the
   largest predicted accuracy for each of the 80 settings.
5. For each dataset-optimizer combination, calculate the Pearson correlation
   between the Box-Cox-transformed Gaussian Lipschitz and test-accuracy
   responses using complete response pairs, with a Bonferroni correction across
   the 10 correlation tests.
6. Generate the manuscript tables, figures, diagnostics, and LaTeX.

Run the complete analysis from the package root:

```bash
python analysis/run_all.py
```

The command reads the two JSONL files under `data/` and writes regenerated
artifacts under `outputs/`.
