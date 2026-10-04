# SimCSE interview notes

DRAFT - NEEDS HUMAN VERIFICATION

## What and why

SimCSE trains sentence embeddings by treating two dropout views of the same sentence as a positive pair. Other batch sentences provide negatives. The validation-selected checkpoint is tested without the training projection head.

## Actual worked comparison

Paper target 76.25%; our reference seed mean 75.3388%; gap -0.9112 percentage points. Trace values through RESULTS_INDEX.md to the final-test JSON; do not describe a gap as a matched-runtime reproduction.

## Five likely interview questions

1. Why two identical texts? Independent dropout creates distinct views while preserving meaning; without stochastic views, training can collapse.
2. Where are negatives from? Other sentences in the batch. False negatives can occur when different sentences mean the same thing.
3. Why remove the projection head at evaluation? Follow the official unsupervised protocol: contrastive training uses the MLP, evaluation uses CLS before it.
4. What did the ablation isolate? Weight decay with fixed data, batch size, learning rate, budget, and seeds. The paired interval describes seed variability.
5. What prevents test tuning? Frozen protocol and validation-only checkpoint selection; final tests occur after all training. Exact-overlap filtering is disclosed, but cannot prove semantic leakage is absent.
