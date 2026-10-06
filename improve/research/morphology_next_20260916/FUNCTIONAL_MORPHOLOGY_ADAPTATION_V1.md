# Functional target identity in cellular morphology

This is a narrower, adaptive revision of the morphology family, motivated by
the already observed `no_alignment` arm. It is not an independent discovery or
an additional mechanism family. The source paper and external data provenance
remain those in `MORPHOLOGY_ADAPTATION_V1.md` and the sealed parent protocol.

The hypothesis is that the identities of experimentally annotated functional
miRNA targets connect CRISPR phenotypes to drug phenotypes usefully for MiDrug.
Full uses the real target identities; the sole mechanism control uses the
already frozen global gene permutation. Both preserve per-miRNA target counts,
profile coverage, network capacity, zero auxiliary alignment weight, random
number consumption, and training budget. Neither receives the project graph.
The mean of target-gene embeddings is a proxy, not a measured miRNA intervention.

The full screen weights are exactly the parent's no-alignment weights. Their
reuse requires input identity, complete training-state equivalence checks and
prediction replay. Copied models do not count as new training. The new control
has zero DTI auxiliary weight too, unlike the parent's shuffled control.

Each new fit uses 400 AdamW updates, learning rates 0.001 and 0.003 for screening,
and the existing checkpoint selection rule. Seed 666 screens, seeds 0 and 1
confirm. The MiDrug score is the frozen strong reference plus 0.1 times the
bounded phenotype residual. Missing phenotype coverage gives exactly zero
residual. LncDrug is the reference identity, fixed before this revision runs.

Required evidence: both mean global MiDrug ROC AUC and AP improve, both are
positive for at least two of three initialization seeds, and both beat the
identity-permuted control. Then the same candidate must pass the existing three
development partitions and frozen historical outer recheck. The original test
set is not used. The development partitions overlap and have been repeatedly
used; they are not independent biological replication. A 0.001 LncDrug
tolerance is the inherited engineering criterion, not a statistical margin.

The parent alignment claim remains failed. A positive result here would support
only the narrower functional-target/phenotype contribution, not DTI alignment,
causal miRNA perturbation effects, or independence of the screening evidence.
