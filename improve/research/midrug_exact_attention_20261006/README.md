# Query-deleted attention inside the M123 encoder

Final requirement remains both original MiDrug test AUC/AP +0.02 and preserved PLMF.
Only independent frozen miRNA/drug data and original-training internal folds are used.
The release implementation is unchanged. No original validation/test labels are loaded.

The two-layer original encoder keeps its message/root linear parameters and M123 head.
Each relation obtains a width-16 source/receiver scorer on fixed per-node LayerNorm
features. Its final projection starts at zero, yielding gate 0.5. Gates are bounded to
[0.1,0.9] and normalized by receiving gate mass, multiplied by degree^(1-alpha).
At alpha=0 uniform gates recover original sum messages; alpha=.5 recovers the prior
degree-scaled encoder. M3 and its training-only feature normalization remain unchanged.

M1 removes both query messages and their normalization mass. Second-layer attention
is updated only on incident edges affected by the two changed first-layer endpoints.
Their grouped state updates are propagated into the original M2 path pools. This is
actual attention under the query-deleted graph, not attention frozen from a graph
containing the query. Derivation and boundaries are in DERIVATION.md.

Six frozen configurations x three folds = 18 fits: target-only/all-relation attention,
each with reconstruction weight 0/10 at alpha=0; all-relation attention with/without
reconstruction at alpha=.5. Uniform controls reuse the previously audited M123,
unmasked10, degree-all05 and degree-all05-rec10 checkpoints. They are replayed again.
The two alpha=.5 variants are compared with their exact matching uniform degree controls.
Promotion requires both mean AUC/AP gains >=.002 over original M123, unmasked10 and
the matching uniform encoder. This gate does not replace the final +.02 test target.

Training: seed42, RMSprop lr=1e-4 alpha=.99, weight decay 1e-4 on gate/reconstruction
parameters and zero on original parameters. Same 800-epoch cap, minimum 300, patience
200 and mean internal AUC/AP checkpoint rule as prior controls. Reconstruction uses
the prior unmasked half-node targets, independent generator and 50-epoch weight ramp.

Initial verification:
- 240 NumPy forward comparisons (historical prototype).
- 12 float64 comparisons of nonuniform-gate predictions and parameter gradients against
  explicitly deleted graphs, covering target/all gates, alpha=0/.5, degree-one and nonedges.
- Initial uniform equivalence to original and alpha=.5 encoders.
- Real-data explicit M1/M2/M3/final-logit deletion, finite nonzero gate gradients and
  heldout-label invariance. CUDA uses numerical tolerances, not bitwise equivalence.
- A full training step including reconstruction peaks at about 3.51 GiB on an A40.

Input geometry is cached by query values on the immutable graph. Duplicate same-type
edges retain separate event indices. Model audits also replay selected checkpoints,
reconstruct loss/epoch selection, check gate behavior and verify protected PLMF hashes.

Reproduce with ncRNAdrug Python and OMP/MKL/OPENBLAS_NUM_THREADS=2:
check.py -> launch.py -> audit.py -> summarize.py.
