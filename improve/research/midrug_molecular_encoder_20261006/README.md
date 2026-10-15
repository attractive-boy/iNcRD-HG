# Original molecular structure as a trainable drug representation

The original goal remains MiDrug test AUC and AP both +0.02, with unchanged PLMF results.
No lncRNA or external observations are permitted. M1/M2/M3 and the original association
graph are retained. Atom/bond graphs are internal descriptors of the existing 216 drugs;
they do not add new entities or supervision to the miRNA/drug association graph.

prepare.py verifies the original Drug.csv/SMILES/ChemBERTa row mapping and historical
hashes. It retains all components, including salts, and records atom/bond categorical
features. There are 7,094 atoms and 15,396 directed bonds; 12 drugs have multiple
components, and molecule sizes range from 1 to 551 atoms. No new pretrained weights
are downloaded or used. The existing original ChemBERTa input matrix remains allowed.

Each candidate retains the prior best unmasked reconstruction weight 10. A drug branch
combines a projected query-deleted miRNA endpoint (64 dimensions) with a drug vector,
their product and absolute difference. A 128-unit MLP produces an additive branch logit.
Its last layer starts at zero, so initial association predictions equal the original
reconstruction-only model. A branch BCE weight 0.1 encourages the branch to carry signal.
Branch-only predictions are diagnostic outputs of a jointly trained model, not an
independently fitted classifier or a standalone performance guarantee.

Three modes, all width 64 / three residual MLP layers / dropout 0.1:

- gnn: summed categorical atom embeddings, edge-aware positive messages, residual
  MLP/LayerNorm layers, mean pooling per drug, then drug projection.
- bag: exactly the same parameters and atom features as gnn, but no bond messages.
  Atom degree/ring/aromatic features still encode local structural properties; this
  is a message-passing ablation, not a completely topology-free chemical descriptor.
- chem: original ChemBERTa features projected into the same three residual layers.
  All modes also receive the same log(atom count) scalar at the final drug projection.

gnn and bag have identical initialization and parameter counts. All three modes share
identical initial RNA/pair heads. ChemBERTa input projection has a different parameter
count, which is reported rather than described as an exactly matched-capacity control.
Two branch weight decays (1e-4/1e-3) give six configs and 18 internal-fold fits.
Backbone lr=1e-4, reconstruction lr=1e-4 wd=1e-4, branch lr=3e-4; RMSprop alpha=.99.
Other stopping/reconstruction settings match the existing controls (800 cap, minimum
300, patience 200, mean internal AUC/AP checkpoint score; independent mask generator).

Both mean AUC/AP gains must be >=0.002 over pure M123 AND reconstruction-only before
outer validation. This internal gate does not replace the final +0.02 test requirement.
Attribution requires comparing bond messages against bag and chem controls, not only
against an unchanged base. The original validation/test data are not loaded in this round.

Initial checks: zero-output identity, common parameter initialization, nonzero drug/bond
gradients, atom/edge permutation invariance, isolated-atom handling, heldout-label
invariance, and nonzero-branch complete query-edge deletion. Audit repeats predictions,
checkpoint selection, exact float32 loss reconstruction and trained-model invariances.

Reproduce with ncRNAdrug Python, OMP/MKL/OPENBLAS_NUM_THREADS=2:
prepare.py -> check.py -> launch.py -> audit.py -> summarize.py.
Release implementation and both protected PLMF routes remain unchanged.
