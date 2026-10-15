# Next hypothesis: learn drug features from the original molecular structures

The degree-normalized encoder did not surpass reconstruction-only in this three-fold
development comparison. Do not promote it, and do not repeat the same normalization grid.

Existing work tested original-SMILES Morgan2/Morgan3/MACCS/pharmacophore similarities
as alternative drug neighbor graphs. Original expression/ChemBERTa bilinear interactions
were also tested. The reviewed MiDrug experiment directories do not contain a learned
atom/bond molecular encoder feeding the association predictor.

Use only the already-permitted 216 original Drug.csv/SMILES records; molecular structure
is an internal representation of the existing drugs, not additional association entities.
The source alignment and hashes are already in the pure-miRNA groups provenance.
Preparation is in ../midrug_molecular_encoder_20261006/prepare.py. No pretrained weights
or outside drug, gene, disease, sequence or lncRNA observations are needed.

Before training, freeze a small controlled comparison: atom/bond message passing versus
an atom-bag control and an original-ChemBERTa projection control. Use a modest shared
capacity, preserve M1/M2/M3, and compare both branch-only information and the proposed
fusion against the stronger reconstruction-only control. Include exact query deletion
of the association path, atom permutation invariance and zero-initialized branch identity.
Final architecture and optimizer settings are not frozen yet. Preparation alone is not
performance evidence; fit and selection must remain within the original-training folds.

The final original-test requirement remains both AUC/AP +0.02. PLMF stays protected.
