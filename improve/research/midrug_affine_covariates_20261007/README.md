# Original affine covariates, strictly miRNA and drug

Current pure-miRNA experiments use row-minmax miRNA expression and ChemBERTa coordinates. This representation removes the original row minimum and range. Adding these two covariates per node preserves all the original float32 information to numerical precision. This is a coverage check of an omitted input, not a claim of a new biological mechanism or a new normalization algorithm.

The older expression study used a mixed RNA graph and tested four expression statistics, but did not establish a pure-miRNA result or restore original ChemBERTa row scale. Current global routing, conditioned expression and raw-feature tree studies consume the already row-minmax inputs. The old loaders are not used.

Three matched arms append two standardized covariates to each node: correctly aligned minimum/range; a fixed within-type permutation of the same covariates; or zeros. Standardization uses node covariates only, as allowed in this transductive protocol. All arms have identical trainable parameters and initialization. The graph, M1/M2/M3, optimizer and original normalized-coordinate reconstruction targets stay fixed. Only `aligned` is eligible for promotion. The shuffled arm is a correspondence control, not evidence about a biological cause.

Nine fits: seed42 on the three existing original-training folds, RMSprop1e-4, 800 max/300 min epochs, patience200, earliest maximum mean(inner AUC,AP). Reconstruction weight10 with the established 50-epoch ramp; decoder weight decay1e-4. Both mean inner metrics must improve by .002 over each of `permuted`, `zero`, existing plain M123 and existing unmasked10 before a new original-validation confirmation is considered. The original frozen-test requirement remains AUC>=.9733929274189016 and AP>=.933683373732677 over seeds42/43/44, with PLMF protected.

Original Drug_Name strings were inspected for prior feature coverage but are not model inputs here. Name substrings are not treated as validated pharmacological classes. Literature search results are a separate evidence review; no external data, model weights, lncRNA nodes, features, relations or labels are imported.

Initial label-flip checks require exact model states and exact input/channel equality. GPU logits allow 2e-5 absolute/relative error, consistent with existing neural replay checks; a first pre-freeze exact-float assertion observed a 3.8e-6 GPU difference. No frozen protocol or training results existed at that time.
