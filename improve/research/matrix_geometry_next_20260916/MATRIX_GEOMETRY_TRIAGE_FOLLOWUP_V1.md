The SPD-Sheaf author repository has now been retrieved at commit
`2c21b53efe9ce1f078c305bd9a4f3ecda57a9833`; twelve selected files were verified
against their Git blob hashes. The official ICML 2026 poster page at
<https://icml.cc/virtual/2026/poster/64132> contains the exact title. This verifies
the conference listing, extending the earlier note that had only the authors'
claim. The repository tree has no license file. No full author network or training
has been executed.

The actual main model is `model/SPDsheaf.py`, rather than the schematic
`layers/sheaf.py`. Its coordinate constructor normalizes each centered position
and forms `u u^T + epsilon I`. Its frame's first column is that same direction;
the first isometry composes a shared trainable orthogonal matrix with the frame
transpose. In evaluation without dropout and for nondegenerate positions, the
canonicalized input is diagonal and orientation-independent. The source's
`norm + 1e-8` leaves a very small norm-dependent variation; it is not exactly
constant in floating-point arithmetic. The main model exponentiates the input
before the congruence, which preserves this conclusion. Training dropout can
alter it and is outside this limited check.

`audit_spd_sheaf_coordinates_v1.py` executes the two unchanged coordinate-method
AST bodies on thirteen synthetic nodes. The saved audit reports a diagonal-form
error of `6.66e-16`, with variation from the author's normalization epsilon of
`1.95e-8`. An independent SciPy calculation verifies orthogonal log-congruence
equivalence to `1.78e-15`. These are mathematical/implementation checks, not a
claim about the complete paper's performance.

The main source also clips and conditionally rescales the eigenvalues of its
log-domain Laplacian result. Those are additional nonlinear operations; the
entire implementation must not be described as only a linear coordinate change.
Nevertheless, the orthogonal restriction-map algebra alone is linear on
symmetric matrices in log coordinates. A useful transfer would need genuine
matrix-valued information and controls that separate those nonlinearities and
edge transformations. No matching physical coordinates or covariance
measurements exist for all entities here. No SPD-Sheaf candidate is promoted to
training from this source audit.

A second retrieved preprint, *Nested Inductive Bias Framework for SPD Manifold
Learning*, arXiv:2609.04466v1 (3 September 2026), has been read through its
framework and experimental methodology. It composes a matrix logarithm and
half-vectorization with hyperbolic/spherical mappings, and also proposes the
bounded map `v/(1+alpha*norm(v))`. Its inputs are observed temporal covariance
matrices in action/radar data. No author code link was found in the extracted
text. Applying the bounded map alone would be a feature transformation; adding
hyperbolic geometry to vectorized matrices overlaps the already investigated
geometry family. Neither supplies a justified new input modality for this
static graph. No implementation or performance claim is made for this preprint.

The follow-up search files retain one arXiv HTTP 429 response rather than hiding
it. Other hits concerning temporal covariance, causal estimators and minimum
spanning tree algorithms have not been treated as ready graph-prediction
mechanisms. This research branch has added no formal neural fit and no validated
innovation. Earlier triage files and all experimental failures remain intact.
