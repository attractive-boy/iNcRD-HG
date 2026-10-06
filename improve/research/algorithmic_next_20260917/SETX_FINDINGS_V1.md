# Study-set cross-attention encoder (novel architecture): results

Date 2026-09-18. `algo_setx_v1.py` / `cells_setx` (bank-only), `algo_setx_v2.py` /
`cells_setx2` (bank + provenance block). Same 7-split gate and controls as the rest.

## The architecture

Instead of collapsing an entity's curated studies into scalar counts (provenance
channel) or treating studies as one more feature column, each curated record is a token
`MiniLM(text_ctx 384) ++ [year, throughput, effect_resistant, effect_sensitive]`; the
RNA's study set and the drug's study set are encoded by permutation-invariant
Transformer encoders (masked mean-pool → `h_r`, `h_d`), then a **cross-attention** head
lets `h_r` attend over the drug's study tokens and `h_d` over the RNA's (`c_r`, `c_d`);
`score = MLP([h_r, h_d, c_r, c_d, bank])`. Training rows use fold-specific study sets
(fold-positive removal), query rows use all training pairs. Control: permute the whole
record-feature matrix across records (entity↔text map destroyed, marginals kept).

## Result 1 — novel architecture on the accepted bank (`setx`)

| surface | MiDrug ΔAUC/ΔAP | LncDrug ΔAUC/ΔAP |
|---|---:|---:|
| inner | +0.00057 / +0.00105 | +0.00019 / +0.00031 |
| resplits | +0.00061 / +0.00127 | +0.00084 / +0.00238 |
| extra | +0.00138 / +0.00234 | +0.00063 / +0.00082 |

**Tolerance pass at w=0.05 and w=0.1** (strict fails only on one LncDrug AP cell,
−0.00032). Control `setx_perm` fails (MiDrug inner +0.00024/+0.00058, LncDrug extra
negative). So the set/cross-attention encoder is a real, novel, *working* member — but
its gain is an order of magnitude below the provenance-count channel.

## Result 2 — the encoder does NOT add beyond the counts (`setx2`)

Feeding the same encoder the provenance block as well:

| recipe | MiDrug inner ΔAUC/ΔAP |
|---|---:|
| `setx` (real records) | +0.00174 / +0.00478 |
| `setx_perm` (records permuted) | **+0.00192 / +0.00511** |

The permuted-record control is **equal or better** than the real one once the
provenance columns are present. The cross-attention contribution collapses to zero: all
the signal is in the provenance columns, exactly as the count-based channel already
captured. A richer set encoder, and by extension a hyper-relational GNN over the same
studies, is not expected to add on top of the counts.

## Reading

* Architectural novelty is real but the metric cannot see it once the provenance
  information is present: the study hypergraph carries co-membership that scalar counts
  already extract, and there is no residual structure for attention/message passing to
  model. This mirrors the 87-mechanism saturation on the association graph.
* A novel architecture that *does* pass is `setx` on the bank alone (tolerance gate,
  failing control), but its gain is ~5e-4.

## Artifacts

`algo_setx_v1.py`, `algorithmic_next_20260917/cells_setx/`; `algo_setx_v2.py`,
`cells_setx2/`; gate tables `/tmp/opencode/eval_setx_gates.csv`.
