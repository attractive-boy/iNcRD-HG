"""Post-selection diagnosis of dot-attention shrinkage; no fitting or heldout scoring."""
import json, sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'midrug_cell_tokens_20261007'
sys.path.insert(0, str(PREVIOUS))
from common import load_inner, seed_all, sha256, CLEAN
from model import make_adapter


def describe(a):
    m = a.model; c = m.cell_context
    m.eval()
    with torch.no_grad():
        xr, xd = a.dense[:2]
        k, v, q = c.encode(xr, xd)
        r, d = a.pairs['train'][:, :128]
        logits = torch.bmm(k[r], q[d, :, None]).squeeze(-1) / c.hidden**.5
        alpha = logits.softmax(-1)
        row = dict(query_norm=float(q.norm(dim=-1).mean()), key_norm=float(k.norm(dim=-1).mean()),
                   score_column_std=float(logits.std(-1).mean()),
                   effective_columns=float((1/alpha.square().sum(-1)).mean()),
                   value_column_std=float(v.std(1).mean()))
    m.zero_grad()
    F.binary_cross_entropy_with_logits(a.logits('train'), a.y).backward()
    row['query_bce_gradient_norm'] = sum(float(p.grad.square().sum()) for p in c.query.parameters())**.5
    row['query_l2_penalty_gradient_norm'] = 1e-4 * sum(float(p.square().sum()) for p in c.query.parameters())**.5
    return row


def main():
    records = []
    for fold in range(3):
        for name in ['drug_rec10', 'mean_rec10']:
            path = PREVIOUS / 'results' / f'{name}_f{fold}_s42.pt'
            seed_all(42); a = make_adapter(load_inner(fold), name, 'cuda:0')
            initial = describe(a)
            a.model.load_state_dict(torch.load(path, map_location='cuda:0', weights_only=True))
            trained = describe(a)
            records.append(dict(config=name,fold=fold,checkpoint_sha256=sha256(path),initial=initial,trained=trained))
            print('PASS', name, fold, flush=True)
            del a; torch.cuda.empty_cache()
    x = torch.load(CLEAN/'graph.pt', map_location='cpu')['miRNA'].x
    result = dict(status='passed',records=records,original_attribute_rows=list(x.shape),
                  distinct_columns=len(torch.unique(x.T,dim=0)),min_rna_column_std=float(x.std(1).min()),
                  no_fitting=True,no_heldout_scoring=True,
                  source_hashes={str(p):sha256(p) for p in [Path(__file__), PREVIOUS/'model.py', PREVIOUS/'archive_recheck.json', PREVIOUS/'selection.json', PREVIOUS/'component_diagnostics.json']},
                  interpretation='Small learned query norms accompany nearly uniform attention. This does not identify weight decay as the unique cause. Test fixed-radius query/key attention with the same optimizer and a matched mean-query arm.')
    (HERE/'shrinkage_precheck.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
