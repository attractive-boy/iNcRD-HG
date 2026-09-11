"""Fixed topology diagnostics on the existing training-internal split only."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from improve import pair_exchange as parent
from improve import support_operator_pilot as previous
from improve.first_round_features import sha256


def main():
    reference = Path('improve/results/support_operator_20260910')
    p = previous.load_protocol(reference)
    output = Path(__file__).with_name('structural_audit.json')
    if output.exists():
        raise ValueError('Preserve existing diagnostics')
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Untrained structural diagnostics, not model performance or a headroom bound',
        source_sha256=sha256(Path(__file__)), reference_protocol_sha256=p['sha256'],
        no_outer_evaluation=True, no_test_evaluation=True, tasks={})
    for task in p['tasks']:
        arrays, graph = parent.load_stage(reference, p, task, 'inner')
        rna = p['training'][task]['rna']
        nr, nd = graph[rna].num_nodes, graph['drug'].num_nodes
        pair, y, queries, truth = (arrays[n] for n in ('pairs_train', 'y_train', 'pairs_val', 'y_val'))
        a = np.zeros((nr, nd), dtype=np.float64)
        a[pair[0, y == 1], pair[1, y == 1]] = 1
        assert not a[queries[0], queries[1]].any()
        blocks = []
        for nt, relation in (('lncRNA', 'LncDrug'), ('miRNA', 'MiDrug')):
            b = np.zeros((graph[nt].num_nodes, nd))
            ei = graph[nt, relation, 'drug'].edge_index.numpy()
            b[ei[0], ei[1]] = 1
            blocks.append(b)
        combined = np.concatenate(blocks)
        path3 = a @ (a.T @ a)
        cross_path3 = a @ (combined.T @ combined)
        row, col = a.sum(1), a.sum(0)
        singular = np.linalg.svd(a, compute_uv=False)
        energy = np.cumsum(singular ** 2) / np.sum(singular ** 2)
        diagnostics = {}
        for name, matrix in (('same_task_path3', path3), ('cross_rna_path3', cross_path3)):
            values = matrix[queries[0], queries[1]]
            diagnostics[name] = dict(all_query_coverage=float(np.mean(values > 0)),
                positive_query_coverage=float(np.mean(values[truth == 1] > 0)),
                fixed_score_auc=float(roc_auc_score(truth, values)),
                fixed_score_ap=float(average_precision_score(truth, values)))
        result['tasks'][task] = dict(n_rna=nr, n_drug=nd, fitting_rows=len(y),
            fitting_positives=int(y.sum()), internal_rows=len(truth),
            positive_query_rna_without_fitting_positive_fraction=float(np.mean(row[queries[0, truth == 1]] == 0)),
            positive_query_drug_without_fitting_positive_fraction=float(np.mean(col[queries[1, truth == 1]] == 0)),
            singular_value_energy={str(k): float(energy[k-1]) for k in (1, 4, 8, 16, 32, 64)},
            diagnostics=diagnostics,
            input_sha256={n: sha256(reference / 'inputs' / task / n) for n in ('inner.npz', 'train_graph.pth')})
        print(task, json.dumps(result['tasks'][task], ensure_ascii=False), flush=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
