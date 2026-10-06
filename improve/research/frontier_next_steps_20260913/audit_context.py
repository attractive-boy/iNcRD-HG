"""Read-only audit of the existing direction/context records and source overlap.

This does not train models or read association-task test predictions. Source
overlap is a limitation of independence, not proof of label leakage or causality.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent


def old_split(frame, kind, seed):
    """Mirror context_controls.make_split's existing 60/20/20 protocol."""
    rng = np.random.default_rng(seed)
    if kind == 'random':
        order = rng.permutation(len(frame))
        a, b = int(.6 * len(frame)), int(.8 * len(frame))
        return frame.iloc[order[:a]], frame.iloc[order[a:b]], frame.iloc[order[b:]]
    column = {'cold_rna': 'ncRNA', 'cold_drug': 'cid'}[kind]
    unique = np.array(sorted(set(frame[column])))
    order = rng.permutation(len(unique))
    a, b = int(.6 * len(unique)), int(.8 * len(unique))
    folds = [set(unique[order[:a]]), set(unique[order[a:b]]), set(unique[order[b:]])]
    return [frame[frame[column].isin(f)] for f in folds]


def main():
    results = {}
    for task in ['MiDir', 'LncDir', 'CircDir']:
        rp = ROOT / 'incrd-run/11.direction' / (task + '_records.csv')
        cp = ROOT / 'incrd-run/13.context' / (task + '_context.csv')
        raw = pd.read_csv(rp).fillna('')
        ctx = pd.read_csv(cp).fillna('')
        keys = ['ncRNA', 'cid']
        grouped = raw.groupby(keys).effect.nunique()
        conflicts = set(grouped[grouped > 1].index)
        condition_counts = raw.groupby(keys + ['condition']).effect.nunique()
        condition_conflicts = {k[:2] for k, v in condition_counts.items() if v > 1}
        mapped = ctx[ctx.disease != ''].copy()
        disease_counts = mapped.groupby(keys + ['disease']).effect.nunique()
        disease_conflicts = {k[:2] for k, v in disease_counts.items() if v > 1}
        mapped_pairs = set(mapped[keys].itertuples(index=False, name=None)) & conflicts
        rows = {
            'raw_records': len(raw), 'unique_pairs': len(grouped),
            'conflicting_pairs': len(conflicts),
            'conflicting_pairs_without_conflict_after_raw_condition': len(conflicts - condition_conflicts),
            'raw_condition_note': 'No within-condition conflict is descriptive; it does not validate a biological reversal.',
            'context_rows': len(ctx), 'original_records_with_mapped_context': int(mapped.record.nunique()),
            'mapped_context_rows': len(mapped), 'mapped_disease_labels': int(mapped.disease.nunique()),
            'nonconflicting_mapped_triples': int((disease_counts == 1).sum()),
            'conflicting_mapped_triples': int((disease_counts > 1).sum()),
            'mapped_original_conflict_pairs': len(mapped_pairs),
            'mapped_original_conflict_pairs_without_within_disease_conflict': len(mapped_pairs - disease_conflicts),
            'pmids': int(raw.loc[raw.pmid != '', 'pmid'].nunique()),
            'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [rp, cp]},
        }
        if task == 'MiDir':
            merged = ctx.merge(raw.reset_index(names='record')[['record', 'pmid', 'throughput', 'year', 'condition']], on='record', how='left', validate='many_to_one')
            grouped_triples = mapped.groupby(keys + ['disease']).effect
            triples = grouped_triples.first()[grouped_triples.nunique() == 1].reset_index()
            for split in ['cold_rna', 'cold_drug', 'random']:
                train, _, test = old_split(triples, split, 1)
                trj = train.merge(merged, on=keys + ['disease'], how='left')
                tej = test.merge(merged, on=keys + ['disease'], how='left')
                train_pmids = set(trj.pmid.dropna()) - {''}
                test_pmids = set(tej.pmid.dropna()) - {''}
                rows[split + '_seed1_source_overlap'] = {
                    'train_pmids': len(train_pmids), 'test_pmids': len(test_pmids),
                    'overlap_pmids': len(train_pmids & test_pmids), 'test_triples': len(test),
                    'test_triples_with_training_pmid': len(tej[tej.pmid.isin(train_pmids)][keys + ['disease']].drop_duplicates()),
                }
        results[task] = rows
        print(task, json.dumps({k: v for k, v in rows.items() if k != 'input_sha256'}, ensure_ascii=False))
    (OUT / 'context_readonly_audit.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
