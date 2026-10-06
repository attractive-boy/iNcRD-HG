"""Direction D, step 1: sentence-embed the experimental-context text of every curated record.

Three text views per record (ncRNADrug DR_Curated columns):
    ctx   : "<Phenotype>; <Condition>"                 (cancer type + cell line / tissue)
    full  : "<Phenotype>; <Condition>; <Support>"      (adds the curated evidence sentence)
    title : "<Reference>"                              (paper title)
Model: sentence-transformers/all-MiniLM-L6-v2 (frozen, local mirror copy).  Output: one
float16 matrix per view aligned to research/exposure_next_20260917/records.csv rows.
Run in the ncRNAdrug env (sentence-transformers 3.2.1).
"""
import sys, re
from pathlib import Path
import numpy as np, pandas as pd
import torch
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
RECORDS = HERE.parent / 'exposure_next_20260917/records.csv'
MODEL = '/data1/licheng/incrd-hg/.external/models/minilm/all-MiniLM-L6-v2'
RAW = '/data1/licheng/incrd-hg/incrd-run/10.circ/raw/DR_Curated.txt'


def clean(s):
    s = str(s) if pd.notna(s) else ''
    return re.sub(r'\s+', ' ', s.replace(' ', ' ')).strip()


def main():
    rec = pd.read_csv(RECORDS)
    raw = pd.read_csv(RAW, sep='\t', encoding='latin-1', low_memory=False)
    raw['cid'] = pd.to_numeric(raw.CID, errors='coerce')
    # records.csv was built per task in DR_Curated order after filtering; rebuild the same filtered frames to fetch text
    frames = []
    from build_records_text import filtered_frames  # noqa
    for task, sub in filtered_frames(raw):
        frames.append(sub.assign(task=task))
    full = pd.concat(frames).reset_index(drop=True)
    assert len(full) == len(rec) and (full.task.values == rec.task.values).all() and (full.ri.values == rec.ri.values).all() and (full.di.values == rec.di.values).all()
    views = {
        'ctx': [clean(p) + '; ' + clean(c) for p, c in zip(full.Phenotype, full.Condition)],
        'full': [clean(p) + '; ' + clean(c) + '; ' + clean(s) for p, c, s in zip(full.Phenotype, full.Condition, full.Support)],
        'title': [clean(t) for t in full.Reference],
    }
    model = SentenceTransformer(MODEL, device='cuda:0')
    for name, texts in views.items():
        uniq = sorted(set(texts)); idx = {t: i for i, t in enumerate(uniq)}
        emb = model.encode(uniq, batch_size=256, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        mat = emb[[idx[t] for t in texts]].astype(np.float16)
        np.save(HERE / ('text_%s.npy' % name), mat)
        print(name, 'unique texts', len(uniq), 'matrix', mat.shape, flush=True)
    pd.DataFrame({'task': full.task, 'ri': full.ri, 'di': full.di, 'ctx': views['ctx'], 'title': views['title']}).to_csv(HERE / 'text_index.csv', index=False)


if __name__ == '__main__':
    main()
