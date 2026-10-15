"""Minimal immutable original-training inputs; no mixed-task loader imports."""
import hashlib,json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
CLEAN=HERE.parent/'midrug_mirna_only_20261006'
GROUPS=HERE.parent/'midrug_class_prior_20261006'
RELEASE=HERE.parents[2]/'release/iNcRD-HG-M1M2M3'
SOURCE_PLAN=HERE.parent/'midrug_training_size_20261007/results/plan.json'
def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))
def load_inner(fold):
    m=json.loads((CLEAN/'manifest.json').read_text())
    assert sha256(CLEAN/'inner_data.npz')==m['data_sha256']
    with np.load(CLEAN/'inner_data.npz') as z:
        tr,va=z[f'train_{fold}'],z[f'val_{fold}'];pairs,y=z['pairs'],z['y']
    assert len(set(tr)&set(va))==0
    return dict(train_pairs=pairs[tr],train_y=y[tr],train_indices=tr,
                val_pairs=pairs[va],val_y=y[va],val_indices=va)
def groups():
    meta=json.loads((GROUPS/'groups.json').read_text())
    assert sha256(GROUPS/'groups.npz')==meta['groups_sha256']
    assert sha256(CLEAN/'graph.pt')==meta['graph_sha256']
    assert meta['no_labels'] and meta['no_lncrna'] and meta['no_external_data']
    return dict(np.load(GROUPS/'groups.npz'))
def controls():return json.loads(SOURCE_PLAN.read_text())['controls']
def control(fold,name,data):
    r=controls()[name][str(fold)];p=Path(r['path'])
    assert sha256(p)==r['metadata_sha256']
    assert sha256(p.with_suffix('.pt'))==r['checkpoint_sha256']
    f=p.with_name(p.stem+'_inner.npz');assert sha256(f)==r['probability_sha256']
    z=dict(np.load(f))
    for key in ['pairs','y','indices']:np.testing.assert_array_equal(z[key],data['val_'+key])
    return z['probability'].astype(np.float64)
def verify_plan():
    p=json.loads((HERE/'results/plan.json').read_text())
    for f,h in {**p['source_hashes'],**p['evidence_hashes']}.items():assert sha256(f)==h,f
    return p

