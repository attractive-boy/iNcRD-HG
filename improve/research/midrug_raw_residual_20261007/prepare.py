"""Generate current-inner fold features and source margins without new model selection."""
import json
from pathlib import Path
import numpy as np
import torch
from common import HERE,CLEAN,load_inner,seed_all,sha256
from model import CONFIGS,make_adapter,pca_coordinates,features,predict_margin,source

def main():
 out=HERE/'inputs';out.mkdir();records=[]
 for fold in range(3):
  data=load_inner(fold);seed_all(42);a=make_adapter(data,'unmasked10','cuda:0',fold=fold);pc=pca_coordinates(a)
  margins={part:predict_margin(a,data[part+'_pairs']) for part in ['train','val']}
  with np.load(Path(source(fold)['path']).with_name(f'unmasked10_f{fold}_s42_inner.npz')) as z:
   np.testing.assert_array_equal(z['pairs'],data['val_pairs']);np.testing.assert_array_equal(z['y'],data['val_y'])
   np.testing.assert_allclose(1/(1+np.exp(-margins['val'])),z['probability'],atol=2e-5,rtol=2e-5)
  np.savez_compressed(out/f'common_f{fold}.npz',**{part+'_margin':margins[part] for part in margins},**{k:data[k] for k in ['train_pairs','train_y','val_pairs','val_y','train_indices','val_indices']},pc_rna=pc[0],pc_drug=pc[1])
  for name in CONFIGS:
   x={part:features(a,data[part+'_pairs'],name,pc) for part in margins}
   np.savez_compressed(out/f'{name}_f{fold}.npz',**x);records.append(dict(config=name,fold=fold,n_features=x['train'].shape[1],n_train=len(x['train']),n_val=len(x['val'])))
  print('PREPARED',fold,flush=True);del a;torch.cuda.empty_cache()
 manifest=dict(status='prepared',records=records,artifact_hashes={str(p):sha256(p) for p in out.iterdir()},graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
  source_checkpoints={str(Path(source(f)['path']).with_suffix('.pt')):source(f)['checkpoint_sha256'] for f in range(3)},no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
 (HERE/'input_manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
