"""Revalidate original provenance; reconstruct frozen last-layer states offline, with no labels."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import json,hashlib
from pathlib import Path
import torch
from transformers import RobertaConfig,AutoTokenizer
from transformers.models.roberta.modeling_roberta import RobertaLayer
import pandas as pd
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'midrug_chemberta_adaptation_20261006'
RAW=HERE.parents[3]/'incrd-orig/2.construct_network'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def main():
 manifest=json.loads((OLD/'input_manifest.json').read_text());weights=Path(manifest['model_directory'])
 checks={str(weights/n):h for n,h in manifest['weight_hashes'].items()}
 checks.update({str(RAW/n):h for n,h in manifest['original_input_hashes'].items()})
 checks.update({str(OLD/'token_cache.pt'):manifest['cache_sha256'],str(OLD/'original_last_layer.pt'):manifest['last_layer_sha256']})
 for p,h in checks.items():assert sha(p)==h,p
 tok=AutoTokenizer.from_pretrained(str(weights),local_files_only=True)
 cids=pd.read_csv(RAW/'Drug.csv')['CID'].astype(int).tolist()
 strings=pd.read_csv(RAW/'drug_smiles.csv').set_index('drug_cid').loc[cids,'smiles'].tolist()
 fresh=tok(strings,return_tensors='pt',padding=True,truncation=True,max_length=512)
 c=torch.load(OLD/'token_cache.pt',map_location='cpu',weights_only=True)
 for k in ['input_ids','attention_mask']:torch.testing.assert_close(fresh[k],c[k],atol=0,rtol=0)
 assert c['drug_cids'].tolist()==cids
 layer=RobertaLayer(RobertaConfig.from_pretrained(str(weights),local_files_only=True)).cuda().eval()
 layer.load_state_dict(torch.load(OLD/'original_last_layer.pt',map_location='cuda',weights_only=True))
 zs=[]
 with torch.no_grad():
  for s in range(0,216,8):
   x=c['last_input'][s:s+8].cuda();mask=c['attention_mask'][s:s+8].cuda()
   extended=(1-mask[:,None,None,:].float())*torch.finfo(torch.float32).min
   zs.append(layer(x,attention_mask=extended)[0].cpu())
 z=torch.cat(zs);m=c['attention_mask'].bool();avg=(z*m[:,:,None]).sum(1)/m.sum(1)[:,None]
 torch.testing.assert_close(avg,c['mean_masked'],atol=5e-6,rtol=5e-6)
 torch.testing.assert_close(z.mean(1),c['mean_padded'],atol=5e-6,rtol=5e-6)
 special=torch.tensor(tok.all_special_ids);keep=m & ~(c['input_ids'][:,:,None]==special).any(-1)
 assert (keep.sum(1)>0).all()
 # BOS/EOS contextual influence retained in representations; exclude them only from pooling.
 z=z*keep[:,:,None]
 target=HERE/'final_tokens.pt'
 assert not target.exists()
 torch.save(dict(tokens=z,mask=keep,input_ids=c['input_ids'],drug_cids=c['drug_cids']),target)
 report=dict(status='passed',source_hashes=checks,original_manifest_sha256=sha(OLD/'input_manifest.json'),
  final_tokens_sha256=sha(target),mean_masked_error=float(abs(avg-c['mean_masked']).max()),
  padded_mean_error=float(abs(torch.cat(zs).mean(1)-c['mean_padded']).max()),shape=list(z.shape),
  token_counts=keep.sum(1).tolist(),special_ids=tok.all_special_ids,
  truncated_drugs=manifest['truncated_drugs'],labels_used=False,new_weights_downloaded=False,lncrna_used=False)
 (HERE/'input_manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['source_hashes','token_counts']},indent=2))
if __name__=='__main__':main()
