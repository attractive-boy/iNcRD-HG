"""Offline audit of the original ChemBERTa SMILES encoding and padding pool."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel,AutoTokenizer

HERE=Path(__file__).resolve().parent
CLEAN=HERE.parent/'midrug_mirna_only_20261006'
RAW=HERE.parents[3]/'incrd-orig/2.construct_network'
WEIGHTS=HERE.parents[3]/'.external/models/hf_home/hub/models--seyonec--ChemBERTa-zinc-base-v1/snapshots/761d6a18cf99db371e0b43baf3e2d21b3e865a20'
def sha256(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    provenance=json.loads((CLEAN/'groups_provenance.json').read_text())['original_source_hashes']
    for n in ['Drug.csv','drug_smiles.csv','drug_features-chemBertA.csv']:assert sha256(RAW/n)==provenance[n]
    cids=pd.read_csv(RAW/'Drug.csv')['CID'].astype(int).tolist()
    source=pd.read_csv(RAW/'drug_smiles.csv').set_index('drug_cid');strings=source.loc[cids,'smiles'].tolist()
    original=pd.read_csv(RAW/'drug_features-chemBertA.csv',index_col=0)
    assert original.index.astype(int).tolist()==cids
    tokenizer=AutoTokenizer.from_pretrained(str(WEIGHTS),local_files_only=True)
    model=AutoModel.from_pretrained(str(WEIGHTS),local_files_only=True,add_pooling_layer=False).eval().cuda()
    tokens=tokenizer(strings,return_tensors='pt',padding=True,truncation=True,max_length=512)
    full_lengths=[len(tokenizer(s,add_special_tokens=True)['input_ids']) for s in strings]
    hidden=[];pooled=[];masked=[];cls=[];replay_errors=[]
    with torch.no_grad():
        for start in range(0,len(cids),8):
            batch={k:v[start:start+8].cuda() for k,v in tokens.items()}
            output=model(**batch,output_hidden_states=True)
            z=output.last_hidden_state;h=output.hidden_states[-2]
            extended=model.get_extended_attention_mask(batch['attention_mask'],batch['input_ids'].shape)
            replay=model.encoder.layer[-1](h,attention_mask=extended)[0]
            torch.testing.assert_close(z,replay,atol=1e-6,rtol=1e-6)
            replay_errors.append(float((z-replay).abs().max()))
            hidden.append(h.cpu());pooled.append(z.mean(1).cpu());cls.append(z[:,0].cpu())
            mask=batch['attention_mask'][:,:,None]
            masked.append(((z*mask).sum(1)/mask.sum(1)).cpu())
    pooled=torch.cat(pooled).numpy();masked=torch.cat(masked).numpy();cls=torch.cat(cls).numpy()
    raw=original.values.astype(np.float32)
    norm=lambda x:(x-x.min(1,keepdims=True))/(x.max(1,keepdims=True)-x.min(1,keepdims=True)+1e-8)
    discrepancy=dict(raw_max_abs=float(abs(raw-pooled).max()),raw_mean_abs=float(abs(raw-pooled).mean()),
        normalized_max_abs=float(abs(norm(raw)-norm(pooled)).max()),normalized_mean_abs=float(abs(norm(raw)-norm(pooled)).mean()),
        normalized_vs_masked_mean_abs=float(abs(norm(raw)-norm(masked)).mean()),
        row_cosine_mean=float(np.mean((raw*pooled).sum(1)/(np.linalg.norm(raw,axis=1)*np.linalg.norm(pooled,axis=1)))))
    torch.save(dict(input_ids=tokens['input_ids'],attention_mask=tokens['attention_mask'],last_input=torch.cat(hidden),
        mean_padded=torch.from_numpy(pooled),mean_masked=torch.from_numpy(masked),cls=torch.from_numpy(cls),drug_cids=torch.tensor(cids)),HERE/'token_cache.pt')
    torch.save(model.encoder.layer[-1].cpu().state_dict(),HERE/'original_last_layer.pt')
    report=dict(status='offline_prepared',model_id='seyonec/ChemBERTa-zinc-base-v1',revision=WEIGHTS.name,
        model_directory=str(WEIGHTS),weight_hashes={n:sha256(WEIGHTS/n) for n in ['config.json','pytorch_model.bin','vocab.json','merges.txt','tokenizer_config.json','special_tokens_map.json']},
        original_input_hashes={n:sha256(RAW/n) for n in ['Drug.csv','drug_smiles.csv','drug_features-chemBertA.csv']},
        cache_sha256=sha256(HERE/'token_cache.pt'),last_layer_sha256=sha256(HERE/'original_last_layer.pt'),source_sha256=sha256(__file__),
        n_drugs=len(cids),tensor_shape=list(torch.cat(hidden).shape),token_length_min=min(full_lengths),token_length_max=max(full_lengths),
        truncated_drugs=[dict(cid=cids[i],tokens=n) for i,n in enumerate(full_lengths) if n>512],
        original_feature_discrepancy=discrepancy,last_layer_replay_max_error=max(replay_errors),
        external_data_used=False,new_weights_downloaded=False,lncrna_used=False,labels_used=False,
        torch_version=torch.__version__,pooling_note='Original code averages every padded position. Masked mean here includes real special tokens and excludes padding only.')
    (HERE/'input_manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
