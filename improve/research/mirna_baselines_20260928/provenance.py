"""Record and preserve the exact local source files and frozen model inputs."""
import hashlib,json,platform,shutil
from datetime import datetime,timezone
from pathlib import Path
import numpy,pandas,scipy,sklearn,torch,torch_geometric,rdkit,openpyxl
from bench_data import HERE,BASE,ROOT

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    folders={
      'baseline_repro_20260918':['repro_gcmdr_ncdr.py','repro_ammgc_ncdr.py','repos/JMSSMMA/model.py'],
      'msffmdi_baselines_20260920':['msffmdi_models.py'],
      'dlstmda_baselines_20260919':['common.py','models.py'],
      'plmf_strong_baselines_20260922':['models_extra.py','repos/SubKNet/code/model.py','repos/SubKNet/code/subgraph_extraction.py','repos/SubKNet/code/parameters.py'],
      'plmf_baselines_20260920':['gcnnma_plmf.py','plmf_data.py'],
      'new5_repro_20260919':['repos/PLMF-MDA-main/code/layers.py','repos/PLMF-MDA-main/code/data_preprocess.py'],
      'mphgnn_plmf_20260921':['build_drug_chemberta.py'],
    }
    sources=list(HERE.glob('*.py'))+[HERE/'registry.json']
    sources+=list((HERE/'original_source').glob('*'))
    for folder,files in folders.items():sources.extend(BASE/folder/f for f in files)
    records=[]
    for source in sources:
        rel=source.relative_to(ROOT);dest=HERE/'source_snapshot'/rel
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        records.append({'path':str(source),'sha256':digest(source),'snapshot':str(dest.relative_to(HERE))})
    weights=[BASE/'new5_repro_20260919/models/rnafm/RNA-FM_pretrained.pth']
    weights+=list((BASE/'new5_repro_20260919/repos/PLMF-MDA-main/code/DeepChem/ChemBERTa-77M-MLM').glob('*'))
    weights+=list((HERE/'data').glob('*_plmf_cache.pt'))
    weights+=list((HERE/'data').glob('*_content.npz'))
    weights+=list((HERE/'data').glob('*_entities.json'))
    weights+=list((HERE/'data').glob('*_incrd_original_inputs.pt'))
    weights+=[HERE/'data/incrd_original_manifest.json',ROOT/'incrd-run/3.heter_data/MiDrug_train_data.pth',
              BASE/'mphgnn_plmf_20260921/cache/drug_chemberta_zinc.npz']
    inputs=[{'path':str(p),'size':p.stat().st_size,'sha256':digest(p)} for p in weights if p.is_file()]
    packages={m.__name__:m.__version__ for m in (numpy,pandas,scipy,sklearn,torch,torch_geometric,rdkit,openpyxl)}
    out={'created_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),
         'packages':packages,'jmssmma_vendor':{'torch-sparse':'0.6.17+pt113cu117','torch-scatter':'2.1.1+pt113cu117'},
         'sources':records,'frozen_model_inputs':inputs}
    (HERE/'source_manifest.json').write_text(json.dumps(out,indent=2))
    print('Saved source snapshots:',len(records),'frozen model input hashes:',len(inputs))

if __name__=='__main__':main()
