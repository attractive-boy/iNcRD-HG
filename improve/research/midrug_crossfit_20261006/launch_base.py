"""Freeze and execute 15 nested base fits plus three full-outer-training controls."""
import sys,json,subprocess
from queue import Queue,Empty
from concurrent.futures import ThreadPoolExecutor
from common import HERE,CLEAN,PREVIOUS,sha256,nested_data

def main():
    out=HERE/'bases';out.mkdir(exist_ok=True)
    plan=dict(outer_folds=[0,1,2],nested_folds=5,nested_seed='20261006 + outer fold',seed=42,epochs=300,
        base_selection='fixed final epoch, no held-out-label selection',no_lncrna=True,
        source_hashes={str(p):sha256(p) for p in [HERE/'common.py',HERE/'base.py',PREVIOUS/'common.py',PREVIOUS/'model.py',PREVIOUS/'extract.py']},
        data_sha256=sha256(CLEAN/'inner_data.npz'),graph_sha256=sha256(CLEAN/'graph.pt'),
        outer_validation_evaluated=False,test_evaluated=False)
    path=out/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:path.write_text(json.dumps(plan,indent=2))
    q=Queue()
    for f in range(3):
        for k in [-1,0,1,2,3,4]:
            data,_=nested_data(f,k)
            assert not set(data['train_indices']).intersection(data['val_indices'])
            if not (out/f'base_f{f}_k{k}.json').exists():q.put((f,k))
    def worker(device):
        while True:
            try:f,k=q.get_nowait()
            except Empty:return
            with (out/f'base_f{f}_k{k}.log').open('w') as log:
                subprocess.run([sys.executable,str(HERE/'base.py'),'--fold',str(f),'--nested',str(k),'--device',device],stdout=log,stderr=subprocess.STDOUT,check=True)
            print('DONE',f,k,device,flush=True)
    with ThreadPoolExecutor(4) as pool:
        for f in [pool.submit(worker,d) for d in ['cuda:0','cuda:1','cuda:0','cuda:1']]:f.result()

if __name__=='__main__':main()
