"""Nine separately logged block-order fits, two devices, no source mutation."""
import json,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from common import HERE,sha256
def main():
    plan=json.loads((HERE/'results/plan.json').read_text())
    for f,h in {**plan['source_hashes'],**plan['evidence_hashes']}.items():assert sha256(f)==h
    jobs=[(name,fold) for fold in range(3) for name in plan['configs']]
    def worker(index):
        for name,fold in jobs[index::2]:
            tag=f'{name}_f{fold}_s42';assert not (HERE/'results'/f'{tag}.json').exists()
            with (HERE/'results'/f'{tag}.log').open('x') as f:
                subprocess.run([sys.executable,str(HERE/'train.py'),'--config',name,'--fold',str(fold),'--device',f'cuda:{index}'],stdout=f,stderr=subprocess.STDOUT,check=True)
            print('DONE',tag,flush=True)
    with ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(worker,i) for i in range(2)]
        for f in futures:f.result()
if __name__=='__main__':main()
