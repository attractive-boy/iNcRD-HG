"""Run the fixed nine fits on available GPUs."""
import sys,subprocess,json
from concurrent.futures import ThreadPoolExecutor
from queue import Queue,Empty
from common import HERE,sha256

def main():
 plan=json.loads((HERE/'results/plan.json').read_text())
 for p,h in plan['source_hashes'].items():assert sha256(p)==h
 q=Queue()
 for fold in plan['folds']:
  for n in plan['configs']:
   if not (HERE/'results'/f'{n}_f{fold}_s42.json').exists():q.put((n,fold))
 def worker(device):
  while True:
   try:n,fold=q.get_nowait()
   except Empty:return
   with (HERE/'results'/f'{n}_f{fold}_s42.log').open('w') as log:
    subprocess.run([sys.executable,str(HERE/'train.py'),'--config',n,'--fold',str(fold),'--device',device],stdout=log,stderr=subprocess.STDOUT,check=True)
   print('DONE',n,fold,device,flush=True)
 with ThreadPoolExecutor(4) as pool:
  for fut in [pool.submit(worker,d) for d in ['cuda:0','cuda:1','cuda:0','cuda:1']]:fut.result()
if __name__=='__main__':main()
