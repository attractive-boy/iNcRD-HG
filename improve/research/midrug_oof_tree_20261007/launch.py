import sys,json,subprocess
from concurrent.futures import ThreadPoolExecutor
from queue import Queue,Empty
from common import HERE,sha256

def main():
 plan=json.loads((HERE/'results/plan.json').read_text());q=Queue()
 for p,h in plan['source_hashes'].items():assert sha256(p)==h
 for f in plan['folds']:
  for n in plan['configs']:q.put((n,f))
 def worker():
  while True:
   try:n,f=q.get_nowait()
   except Empty:return
   assert not (HERE/'results'/f'{n}_f{f}_s42.json').exists()
   with (HERE/'results'/f'{n}_f{f}_s42.log').open('w') as log:
    subprocess.run([sys.executable,str(HERE/'train.py'),'--config',n,'--fold',str(f)],stdout=log,stderr=subprocess.STDOUT,check=True)
   print('DONE',n,f,flush=True)
 with ThreadPoolExecutor(3) as pool:
  for fut in [pool.submit(worker) for _ in range(3)]:fut.result()
if __name__=='__main__':main()
