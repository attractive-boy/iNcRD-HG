"""Freeze six gate configurations and their matching uniform encoders."""
import sys,json,subprocess
from queue import Queue,Empty
from concurrent.futures import ThreadPoolExecutor
from common import HERE,CLEAN,RECON,DEGREE,RELEASE,STRICT,sha256
from model import CONFIGS

MATCHED={'target':'control','all':'control','target_rec10':'unmasked10','all_rec10':'unmasked10',
         'all05':'degree_all05','all05_rec10':'degree_all05_rec10'}


def main():
    checks=json.loads((HERE/'initial_checks.json').read_text());assert checks['status']=='passed'
    assert checks['full_batch_peak_cuda_gib']<18 and 'alpha05_uniform_reference_error' in checks
    out=HERE/'results';out.mkdir(exist_ok=True);controls={}
    for name,root,prefix in [('control',CLEAN,'control'),('unmasked10',RECON/'phase2','unmasked10'),
                             ('degree_all05',DEGREE,'all05'),('degree_all05_rec10',DEGREE,'all05_rec10')]:
        controls[name]={str(f):dict(path=str(root/'results'/f'{prefix}_f{f}_s42.json'),
            checkpoint_sha256=sha256(root/'results'/f'{prefix}_f{f}_s42.pt')) for f in range(3)}
    sources=[HERE/f for f in ['common.py','events.py','model.py','train.py','launch.py']]
    sources += [STRICT,RECON/'model.py',DEGREE/'model.py',*(RELEASE/'incrd_hg_m123').glob('*.py')]
    plan=dict(configs=CONFIGS,folds=[0,1,2],seed=42,epochs=800,minimum_epochs=300,patience=200,controls=controls,
        matched_uniform_controls=MATCHED,source_hashes={str(p):sha256(p) for p in sources},
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        optimizer='RMSprop lr=1e-4 alpha=.99; attention and reconstruction parameter wd=1e-4, original base wd=0',
        gate='width16 fixed layer-normalized source/receiver projections, leakyrelu .2, gate .1+.8*sigmoid(score); final score weights zero initialized',
        checkpoint_selection='mean(inner AUC, inner AP)',
        config_selection='minimum mean AUC/AP gains over pure M123, unmasked10, and matching uniform encoder',
        promotion_minimum_both_gains_over_each_control=.002,
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    path=out/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan
    else:path.write_text(json.dumps(plan,indent=2))
    q=Queue()
    for fold in range(3):
        for name in CONFIGS:
            if not (out/f'{name}_f{fold}_s42.json').exists():q.put((name,fold))
    def worker(device):
        while True:
            try:name,fold=q.get_nowait()
            except Empty:return
            with (out/f'{name}_f{fold}_s42.log').open('w') as log:
                subprocess.run([sys.executable,str(HERE/'train.py'),'--config',name,'--fold',str(fold),'--device',device],
                               stdout=log,stderr=subprocess.STDOUT,check=True)
            print('DONE',name,fold,device,flush=True)
    with ThreadPoolExecutor(4) as pool:
        for future in [pool.submit(worker,d) for d in ['cuda:0','cuda:1','cuda:0','cuda:1']]:future.result()


if __name__=='__main__':main()
