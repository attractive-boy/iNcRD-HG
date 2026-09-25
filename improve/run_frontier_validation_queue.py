"""Queue registered GPU jobs after the corresponding site validation jobs."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

from improve import tabular_foundation as base
from improve import tabicl_pathway as operations


def run(args):
    site,drug = Path(args.site_dir).resolve(),Path(args.drug_dir).resolve()
    sp,dp = (base.check_seal(base.read_json(p/'protocol.json')) for p in (site,drug))
    if sp['smoke'] or dp['smoke'] or dp['family']!='drug':
        raise ValueError('Queue requires frozen formal protocols')
    previous = (['generic','flat','evidence','evidence_stats'] if args.gpu==0 else
                ['evidence_shuffled','target','target_stats','target_shuffled'])
    jobs = ([('MiDrug',['descriptors','fingerprint','chemberta'])] if args.gpu==0 else
            [('MiDrug',['m2umol','m2umol_shuffled']),('LncDrug',dp['variants'])])
    prerequisite = [operations.directory(site,'inner','MiDrug',v,s,k)
        for v in previous for s in sp['seeds'] for k in operations.kinds(sp,'MiDrug')]
    record = dict(queue_source_sha256=base.digest(__file__),site_protocol_sha256=sp['sha256'],
        drug_protocol_sha256=dp['sha256'],gpu=args.gpu,site_variants=previous,jobs=jobs,
        expected_prerequisite_jobs=len(prerequisite))
    base.write_json(drug/('QUEUE_GPU'+str(args.gpu)+'.json'),record)
    start = time.monotonic()
    while True:
        complete = sum((p/'result.json').exists() for p in prerequisite)
        if complete==len(prerequisite):
            for directory in prerequisite:
                operations.checked_result(directory,sp)
            break
        if time.monotonic()-start>7200:
            raise TimeoutError('Preceding site jobs did not complete within two hours')
        print('WAITING_FOR_SITE',args.gpu,complete,'/',len(prerequisite),flush=True)
        time.sleep(30)
    for task,variants in jobs:
        command = [sys.executable,'-m','improve.frozen_feature_validation','run',
            '--out-dir',str(drug),'--task',task,'--variant',','.join(variants),'--device','cuda:'+str(args.gpu)]
        print('STARTING_DRUG',task,variants,flush=True)
        subprocess.run(command,check=True)
    base.write_json(drug/('QUEUE_GPU'+str(args.gpu)+'_COMPLETE.json'),dict(record,seconds=time.monotonic()-start))
    print('QUEUE_COMPLETE',args.gpu,flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',type=int,choices=[0,1],required=True)
    parser.add_argument('--site-dir',default='improve/results/site_evidence_association_20260914')
    parser.add_argument('--drug-dir',default='improve/results/drug_multimodal_association_20260914')
    run(parser.parse_args())
