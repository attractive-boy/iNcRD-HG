"""Register a bounded, adaptive inner calibration after the first pathway trial.

This reuses verified predictions; it is not an independent experiment or a new
training run. The earlier trial and its failed selection remain immutable.
"""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil

from improve import tabicl_pathway as experiment
from improve import tabular_foundation as base


PARENT = Path('improve/results/tabicl_pathway_20260913')
OUT = Path('improve/results/tabicl_pathway_calibration_20260913')
ALPHAS = (.001, .0025, .005, .01, .02, .03, .04, .075, .125, .15)


def main():
    source = PARENT.resolve()
    out = OUT.resolve()
    original = experiment.protocol(source)
    selection = base.check_seal(base.read_json(source/'SELECTION.json'))
    validation = base.read_json(source/'VALIDATION_inner.json')
    if original['smoke'] or selection['eligible'] or not validation['all_states_replayed']:
        raise ValueError('This follow-up requires the completed, unsuccessful original trial')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new calibration directory')
    p = deepcopy(original)
    p.pop('sha256')
    p['created_at_utc'] = datetime.now(timezone.utc).isoformat()
    p['recipes'] = [dict(id='baseline',variant='baseline',alpha=0.)] + [
        dict(id=(f'{variant}_a{alpha:g}').replace('.','p'),variant=variant,alpha=alpha)
        for variant in p['variants'] for alpha in ALPHAS]
    p['adaptive_calibration'] = dict(
        original_dir=str(source), original_protocol_sha256=original['sha256'],
        original_selection_sha256=base.digest(source/'SELECTION.json'),
        registration_script=str(Path(__file__).resolve()), registration_sha256=base.digest(__file__),
        new_alphas=list(ALPHAS),
        motivation='Original 10 settings improved MiDrug AUC/AP in all seeds but every LncDrug mean AUC declined. Examine smaller common weights around zero and near 0.1. This is adaptive development using already observed inner data.',
        constraints='One nonzero alpha and one variant apply to both tasks. Same original gate, no per-task identity branch, no new fitting or independent replication claim.',
        outer_policy='Only this trial\'s eligible, frozen inner selection can open this trial\'s outer data. The original trial stays closed; no outer reselection.')
    p = base.seal(p)
    out.mkdir(parents=True)
    (out/'source_snapshot').mkdir()
    for name in experiment.OWN:
        shutil.copy2(Path(experiment.__file__).with_name(name),out/'source_snapshot'/name)
    shutil.copy2(__file__,out/'source_snapshot'/Path(__file__).name)
    # Seal the new grid before calculating any new-mixture metric.
    base.write_json(out/'protocol.json',p)
    reused=[]
    for task in p['tasks']:
        for variant in p['variants']:
            for seed in p['seeds']:
                for kind in experiment.kinds(p,task):
                    src=experiment.directory(source,'inner',task,variant,seed,kind)
                    old=experiment.checked_result(src,original)
                    dest=experiment.directory(out,'inner',task,variant,seed,kind)
                    dest.mkdir(parents=True)
                    for name in old['artifacts']:
                        shutil.copy2(src/name,dest/name)
                        if base.digest(dest/name)!=old['artifacts'][name]:
                            raise ValueError('Prediction/state reuse changed bytes')
                    r=deepcopy(old)
                    r.update(protocol_sha256=p['sha256'],reused_fitted_state=True,
                             replay_source_protocol_sha256=original['sha256'],
                             replay_source_result_sha256=base.digest(src/'result.json'))
                    base.write_json(dest/'result.json',r)
                    reused.append(dict(source=str(src),destination=str(dest),
                                       source_result_sha256=base.digest(src/'result.json'),
                                       original_replay_max_abs=old['replay_max_abs']))
    shutil.copy2(source/'INPUT_COVERAGE.json',out/'INPUT_COVERAGE.json')
    shutil.copy2(source/'INNER_PAIR_COVERAGE.json',out/'INNER_PAIR_COVERAGE.json')
    base.write_json(out/'REUSE_AUDIT.json',dict(protocol_sha256=p['sha256'],
                    new_fitted_states=0,independent_replications=0,reused_states=len(reused),
                    unchanged_predictions=True,records=reused))
    print('CALIBRATION_REGISTERED',out,'new common weights',ALPHAS,'reused states',len(reused),flush=True)


if __name__=='__main__':
    main()
