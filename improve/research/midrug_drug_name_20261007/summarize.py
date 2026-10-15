"""Apply the frozen dual-metric gate and verify protected PLMF hashes."""
import csv, json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from common import HERE, RELEASE, sha256, load_inner
from model import CONFIGS

def main():
    assert not (HERE/'selection.json').exists()
    plan=json.loads((HERE/'results/plan.json').read_text())
    verification=json.loads((HERE/'verification.json').read_text())
    assert verification['status']=='passed'
    metrics={}; rows=[]
    for name in list(plan['controls'])+list(CONFIGS):
        aucs=[]; aps=[]; epochs=[]
        for fold in range(3):
            path=Path(plan['controls'][name][str(fold)]['path']) if name in plan['controls'] else HERE/'results'/f'{name}_f{fold}_s42.json'
            r=json.loads(path.read_text()); data=load_inner(fold)
            with np.load(path.with_name(path.stem+'_inner.npz')) as z:
                np.testing.assert_array_equal(z['indices'],data['val_indices'])
                np.testing.assert_array_equal(z['y'],data['val_y'])
                np.testing.assert_array_equal(z['pairs'],data['val_pairs'])
                auc=roc_auc_score(z['y'],z['probability']); ap=average_precision_score(z['y'],z['probability'])
                assert abs(auc-r['inner_auc'])<1e-12 and abs(ap-r['inner_ap'])<1e-12
            aucs.append(auc); aps.append(ap); epochs.append(r['best_epoch'])
        metrics[name]=dict(auc=aucs,ap=aps)
        rows.append(dict(config=name,mean_inner_auc=float(np.mean(aucs)),mean_inner_ap=float(np.mean(aps)),selected_epochs=epochs))
    comparisons={name:{metric:dict(fold_deltas=(np.asarray(metrics['aligned'][metric])-metrics[name][metric]).tolist(),
                                mean_delta=float(np.mean(np.asarray(metrics['aligned'][metric])-metrics[name][metric])))
                       for metric in ['auc','ap']} for name in plan['comparators']}
    minimum=min(x['mean_delta'] for c in comparisons.values() for x in c.values())
    promote=minimum>=plan['minimum_both_mean_gains']
    result=dict(eligible_candidate='aligned',metrics=metrics,comparisons=comparisons,
        minimum_mean_gain=minimum,promote_to_outer_validation=promote,plan_sha256=sha256(HERE/'results/plan.json'),
        verification_sha256=sha256(HERE/'verification.json'),no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'selection.json').write_text(json.dumps(result,indent=2))
    with (HERE/'inner_summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    original=HERE.parent/'midrug_internal_20261006/plmf_protection.json'
    protected=json.loads(original.read_text()); frozen=RELEASE/'data/frozen'
    manifest=json.loads((frozen/'manifest.json').read_text()); refs=HERE.parent/'baselines_m123_20261004/results'
    for p,h in protected['source_hashes'].items(): assert sha256(RELEASE/p)==h
    checked={str(RELEASE/p):h for p,h in protected['source_hashes'].items()}
    for r in protected['records']:
        ds,fold=r['dataset'],r['fold']
        checks={refs/f'incrd_m123_{ds}_f{fold}_s42.pt':r['checkpoint_sha256'],
                frozen/f'{ds}_fold{fold}.npz':r['split_sha256'],
                frozen/f'{ds}_graph.pt':manifest['datasets'][ds]['graph_sha256']}
        for p,h in checks.items(): assert sha256(p)==h;checked[str(p)]=h
    (HERE/'plmf_hash_recheck.json').write_text(json.dumps(dict(status='passed',checked_hashes=checked,
        prior_full_replay_sha256=sha256(original),new_gpu_replay=False),indent=2))
    table='\n'.join(f"| {r['config']} | {r['mean_inner_auc']:.9f} | {r['mean_inner_ap']:.9f} | {r['selected_epochs']} |" for r in rows)
    gains='\n'.join(f"| {n} | {c['auc']['mean_delta']:+.9f} | {c['ap']['mean_delta']:+.9f} |" for n,c in comparisons.items())
    deletion_error=max(x['errors']['full_logit'] for r in verification['records'] for x in r['explicit_deletions'])
    report=f'''# 原始药物名称的共享字符表示

原测试 AUC/AP 同时绝对 +0.02 尚未证实。本轮完成9次新训练及9个选中模型重放。
仅使用原Drug_Name文本中至少两个药物共享的3–5字符片段；重建任务仍以原938/768维归一化特征为目标。

| 配置 | 内部AUC | 内部AP | 选择轮次 |
|---|---:|---:|---|
{table}

唯一候选 aligned 相对预定对照的三折配对均值差：

| 对照 | ΔAUC | ΔAP |
|---|---:|---:|
{gains}

每项增益均须≥.002；最小值{minimum:+.9f}，推进门槛通过：{promote}。
这是一次原始药名信息覆盖核验，不是药理机制或药物类别的证明。
固定打乱协变量仅控制实体对应关系，不提供因果解释。

三组初始可学习参数一致，药物输入修正线性层零初始化；冻结的文本输入分别为正确、打乱、全零。
原节点特征、图边及重建目标保持一致；902维TF-IDF由原216个药名独立生成，不读取关联标签。
训练后checkpoint重放、双指标选轮重算及物理删边检查均通过；删边完整logit最大误差{deletion_error:.3g}。
PLMF检查仅为原checkpoint、输入和发布代码哈希复核，未记作新GPU重放。
没有新原始验证集或测试集评估；目标保持active。
'''
    (HERE/'REPORT.md').write_text(report)
    (HERE/'CONTINUATION.json').write_text(json.dumps(dict(goal_status='active',completion_proven=False,
        stage='completed_and_audited',new_fits=9,new_checkpoint_replays=9,promote_to_outer_validation=promote,
        selection_sha256=sha256(HERE/'selection.json'),processes_expected_live=[],no_lncrna=True,
        next_action='If the gate passes, freeze original-validation confirmation. Otherwise inspect fixed-model text contribution and novel miRNA-compatible evidence; do not retune ngrams, promote controls or infer pharmacological classes from spelling.'),indent=2))
    print(table);print(gains);print('PROMOTE',promote,flush=True)
if __name__=='__main__': main()
