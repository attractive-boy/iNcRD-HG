"""Read-only fixed-tree attribution and training loss diagnostic after failed selection."""
import json,hashlib
from pathlib import Path
import numpy as np
import xgboost as xgb
HERE=Path(__file__).resolve().parent
EXP=HERE.parent/'midrug_raw_residual_v2_20261007'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def bce(y,z):return float(np.mean(np.logaddexp(0,z.astype(float))-y*z))
def main():
 plan=json.loads((HERE/'plan.json').read_text())
 for p,h in plan['hashes'].items():assert sha(p)==h
 assert not json.loads((EXP/'selection.json').read_text())['promote_to_outer_validation']
 rows=[]
 for fold in range(3):
  with np.load(EXP/'inputs'/f'common_f{fold}.npz') as z:c={k:z[k] for k in z.files}
  for name in ['raw','pca32','structure']:
   with np.load(EXP/'inputs'/f'{name}_f{fold}.npz') as z:xs={k:z[k] for k in ['train','val']}
   for stage in ['selected','final']:
    tag=f'{name}_f{fold}_s42';path=EXP/'results'/f'{tag}_{"model" if stage=="selected" else "final_model"}.json'
    b=xgb.Booster();b.load_model(path);gains=b.get_score(importance_type='total_gain');counts=b.get_score(importance_type='weight')
    group_gains={k:0. for k in ['topology','miRNA','drug']};group_counts={k:0 for k in group_gains}
    for f,g in gains.items():
     i=int(f[1:]);group='topology' if i<8 else 'miRNA' if i<(946 if name=='raw' else 40) else 'drug'
     assert np.isfinite(g) and g>=0;group_gains[group]+=g;group_counts[group]+=int(counts[f])
    losses={};shifts={}
    for part in ['train','val']:
     dm=xgb.DMatrix(xs[part],base_margin=c[part+'_margin'],nthread=2);pred=b.predict(dm,output_margin=True);assert np.isfinite(pred).all()
     losses[part+'_base_bce']=bce(c[part+'_y'],c[part+'_margin']);losses[part+'_bce']=bce(c[part+'_y'],pred)
     shifts[part+'_mean_abs_logit_correction']=float(abs(pred-c[part+'_margin']).mean())
    total=sum(group_gains.values());rows.append(dict(config=name,fold=fold,stage=stage,rounds=b.num_boosted_rounds(),group_total_gain=group_gains,
      group_split_count=group_counts,group_gain_fraction={k:v/total if total else 0 for k,v in group_gains.items()},**losses,**shifts))
    print('PASS',name,fold,stage,flush=True)
 means=[]
 for name in ['raw','pca32','structure']:
  for stage in ['selected','final']:
   rs=[r for r in rows if r['config']==name and r['stage']==stage]
   means.append(dict(config=name,stage=stage,mean_rounds=float(np.mean([r['rounds'] for r in rs])),
    **{k:float(np.mean([r[k] for r in rs])) for k in ['train_base_bce','train_bce','val_base_bce','val_bce','train_mean_abs_logit_correction','val_mean_abs_logit_correction']},
    mean_gain_fraction={k:float(np.mean([r['group_gain_fraction'][k] for r in rs])) for k in ['topology','miRNA','drug']}))
 (HERE/'results.json').write_text(json.dumps(dict(status='passed',records=rows,means=means,no_new_training=True,no_auc_ap_reselection=True,selection_sha256=sha(EXP/'selection.json'),no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))
 table='\n'.join(f"| {r['config']} | {r['stage']} | {r['train_bce']:.6f} | {r['val_bce']:.6f} | {r['mean_gain_fraction']['topology']:.3f} | {r['mean_gain_fraction']['miRNA']:.3f} | {r['mean_gain_fraction']['drug']:.3f} |" for r in means)
 report=f'''# 固定树修正的特征使用和BCE诊断

冻结失败选择后，对9个选择模型和9个最终模型做只读诊断。没有新拟合、AUC/AP选型
或原验证/测试。原M123三折平均train/val BCE为{means[0]['train_base_bce']:.6f}/{means[0]['val_base_bce']:.6f}。

| 配置 | 状态 | 训练BCE | 内部留出BCE | 拓扑gain占比 | miRNA gain占比 | 药物gain占比 |
|---|---|---:|---:|---:|---:|---:|
{table}

gain是XGBoost训练目标的累计分裂改进，不是生物因果贡献，也不能跨相关特征/不同
维数当作公平重要性比较。raw拥有938维miRNA和768维药物，而PCA每侧32维。
本表只判断选择树是否实际使用相应输入；最终模型的损失只描述继续拟合的行为，
不用于替换已冻结的模型选择。源M123来自同一内部折，所有结果属于开发诊断。
'''
 (HERE/'REPORT.md').write_text(report)
 files=[p for p in HERE.iterdir() if p.is_file() and p.name not in ['archive_manifest.json','archive_recheck.json','run.log']]
 (HERE/'archive_manifest.json').write_text(json.dumps(dict(artifact_hashes={p.name:sha(p) for p in files},frozen_dependencies=plan['hashes']),indent=2));print(json.dumps(means,indent=2))
if __name__=='__main__':main()
