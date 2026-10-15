"""Fixed selected-model offset/residual alignment; no training or new AUC/AP selection."""
import json,hashlib
from pathlib import Path
import numpy as np
import xgboost as xgb
HERE=Path(__file__).resolve().parent
EXP=HERE.parent/'midrug_oof_tree_20261007'
FEATURES=HERE.parent/'midrug_raw_residual_v2_20261007'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def bce(y,z):return float(np.mean(np.logaddexp(0,z.astype(float))-y*z))
def correlation(a,b):return float(np.corrcoef(a,b)[0,1]) if a.std()>1e-12 and b.std()>1e-12 else None

def main():
 plan=json.loads((HERE/'plan.json').read_text())
 for p,h in plan['hashes'].items():assert sha(p)==h
 selection=json.loads((EXP/'selection.json').read_text());assert not selection['promote_to_outer_validation']
 configs=json.loads((EXP/'results/plan.json').read_text())['configs'];rows=[];align=[]
 for fold in range(3):
  with np.load(EXP/'inputs'/f'common_f{fold}.npz') as z:c={k:z[k] for k in z.files}
  corrections={}
  for name,cfg in configs.items():
   with np.load(FEATURES/'inputs'/f'{cfg["features"]}_f{fold}.npz') as z:xs={k:z[k] for k in ['train','val']}
   b=xgb.Booster();b.load_model(EXP/'results'/f'{name}_f{fold}_s42_model.json')
   delta={}
   for part in ['train','val']:
    dm=xgb.DMatrix(xs[part],base_margin=np.zeros(len(xs[part]),dtype=np.float32),nthread=2);delta[part]=b.predict(dm,output_margin=True);assert np.isfinite(delta[part]).all()
   corrections[name]=delta['train']
   with np.load(EXP/'results'/f'{name}_f{fold}_s42_inner.npz') as z:
    np.testing.assert_allclose(c['val_margin']+delta['val'],z['margin'],atol=5e-5,rtol=2e-5)
   row=dict(config=name,fold=fold,selected_rounds=b.num_boosted_rounds(),train_fitting_offset_bce=bce(c['train_y'],c[cfg['offset']]),
    train_fitting_bce=bce(c['train_y'],c[cfg['offset']]+delta['train']),train_inference_base_bce=bce(c['train_y'],c['in_sample']),
    train_inference_bce=bce(c['train_y'],c['in_sample']+delta['train']),inner_val_bce=bce(c['val_y'],c['val_margin']+delta['val']),
    train_delta_mean=float(delta['train'].mean()),val_delta_mean=float(delta['val'].mean()),train_delta_rms=float(np.sqrt(np.mean(delta['train']**2))),val_delta_rms=float(np.sqrt(np.mean(delta['val']**2))))
   rows.append(row);print('PASS',name,fold,flush=True)
  source_gap=c['oof']-c['in_sample'];delta_gap=corrections['oof_raw']-corrections['insample_raw']
  groups={}
  for label in [0,1]:
   ix=c['train_y']==label;groups[str(label)]=dict(mean_source_gap=float(source_gap[ix].mean()),mean_delta_gap=float(delta_gap[ix].mean()),correlation_with_negative_source_gap=correlation(delta_gap[ix],-source_gap[ix]))
  align.append(dict(fold=fold,correlation_with_negative_source_gap=correlation(delta_gap,-source_gap),source_gap_mean=float(source_gap.mean()),delta_gap_mean=float(delta_gap.mean()),by_class=groups))
 means=[dict(config=name,**{k:float(np.mean([r[k] for r in rows if r['config']==name])) for k in rows[0] if k not in ['config','fold']}) for name in configs]
 result=dict(status='passed',records=rows,means=means,source_residual_alignment=align,selection_sha256=sha(EXP/'selection.json'),no_new_training=True,no_auc_ap_reselection=True,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
 (HERE/'results.json').write_text(json.dumps(result,indent=2))
 table='\n'.join(f"| {r['config']} | {r['train_fitting_offset_bce']:.6f} | {r['train_fitting_bce']:.6f} | {r['train_inference_bce']:.6f} | {r['inner_val_bce']:.6f} | {r['train_delta_rms']:.6f} | {r['val_delta_rms']:.6f} |" for r in means)
 report=f'''# OOF树修正：源margin和残差的固定模型诊断

冻结失败选择后检查9个已选模型，没有新训练或AUC/AP重选，原验证/测试未用。
“拟合offset”分别是各臂实际训练用的OOF或训练内logit；“训练推理”统一使用完整
300轮基模型，不能把其成绩称作整个树模型的折外泛化。

| 配置 | 源offset训练BCE | 加修正后的拟合BCE | 训练推理BCE | 内部留出BCE | 训练delta RMS | 留出delta RMS |
|---|---:|---:|---:|---:|---:|---:|
{table}

完整300轮模型的平均训练BCE为{means[0]['train_inference_base_bce']:.6f}。
训练配对上(OOF树delta−训练内树delta)与−(OOF源logit−训练内源logit)的相关系数
逐折为{[r['correlation_with_negative_source_gap'] for r in align]}；另保存0/1标签组内相关。
这里只描述不同学习过程的对齐程度：两臂选择轮次可不同，相关性不是因果证据，
OOF的图密度和训练样本量也不同。没有据此改树、加融合系数或重新选择模型。

树修正以显式零base_margin独立提取，并与原保存的留出margin核对加性关系。
本诊断确认训练offset改变了拟合信号及树修正规模，但不能证明额外校准必然改善排序。
'''
 (HERE/'REPORT.md').write_text(report)
 files=[p for p in HERE.iterdir() if p.is_file() and p.name not in ['archive_manifest.json','archive_recheck.json','run.log']]
 (HERE/'archive_manifest.json').write_text(json.dumps(dict(artifact_hashes={p.name:sha(p) for p in files},frozen_dependencies=plan['hashes']),indent=2));print(json.dumps(means,indent=2))
if __name__=='__main__':main()
