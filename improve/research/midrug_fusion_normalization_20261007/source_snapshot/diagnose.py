"""Frozen fusion geometry and removal of scale inputs; no refitting or reselection."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,load_inner,seed_all,sha256,read_module
from model import CONFIGS,make_adapter,combine,decompose

source=HERE.parent/'midrug_representation_geometry_20261007/precheck.py'
spectrum=read_module('frozen_fusion_geometry_measurements',source).spectrum


def metric(y,p):return dict(auc=float(roc_auc_score(y,p)),ap=float(average_precision_score(y,p)))


def main():
    frozen=sha256(HERE/'selection.json');assert json.loads((HERE/'verification.json').read_text())['status']=='passed'
    records=[]
    for fold in range(3):
        data=load_inner(fold)
        for name in CONFIGS:
            path=HERE/'results'/f'{name}_f{fold}_s42.pt';meta=json.loads(path.with_suffix('.json').read_text());assert sha256(path)==meta['checkpoint_sha256']
            seed_all(42);a=make_adapter(data,name,'cuda:0');a.model.load_state_dict(torch.load(path,map_location='cuda:0',weights_only=True));m=a.model;m.eval()
            before={k:v.detach().clone() for k,v in m.state_dict().items()};row=dict(config=name,fold=fold)
            with torch.no_grad():
                parts=m.parts(a.pairs['train'],a.ps,a.qis['train'],a.ease['train'],a.dense)
                features=combine(parts,name)
                logits=m.lin3(F.leaky_relu(m.lin2(F.leaky_relu(m.lin1(features),.1)),.1)).view(-1)
                prob=logits.sigmoid().cpu().numpy()
            row['train']=metric(data['train_y'],prob);row['val']=dict(auc=meta['inner_auc'],ap=meta['inner_ap'])
            row['geometry']={}
            for i,h in enumerate(parts[:-2]):
                u,t=decompose(h)
                row['geometry'][str(i)]=dict(raw=spectrum(h),direction=spectrum(u),mean_stats=t.mean(0).cpu().tolist(),
                    std_stats=t.std(0).cpu().tolist(),raw_mean_row_norm=float(h.norm(dim=1).mean()))
            x=features.detach().requires_grad_();s=m.lin3(F.leaky_relu(m.lin2(F.leaky_relu(m.lin1(x),.1)),.1)).view(-1)
            grad=torch.autograd.grad(F.binary_cross_entropy_with_logits(s,a.y),x)[0]
            row['scale_input_gradient_l1_fraction']=float(grad[:,-m.extra_width:].abs().sum()/grad.abs().sum())
            row['scale_weight_norm_over_original_columns']=float(m.lin1.weight[:,-m.extra_width:].norm()/m.lin1.weight[:,:-m.extra_width].norm())
            if name=='direction_scale':
                with torch.no_grad():original=a.predict('val')
                try:
                    m.fusion_mode='direction'
                    with torch.no_grad():removed=a.predict('val')
                finally:m.fusion_mode=name
                row['without_scale_inputs']=dict(**metric(data['val_y'],removed),mean_abs_probability_change=float(abs(original-removed).mean()))
            for k,v in before.items():torch.testing.assert_close(v,m.state_dict()[k],atol=0,rtol=0)
            assert m.fusion_mode==name
            records.append(row);print('PASS',name,fold,flush=True);del a,m,before,parts,features,x,grad;torch.cuda.empty_cache()
    means=[]
    for name in CONFIGS:
        rows=[r for r in records if r['config']==name];item=dict(config=name)
        for part in ['train','val']:item[part]={m:float(np.mean([r[part][m] for r in rows])) for m in ['auc','ap']}
        for key in ['scale_input_gradient_l1_fraction','scale_weight_norm_over_original_columns']:
            item[key]=float(np.mean([r[key] for r in rows]))
        item['M1_geometry']={side:{kind:float(np.mean([r['geometry'][str(i)][kind]['participation_rank'] for r in rows])) for kind in ['raw','direction']} for i,side in enumerate(['miRNA','drug'])}
        if name=='direction_scale':item['without_scale_inputs']={m:float(np.mean([r['without_scale_inputs'][m] for r in rows])) for m in ['auc','ap','mean_abs_probability_change']}
        means.append(item)
    assert sha256(HERE/'selection.json')==frozen
    (HERE/'component_diagnostics.json').write_text(json.dumps(dict(status='passed',records=records,means=means,selection_sha256=frozen,
        diagnostic_source_sha256=sha256(source),selection_changed=False,no_parameter_or_buffer_updates=True,
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))
    table='\n'.join(f"| {r['config']} | {r['train']['auc']:.6f}/{r['train']['ap']:.6f} | {r['val']['auc']:.6f}/{r['val']['ap']:.6f} | {r['scale_input_gradient_l1_fraction']:.6f} | {r['scale_weight_norm_over_original_columns']:.6f} |" for r in means)
    candidate=next(r for r in means if r['config']=='direction_scale')['without_scale_inputs']
    (HERE/'COMPONENT_DIAGNOSTICS.md').write_text(f'''# 冻结选择后的融合诊断

| 配置 | 训练AUC/AP | 内部AUC/AP | 尺度输入梯度L1占比 | 尺度列/原列权重范数 |
|---|---:|---:|---:|---:|
{table}

在训练后的direction_scale模型中把十二个尺度输入置零，内部AUC/AP变为
{candidate['auc']:.9f}/{candidate['ap']:.9f}；平均概率变化
{candidate['mean_abs_probability_change']:.6g}。这是推理干预，不撤销联合训练，
不能替代direction对照或用于改选配置。

梯度是eval模式训练配对BCE对融合输入的局部导数，不是参数更新量。JSON还保存
每块原始/方向表示谱及尺度统计。所有参数、缓冲、模式和冻结选择均保持不变。
''')


if __name__=='__main__':main()
