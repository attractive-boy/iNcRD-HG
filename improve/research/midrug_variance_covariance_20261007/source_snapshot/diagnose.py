"""Frozen selected-checkpoint geometry; no new selection or parameter updates."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,load_inner,seed_all,sha256,read_module
from model import CONFIGS,make_adapter,regularizers

# spectrum() operates only on the supplied representation tensor; it does not
# execute the precheck's main() or load a different model/data entry point.
source=HERE.parent/'midrug_representation_geometry_20261007/precheck.py'
spectrum=read_module('frozen_geometry_measurements',source).spectrum


def main():
    frozen=sha256(HERE/'selection.json');assert json.loads((HERE/'verification.json').read_text())['status']=='passed'
    records=[]
    for fold in range(3):
        data=load_inner(fold)
        for name in CONFIGS:
            path=HERE/'results'/f'{name}_f{fold}_s42.pt';meta=json.loads(path.with_suffix('.json').read_text());assert sha256(path)==meta['checkpoint_sha256']
            seed_all(42);a=make_adapter(data,name,'cuda:0');a.model.load_state_dict(torch.load(path,map_location='cuda:0',weights_only=True));a.model.eval()
            before={k:v.detach().clone() for k,v in a.model.state_dict().items()};row=dict(config=name,fold=fold)
            cache={}
            def capture(module,args):cache['h']=args[0]
            hook=a.model.lin1.register_forward_pre_hook(capture)
            try:
                with torch.no_grad():logits=a.logits('train')
            finally:hook.remove()
            prob=logits.sigmoid().cpu().numpy();y=data['train_y']
            row['train']=dict(auc=float(roc_auc_score(y,prob)),ap=float(average_precision_score(y,prob)),
                bce=float(F.binary_cross_entropy_with_logits(logits,a.y)))
            row['val']=dict(auc=meta['inner_auc'],ap=meta['inner_ap'])
            width=a.cfg.hidden;row['geometry']={}
            for endpoint,h in [('miRNA',cache['h'][:,:width]),('drug',cache['h'][:,width:2*width])]:
                normalized=F.layer_norm(h,(width,),eps=1e-5)
                v,c=regularizers(h.double())
                row['geometry'][endpoint]=dict(raw=spectrum(h),normalized=spectrum(normalized),variance_loss=float(v),covariance_loss=float(c),
                    mean_row_norm=float(h.norm(dim=1).mean()),mean_normalized_column_std=float(normalized.std(0).mean()))
            for k,v in before.items():torch.testing.assert_close(v,a.model.state_dict()[k],atol=0,rtol=0)
            assert not a.model.lin1._forward_pre_hooks
            records.append(row);print('PASS',name,fold,flush=True);del a,before,cache;torch.cuda.empty_cache()
    means=[]
    for name in CONFIGS:
        rows=[r for r in records if r['config']==name];item=dict(config=name)
        for part in ['train','val']:item[part]={m:float(np.mean([r[part][m] for r in rows])) for m in ['auc','ap']}
        item['geometry']={}
        for endpoint in ['miRNA','drug']:
            item['geometry'][endpoint]={k:float(np.mean([r['geometry'][endpoint][k] for r in rows])) for k in ['variance_loss','covariance_loss','mean_row_norm','mean_normalized_column_std']}
            for k in ['raw','normalized']:
                item['geometry'][endpoint][k]={m:float(np.mean([r['geometry'][endpoint][k][m] for r in rows])) for m in ['participation_rank','entropy_rank','leading_variance_fraction']}
        means.append(item)
    assert sha256(HERE/'selection.json')==frozen
    (HERE/'component_diagnostics.json').write_text(json.dumps(dict(status='passed',records=records,means=means,selection_sha256=frozen,
        diagnostic_source_sha256=sha256(source),selection_changed=False,no_parameter_or_buffer_updates=True,
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))
    table='\n'.join(f"| {r['config']} | {r['train']['auc']:.6f}/{r['train']['ap']:.6f} | {r['val']['auc']:.6f}/{r['val']['ap']:.6f} | {r['geometry']['miRNA']['raw']['participation_rank']:.3f}/{r['geometry']['drug']['raw']['participation_rank']:.3f} | {r['geometry']['miRNA']['normalized']['participation_rank']:.3f}/{r['geometry']['drug']['normalized']['participation_rank']:.3f} |" for r in means)
    (HERE/'COMPONENT_DIAGNOSTICS.md').write_text(f'''# 选型冻结后的端点几何诊断

| 配置 | 训练AUC/AP | 内部AUC/AP | 原始M1 miRNA/drug参与率有效秩 | 正则归一化后参与率有效秩 |
|---|---:|---:|---:|---:|
{table}

全部几何值来自原内部训练查询；留出成绩复用已审计元数据。JSON另保存谱集中度、
熵有效秩、方差/协方差目标值、均值范数与标准差。低秩或有效秩提高均不自动解释
排序泛化。未按这些诊断重新选型、调权重或延长训练，全部参数/缓冲和冻结选择不变。
''')


if __name__=='__main__':main()
