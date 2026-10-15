"""Post-selection sensitivity in the same training-only pair-feature geometry."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from common import HERE,load_inner,seed_all,sha256
from model import make_adapter,capture_features,decode,feature_delta,mask_generator

def diagnose(a,fold):
    a.model.eval();original={n:p.clone() for n,p in a.model.state_dict().items()}
    logits,h=capture_features(a);bce=F.binary_cross_entropy_with_logits(logits,a.y)
    g=torch.autograd.grad(bce,h)[0];directions=[];rng=mask_generator(a.y.device,89000+fold)
    for name in ['adv','random0','random1','random2','random3']:
        d,info=feature_delta(h,g,.05,'adv' if name=='adv' else 'random',rng)
        with torch.no_grad():loss=F.binary_cross_entropy_with_logits(decode(a.model,h.detach()+d),a.y)
        directions.append(dict(direction=name,bce=float(loss),excess=float(loss-bce.detach()),active_queries=int(info['active'].sum())))
    for n,p in a.model.state_dict().items():torch.testing.assert_close(p,original[n],atol=0,rtol=0)
    return dict(training_bce=float(bce),width=h.shape[1],normalized_rms=.05,directions=directions)

def main():
    plan=json.loads((HERE/'results/plan.json').read_text());selection=sha256(HERE/'selection.json')
    assert json.loads((HERE/'selection.json').read_text())['status']=='frozen_inner_selection'
    assert json.loads((HERE/'verification.json').read_text())['status']=='passed'
    for p,h in plan['source_hashes'].items():assert sha256(p)==h
    rows=[]
    for fold in range(3):
        data=load_inner(fold);paths={n:Path(c[str(fold)]['path']) for n,c in plan['controls'].items()}
        paths.update({n:HERE/'results'/f'{n}_f{fold}_s42.json' for n in plan['configs']})
        for name,p in paths.items():
            r=json.loads(p.read_text());assert sha256(p.with_suffix('.pt'))==r['checkpoint_sha256']
            seed_all(42);a=make_adapter(data,'control','cuda:0');state=torch.load(p.with_suffix('.pt'),map_location='cuda:0',weights_only=True)
            a.model.load_state_dict({k:v for k,v in state.items() if not k.startswith('reconstruction.')})
            rows.append(dict(config=name,fold=fold,checkpoint_sha256=r['checkpoint_sha256'],**diagnose(a,fold)))
    means=[]
    for name in paths:
        rs=[r for r in rows if r['config']==name]
        means.append(dict(config=name,training_bce=float(np.mean([r['training_bce'] for r in rs])),
            adversarial_excess=float(np.mean([r['directions'][0]['excess'] for r in rs])),
            random_excess=float(np.mean([v['excess'] for r in rs for v in r['directions'][1:]]))))
    assert sha256(HERE/'selection.json')==selection
    (HERE/'sensitivity_diagnostics.json').write_text(json.dumps(dict(status='passed',records=rows,means=means,selection_sha256=selection,selection_changed=False,
        scope='Original-training pairs only; eval mode; one linearized gradient ascent and four random pair-representation perturbations, normalized RMS .05. Not a sharpness/generalization bound.',no_lncrna=True),indent=2))
    table='\n'.join(f"| {r['config']} | {r['training_bce']:.9f} | {r['adversarial_excess']:+.9f} | {r['random_excess']:+.9f} |" for r in means)
    (HERE/'SENSITIVITY_DIAGNOSTICS.md').write_text(f'''# 冻结选择后的配对表示敏感性

所有模型使用同一原训练配对、eval模式、未扰动表示的训练标准差尺度，
标准化RMS半径.05；原M123关联BCE，重建解码器移除。模型权重和BN统计均不改变。
这里只测一次梯度方向和4个随机方向，不是精确非线性最大损失或泛化保证。

| 配置 | 训练BCE均值 | 表示梯度方向损失增加 | 4随机方向平均增加 |
|---|---:|---:|---:|
{table}

参数SAM的.05是全模型参数L2半径，与本表的表示标准化RMS .05不是同一个度量。
本表只是把已选检查点放在同一表示扰动下诊断；内部指标和最终测试验收线不变。
selection.json前后哈希不变，没有因诊断结果追加配置或访问原验证/测试标签。
''')
    print(json.dumps(means,indent=2))

if __name__=='__main__':main()
