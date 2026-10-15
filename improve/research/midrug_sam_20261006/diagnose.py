"""Post-selection, training-only one-step weight sensitivity; not a test estimate."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from common import HERE,load_inner,seed_all,sha256
from model import make_adapter,generator

def sensitivity(a,fold,rho=.05):
    m=a.model;m.eval();m.zero_grad();loss=F.binary_cross_entropy_with_logits(a.logits('train'),a.y);loss.backward()
    params=[(n,p) for n,p in m.named_parameters() if p.grad is not None]
    old={n:p.detach().clone() for n,p in params};full={n:t.clone() for n,t in m.state_dict().items()}
    g=[p.grad.detach().clone() for n,p in params];gnorm=torch.stack([v.norm() for v in g]).norm()
    rng=generator(a.y.device,79000+fold);values=[]
    for direction in ['gradient','random0','random1','random2','random3']:
        vs=g if direction=='gradient' else [torch.randn(p.shape,device=p.device,dtype=p.dtype,generator=rng) for n,p in params]
        norm=torch.stack([v.norm() for v in vs]).norm()
        try:
            with torch.no_grad():
                for (n,p),v in zip(params,vs):p.copy_(old[n]+rho*v/(norm+1e-12))
                after=F.binary_cross_entropy_with_logits(a.logits('train'),a.y)
            values.append(dict(direction=direction,bce=float(after),excess=float(after-loss.detach())))
        finally:
            with torch.no_grad():
                for n,p in params:p.copy_(old[n])
    for n,t in m.state_dict().items():torch.testing.assert_close(t,full[n],atol=0,rtol=0)
    return dict(training_bce=float(loss),gradient_norm=float(gnorm),parameter_count=sum(p.numel() for n,p in params),rho=rho,
        weights_norm=float(torch.stack([p.detach().norm() for n,p in params]).norm()),directions=values)

def main():
    plan=json.loads((HERE/'results/plan.json').read_text());selection_path=HERE/'selection.json';selection_hash=sha256(selection_path)
    assert json.loads(selection_path.read_text())['status']=='frozen_inner_selection'
    assert json.loads((HERE/'verification.json').read_text())['status']=='passed'
    for p,h in plan['source_hashes'].items():assert sha256(p)==h
    rows=[]
    for fold in range(3):
        data=load_inner(fold)
        paths={n:Path(c[str(fold)]['path']) for n,c in plan['controls'].items()}
        paths.update({n:HERE/'results'/f'{n}_f{fold}_s42.json' for n in plan['configs']})
        for n,p in paths.items():
            seed_all(42);a=make_adapter(data,'control','cuda:0');r=json.loads(p.read_text())
            assert sha256(p.with_suffix('.pt'))==r['checkpoint_sha256']
            state=torch.load(p.with_suffix('.pt'),map_location='cuda:0',weights_only=True)
            a.model.load_state_dict({k:v for k,v in state.items() if not k.startswith('reconstruction.')})
            rows.append(dict(config=n,fold=fold,checkpoint_sha256=r['checkpoint_sha256'],**sensitivity(a,fold)))
    means=[]
    for n in paths:
        rs=[r for r in rows if r['config']==n]
        means.append(dict(config=n,training_bce=float(np.mean([r['training_bce'] for r in rs])),
            gradient_direction_excess=float(np.mean([r['directions'][0]['excess'] for r in rs])),
            random_direction_mean_excess=float(np.mean([x['excess'] for r in rs for x in r['directions'][1:]]))))
    assert sha256(selection_path)==selection_hash
    (HERE/'sensitivity_diagnostics.json').write_text(json.dumps(dict(status='passed',records=rows,means=means,
        selection_sha256=selection_hash,selection_changed=False,heldout_metrics_used=False,
        scope='One gradient step at fixed Euclidean radius and four random directions, original-training BCE in eval mode. Not exact worst-case sharpness, Hessian eigenvalue or generalization bound.'),indent=2))
    table='\n'.join(f"| {r['config']} | {r['training_bce']:.9f} | {r['gradient_direction_excess']:+.9f} | {r['random_direction_mean_excess']:+.9f} |" for r in means)
    (HERE/'SENSITIVITY_DIAGNOSTICS.md').write_text(f'''# 冻结选择后的训练损失敏感性

只用当前折原训练配对/标签，以eval模式和相同M123推理参数计算关联BCE。
所有重建解码器移除，梯度方向半径固定.05，另取4个独立随机方向；均精确恢复权重。
selection.json在诊断前后哈希不变，结果不用于追加配置选择。

| 配置 | 训练BCE均值 | 梯度方向损失增加 | 4个随机方向平均损失增加 |
|---|---:|---:|---:|
{table}

这里是有限半径的一步方向敏感性，不是精确最大损失、Hessian特征值或泛化上限。
固定参数尺度会影响结果，不能仅凭较低敏感性宣称已获得更好的测试AUC/AP。
它与三折内部结果一起解释该训练机制，最终验收线与未访问原测试的状态不变。
''')
    archive_path=HERE/'archive_manifest.json';archive=json.loads(archive_path.read_text())
    for n in ['sensitivity_diagnostics.json','SENSITIVITY_DIAGNOSTICS.md']:archive['artifact_hashes'][n]=sha256(HERE/n)
    archive_path.write_text(json.dumps(archive,indent=2));print(json.dumps(means,indent=2))

if __name__=='__main__':main()
