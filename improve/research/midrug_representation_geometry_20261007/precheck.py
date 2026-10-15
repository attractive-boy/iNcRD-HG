"""Training-only representation spectra; no fits and no validation selection."""
import json
from pathlib import Path
import sys
import torch
import numpy as np

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'midrug_regime_ranking_20261007'
sys.path.insert(0, str(PRIOR))
from common import load_inner, seed_all, sha256, CLEAN, RELEASE
from model import make_adapter


def spectrum(x):
    x = x.detach().double()
    x = x - x.mean(0, keepdim=True)
    covariance = (x.T @ x) / (len(x) - 1)
    eig = torch.linalg.eigvalsh(covariance).clamp_min(0)
    energy = eig.sum()
    assert energy > 0 and torch.isfinite(eig).all()
    frac = eig / energy
    sd = covariance.diag().sqrt()
    active = sd > sd.max() * 1e-6
    corr = covariance[active][:, active] / (sd[active, None] * sd[None, active])
    offdiag = corr[~torch.eye(len(corr), device=x.device, dtype=torch.bool)]
    return dict(shape=list(x.shape), max_centered_rank=min(len(x)-1, x.shape[1]),
                active_dimensions=int(active.sum()), participation_rank=float(1 / frac.square().sum()),
                entropy_rank=float(torch.exp(-(frac * frac.clamp_min(1e-300).log()).sum())),
                leading_variance_fraction=float(frac[-1]), top10_variance_fraction=float(frac[-10:].sum()),
                median_abs_column_correlation=float(offdiag.abs().median()),
                mean_squared_offdiag_correlation=float(offdiag.square().mean()))


def main():
    # Known nonzero centered covariance eigenvalues are both 2/3.
    toy = torch.tensor([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    s = spectrum(toy)
    assert abs(s['participation_rank'] - 2) < 1e-12
    assert abs(s['leading_variance_fraction'] - .5) < 1e-12
    assert abs(spectrum(7 * toy)['entropy_rank'] - s['entropy_rank']) < 1e-12
    prior_plan = HERE.parent/'midrug_training_size_20261007/results/plan.json'
    controls = json.loads(prior_plan.read_text())['controls']
    paths = [Path(__file__), PRIOR/'common.py', PRIOR/'model.py', prior_plan, CLEAN/'graph.pt', CLEAN/'inner_data.npz']
    paths += list((RELEASE/'incrd_hg_m123').glob('*.py'))
    rows=[]
    for fold in range(3):
        data=load_inner(fold)
        for name in ['initial','control','unmasked10']:
            seed_all(42)
            a=make_adapter(data, 'control' if name=='initial' else name, 'cuda:0')
            if name!='initial':
                c=controls[name][str(fold)]; meta=Path(c['path']); assert sha256(meta)==c['metadata_sha256']
                ckpt=meta.with_suffix('.pt'); assert sha256(ckpt)==c['checkpoint_sha256']
                a.model.load_state_dict(torch.load(ckpt,map_location='cuda:0',weights_only=True))
                paths.extend([meta,ckpt])
            m=a.model; m.eval()
            before={k:v.detach().clone() for k,v in m.state_dict().items()}; cache={}
            def capture_h(module,args): cache['h']=args[0]
            def capture_z(module,args): cache['z']=args[0]
            h1=m.lin1.register_forward_pre_hook(capture_h);h2=m.lin3.register_forward_pre_hook(capture_z)
            try:
                with torch.no_grad(): a.logits('train')
            finally: h1.remove(); h2.remove()
            sizes=[a.cfg.hidden]*2+list(a.ps.ncn_widths(a.cfg))+[a.cfg.hidden]*2
            assert sum(sizes)==cache['h'].shape[1]
            names=['M1_miRNA','M1_drug']+['M2_'+str(i) for i in range(len(sizes)-4)]+['M3_statistics','M3_EASE']
            representations={k:v for k,v in zip(names,cache['h'].split(sizes,dim=1))}
            representations['final_hidden']=cache['z']
            row=dict(config=name,fold=fold,train_pairs=len(data['train_y']),
                     spectra={k:spectrum(v) for k,v in representations.items()})
            for key,value in before.items():torch.testing.assert_close(value,m.state_dict()[key],rtol=0,atol=0)
            rows.append(row); print('PASS',name,fold,flush=True)
            del a,m,before,cache,representations;torch.cuda.empty_cache()
    result=dict(status='passed',records=rows,new_fits=0,new_checkpoints_loaded=6,initializations=3,
                no_validation_or_test_forward=True,no_parameter_or_buffer_updates=True,no_lncrna=True,
                scope='diagnostic spectra are not a claim that low rank causes error',
                source_hashes={str(p):sha256(p) for p in paths})
    (HERE/'precheck.json').write_text(json.dumps(result,indent=2))
    lines=[]
    for name in ['initial','control','unmasked10']:
        rs=[r for r in rows if r['config']==name]
        for part in rs[0]['spectra']:
            a=[r['spectra'][part] for r in rs]
            lines.append('| %s | %s | %.2f | %.2f | %.3f | %.3f |' % (name,part,np.mean([x['participation_rank'] for x in a]),np.mean([x['entropy_rank'] for x in a]),np.mean([x['leading_variance_fraction'] for x in a]),np.mean([x['mean_squared_offdiag_correlation'] for x in a])))
    (HERE/'PRECHECK.md').write_text('# 训练表示谱的只读预检\n\n| 模型 | 表示 | 参与率有效秩 | 熵有效秩 | 第一方向方差占比 | 列间平方相关均值 |\n|---|---|---:|---:|---:|---:|\n'+'\n'.join(lines)+'\n\n三折原训练查询；初始模型与六个已验证的选中模型。逐项核验参数/缓冲不变。没有新拟合、原验证/测试前向，也没有用这些谱重新选择历史模型。低维监督、重复端点和原始输入相关性都可能导致低有效秩，不能由此单独认定表示坍塌或保证去相关收益。\n')


if __name__=='__main__':main()
