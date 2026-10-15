"""Frozen-checkpoint prototype and gradient diagnostics; no updates or selection."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from common import HERE, load_inner, seed_all, sha256
from model import CONFIGS, make_adapter, mask_generator, capture_features, training_context, prototype_loss


def gradient_stats(reference, auxiliary):
    dot = (reference*auxiliary).sum()
    norm = (reference.norm()*auxiliary.norm()).clamp_min(1e-30)
    return dict(cosine_to_clean=float(dot/norm),
                weighted_norm_ratio=float(.05*auxiliary.norm()/reference.norm().clamp_min(1e-30)))


def examine(a, fold):
    m = a.model; m.eval()
    before = {name: value.clone() for name, value in m.state_dict().items()}
    logits, z = capture_features(a)
    clean = F.binary_cross_entropy_with_logits(logits, a.y)
    cfg = CONFIGS['boundary']
    context = training_context(z, a.y, cfg, mask_generator(a.y.device, 20261007+fold))
    other = {**context, 'selected': context['random']}
    boundary = prototype_loss(z, a.y, context, cfg['temperature'])
    random = prototype_loss(z, a.y, other, cfg['temperature'])
    gb = torch.autograd.grad(clean, z, retain_graph=True)[0]
    gs = torch.autograd.grad(boundary, z, retain_graph=True)[0]
    gr = torch.autograd.grad(random, z)[0]
    assert all(param.grad is None for param in m.parameters())
    assert not m.lin3._forward_pre_hooks
    for name, value in m.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    def group(indices):
        return dict(mean_abs_margin=float(context['margin'][indices].mean()),
                    mean_clean_bce=float(F.binary_cross_entropy_with_logits(logits[indices], a.y[indices])),
                    zero_count=int((a.y[indices] == 0).sum()), positive_count=int((a.y[indices] == 1).sum()))
    bg, rg = group(context['boundary']), group(context['random'])
    assert bg['zero_count'] == rg['zero_count'] and bg['positive_count'] == rg['positive_count']
    assert bg['mean_abs_margin'] <= rg['mean_abs_margin']+1e-10
    return dict(clean_bce=float(clean), boundary_auxiliary=float(boundary), random_auxiliary=float(random),
                boundary_group=bg, random_group=rg,
                prototype_cosine=float((context['prototypes'][0]*context['prototypes'][1]).sum()),
                boundary_gradient=gradient_stats(gb, gs), random_gradient=gradient_stats(gb, gr))


def main():
    plan = json.loads((HERE/'results/plan.json').read_text())
    selection = sha256(HERE/'selection.json')
    assert json.loads((HERE/'verification.json').read_text())['status'] == 'passed'
    paths = {name: [Path(refs[str(fold)]['path']) for fold in range(3)] for name, refs in plan['controls'].items()}
    paths.update({name: [HERE/'results'/f'{name}_f{fold}_s42.json' for fold in range(3)] for name in CONFIGS})
    rows = []
    for fold in range(3):
        data = load_inner(fold)
        for name, records in paths.items():
            path = records[fold]
            metadata = json.loads(path.read_text())
            assert sha256(path.with_suffix('.pt')) == metadata['checkpoint_sha256']
            seed_all(42)
            a = make_adapter(data, name, 'cuda:0')
            a.model.load_state_dict(torch.load(path.with_suffix('.pt'), map_location='cuda:0', weights_only=True))
            row = dict(config=name, fold=fold, checkpoint_sha256=metadata['checkpoint_sha256'], **examine(a, fold))
            if name in CONFIGS:
                curves = json.loads(path.with_name(path.stem+'_curves.json').read_text())
                active = [r for r in curves if r['auxiliary_weight']]
                row['mean_training_auxiliary_to_bce_ratio'] = float(np.mean([r['auxiliary_weight']*r['auxiliary']/r['clean_bce'] for r in active]))
            rows.append(row)
            print('PASS',name,fold,flush=True)
            del a; torch.cuda.empty_cache()
    means = []
    for name in paths:
        rs = [r for r in rows if r['config'] == name]
        row = dict(config=name)
        for key in ['clean_bce','boundary_auxiliary','random_auxiliary','prototype_cosine']:
            row[key] = float(np.mean([r[key] for r in rs]))
        for kind in ['boundary_gradient','random_gradient']:
            row[kind] = {key: float(np.mean([r[kind][key] for r in rs])) for key in ['cosine_to_clean','weighted_norm_ratio']}
        row['boundary_mean_abs_margin'] = float(np.mean([r['boundary_group']['mean_abs_margin'] for r in rs]))
        row['random_mean_abs_margin'] = float(np.mean([r['random_group']['mean_abs_margin'] for r in rs]))
        means.append(row)
    assert sha256(HERE/'selection.json') == selection
    for path, digest in plan['source_hashes'].items():
        assert sha256(path) == digest
    result = dict(status='passed',records=rows,means=means,selection_sha256=selection,
                  no_parameter_or_buffer_updates=True, selection_changed=False,
                  original_validation_evaluated=False,test_evaluated=False,
                  limitation='Gradients are measured on existing checkpoints in eval mode, not replayed training gradients or proof of historical causality. Statistics only use training pairs.')
    (HERE/'component_diagnostics.json').write_text(json.dumps(result,indent=2))
    table = '\n'.join(f"| {r['config']} | {r['boundary_auxiliary']:.6f} | {r['random_auxiliary']:.6f} | {r['boundary_mean_abs_margin']:.6f} | {r['random_mean_abs_margin']:.6f} | {r['boundary_gradient']['cosine_to_clean']:.6f} | {r['boundary_gradient']['weighted_norm_ratio']:.6f} |" for r in means)
    (HERE/'COMPONENT_DIAGNOSTICS.md').write_text(f'''# 冻结选型后的训练信号诊断

检查12个新模型及6个已有对照，不做参数更新、不访问原验证/测试、不重新选择。
eval模式原训练配对表示上分别计算类内边界及同样本数/类别比例随机损失。
所有参数/缓冲前后逐元素相同，临时hook清理，模型参数不累积梯度。

| 配置 | 边界辅助损失 | 随机辅助损失 | 边界平均距离 | 随机平均距离 | 边界辅助与BCE梯度余弦 | .05辅助/BCE梯度范数比 |
|---|---:|---:|---:|---:|---:|---:|
{table}

梯度相对于lin3前的配对表示，mean/scale/原型/索引保持停止梯度。
它只说明当前检查点的训练信号是否存在、是否对准BCE；不是训练轨迹重放，也不能
证明辅助目标造成了某项性能变化。eval模式BN不同于训练批次BN，数值不与训练
曲线逐步等同。该目标没有推理分支，不能用“关掉推理损失”作因果消融。
''')


if __name__ == '__main__':
    main()
