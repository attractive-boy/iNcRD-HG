"""Frozen-checkpoint drug-query and expression-column alignment interventions."""
import json
import math
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from common import HERE, load_inner, seed_all, sha256
from model import CONFIGS, make_adapter


def metrics(y, probability):
    return dict(auc=roc_auc_score(y, probability), ap=average_precision_score(y, probability))


@torch.no_grad()
def examine(a, data):
    m = a.model; m.eval()
    before = {name: value.clone() for name, value in m.state_dict().items()}
    captured = {}
    h1 = m.lin1.register_forward_pre_hook(lambda module, args: captured.update(features=args[0]))
    h2 = m.cell_projection.register_forward_hook(lambda module, args, value: captured.update(addition=value))
    try:
        full = a.predict('val')
    finally:
        h1.remove(); h2.remove()
    base = m.lin1(captured['features'])
    weight = m.cell_projection.weight.clone()
    try:
        m.cell_projection.weight.zero_()
        zero = a.predict('val')
    finally:
        m.cell_projection.weight.copy_(weight)
    conditioned = m.cell_context.conditioned
    try:
        m.cell_context.conditioned = not conditioned
        swapped = a.predict('val')
    finally:
        m.cell_context.conditioned = conditioned
    # Deliberately break column-profile/scalar alignment, keeping RNA scalar
    # expression positions unchanged. This is a post-training intervention.
    permutation = torch.arange(a.dense[0].shape[1]-1, -1, -1, device=a.y.device)
    hook = m.cell_context.cell_encoder.register_forward_hook(lambda module, args, value: value[permutation])
    try:
        misaligned = a.predict('val')
    finally:
        hook.remove()
    keys, values, queries = m.cell_context.encode(a.dense[0], a.dense[1])
    r, d = a.pairs['val']
    entropy, effective, maximum = [], [], []
    for start in range(0, len(r), 256):
        _, alpha = m.cell_context.attend(keys, values, queries, r[start:start+256], d[start:start+256])
        entropy.append(-(alpha*alpha.clamp_min(1e-30).log()).sum(-1)/math.log(alpha.shape[1]))
        effective.append(1/alpha.square().sum(-1))
        maximum.append(alpha.max(-1).values)
    for name, value in m.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    assert m.cell_context.conditioned == conditioned
    assert not m.lin1._forward_hooks and not m.lin1._forward_pre_hooks
    assert not m.cell_projection._forward_hooks and not m.cell_context.cell_encoder._forward_hooks
    result = dict(full=metrics(data['val_y'],full), zero_projection=metrics(data['val_y'],zero),
                  swapped_condition=metrics(data['val_y'],swapped), misaligned_columns=metrics(data['val_y'],misaligned),
                  zero_mean_abs_probability_shift=float(abs(full-zero).mean()),
                  swapped_mean_abs_probability_shift=float(abs(full-swapped).mean()),
                  misalignment_mean_abs_probability_shift=float(abs(full-misaligned).mean()),
                  projection_weight_norm=float(weight.norm()), addition_rms=float(captured['addition'].square().mean().sqrt()),
                  base_pre_activation_rms=float(base.square().mean().sqrt()),
                  normalized_attention_entropy=float(torch.cat(entropy).mean()),
                  effective_columns=float(torch.cat(effective).mean()), mean_max_attention=float(torch.cat(maximum).mean()))
    return result, full


def main():
    plan = json.loads((HERE/'results/plan.json').read_text())
    selection = sha256(HERE/'selection.json')
    assert json.loads((HERE/'verification.json').read_text())['status'] == 'passed'
    rows = []
    for fold in range(3):
        data = load_inner(fold)
        for name in CONFIGS:
            path = HERE/'results'/f'{name}_f{fold}_s42.json'
            metadata = json.loads(path.read_text())
            assert sha256(path.with_suffix('.pt')) == metadata['checkpoint_sha256']
            seed_all(42); a = make_adapter(data, name, 'cuda:0')
            a.model.load_state_dict(torch.load(path.with_suffix('.pt'), map_location='cuda:0', weights_only=True))
            result, full = examine(a,data)
            with np.load(path.with_name(path.stem+'_inner.npz')) as stored:
                np.testing.assert_allclose(full,stored['probability'],atol=2e-5,rtol=2e-5)
            rows.append(dict(config=name,fold=fold,checkpoint_sha256=metadata['checkpoint_sha256'],**result))
            print('PASS',name,fold,flush=True)
            del a; torch.cuda.empty_cache()
    means = []
    modes = ['full','zero_projection','swapped_condition','misaligned_columns']
    for name in CONFIGS:
        records = [row for row in rows if row['config'] == name]
        row = dict(config=name)
        for mode in modes:
            row[mode] = {metric:float(np.mean([r[mode][metric] for r in records])) for metric in ['auc','ap']}
        for key in records[0]:
            if key not in ['config','fold','checkpoint_sha256']+modes:
                row[key] = float(np.mean([r[key] for r in records]))
        means.append(row)
    assert sha256(HERE/'selection.json') == selection
    for path, digest in plan['source_hashes'].items():
        assert sha256(path) == digest
    (HERE/'component_diagnostics.json').write_text(json.dumps(dict(status='passed',records=rows,means=means,
        selection_sha256=selection,selection_changed=False,no_parameter_or_buffer_updates=True,
        original_validation_evaluated=False,test_evaluated=False,
        limitation='Post-training interventions do not undo joint backbone training; query swapping and reversed profile alignment are not refitted models. Attention is not biological response evidence.'),indent=2))
    table = '\n'.join(f"| {r['config']} | {r['full']['auc']:.9f}/{r['full']['ap']:.9f} | {r['zero_projection']['auc']:.9f}/{r['zero_projection']['ap']:.9f} | {r['swapped_condition']['auc']:.9f}/{r['swapped_condition']['ap']:.9f} | {r['misaligned_columns']['auc']:.9f}/{r['misaligned_columns']['ap']:.9f} | {r['effective_columns']:.2f} |" for r in means)
    (HERE/'COMPONENT_DIAGNOSTICS.md').write_text(f'''# 冻结选型后的表达列诊断

使用相同固定检查点，比较完整模型、临时置零新投影、切换真实/平均药物query，
以及仅反转列背景编码的顺序（RNA标量表达位置保持不变），不重新训练或换候选。
最后一个干预故意破坏背景与标量的列对齐；不同于同步置换整个矩阵的无害不变性。

| 配置 | 完整AUC/AP | 置零投影 | 切换query | 列背景错位 | 平均有效列数 |
|---|---:|---:|---:|---:|---:|
{table}

有效列数为`1/sum(alpha²)`，最大938；归一化熵、最大权重、概率变化和增量RMS
保存在JSON。它们描述已训练注意力分布，不提供细胞特异反应或因果解释。
所有参数、缓冲及模式精确恢复，hook清理，selection.json哈希不变。
置零投影不撤销它对主干训练的影响，其余干预不是重新训练的机制对照。
''')


if __name__ == '__main__':
    main()
