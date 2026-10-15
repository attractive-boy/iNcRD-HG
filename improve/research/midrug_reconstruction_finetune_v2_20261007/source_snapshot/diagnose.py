"""Read-only, post-selection diagnostics of saved models and training curves."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from common import HERE, load_inner, seed_all, sha256
from model import CONFIGS, make_adapter, source_record


@torch.no_grad()
def examine(a, data):
    result = {}
    for part in ['train', 'val']:
        logits = a.logits(part); prob = logits.sigmoid().cpu().numpy()
        labels = torch.as_tensor(data[part + '_y'], dtype=logits.dtype, device=logits.device)
        result[part] = dict(auc=float(roc_auc_score(data[part + '_y'], prob)),
                            ap=float(average_precision_score(data[part + '_y'], prob)),
                            bce=float(F.binary_cross_entropy_with_logits(logits, labels)))
    indices = {n: torch.arange(a.graph[n].num_nodes, device=a.y.device) for n in ('miRNA', 'drug')}
    rec, separate = a.model.reconstruction.loss(a.model.encoder, a.graph, indices, False)
    result['full_node_reconstruction'] = dict(mean=float(rec), **{k: float(v) for k, v in separate.items()})
    result['train_minus_val'] = {m: result['train'][m] - result['val'][m] for m in ['auc', 'ap']}
    return result


def main():
    selection_hash = sha256(HERE / 'selection.json')
    assert json.loads((HERE / 'verification.json').read_text())['status'] == 'passed'
    plan = json.loads((HERE / 'results/plan.json').read_text()); records = []
    for fold in range(3):
        data = load_inner(fold); source, origin = source_record(fold)
        origin_state = torch.load(source.with_suffix('.pt'), map_location='cpu', weights_only=True)
        for name in CONFIGS:
            path = HERE / 'results' / f'{name}_f{fold}_s42.pt'
            meta = json.loads(path.with_suffix('.json').read_text()); assert sha256(path) == meta['checkpoint_sha256']
            seed_all(42); a = make_adapter(data, name, 'cuda:0')
            a.model.load_state_dict(torch.load(path, map_location='cuda:0', weights_only=True)); a.model.eval()
            before = {k: v.detach().clone() for k, v in a.model.state_dict().items()}
            values = examine(a, data)
            for key, value in before.items(): torch.testing.assert_close(value, a.model.state_dict()[key], atol=0, rtol=0)
            shift = {}
            for group in ['encoder', 'reconstruction', 'head']:
                selected = [n for n, _ in a.model.named_parameters()
                            if (n.startswith(group + '.') if group != 'head' else not n.startswith(('encoder.', 'reconstruction.')))]
                delta = sum(float((before[n].cpu().double() - origin_state[n].double()).square().sum()) for n in selected)
                norm = sum(float(origin_state[n].double().square().sum()) for n in selected)
                shift[group] = dict(l2=float(np.sqrt(delta)), relative_l2=float(np.sqrt(delta / norm)), parameter_tensors=len(selected))
            curves = json.loads(path.with_name(path.stem + '_curves.json').read_text())
            for metric in ['auc', 'ap']: assert abs(values['val'][metric] - meta['inner_' + metric]) < 2e-6
            final = curves[-1]
            result = dict(config=name, fold=fold, checkpoint_sha256=sha256(path), selected_stage_epoch=meta['best_epoch'],
                          selected=values, selected_parameter_shift_from_source=shift,
                          final_step=dict(stage_epoch=final['stage_epoch'], auc=final['inner_auc'], ap=final['inner_ap'],
                                          train_mode_pre_update_bce=final['bce'], sampled_pre_update_reconstruction=final['reconstruction']),
                          final_minus_source={m: final['inner_' + m] - origin['inner_' + m] for m in ['auc', 'ap']},
                          curve_snapshots=[r for r in curves if r['stage_epoch'] in [0, 1, 10, 50, 99, 100, 101, meta['epochs_run']]])
            records.append(result); print('PASS', name, fold, flush=True)
            del a, before; torch.cuda.empty_cache()
    means = []
    for name in CONFIGS:
        rows = [r for r in records if r['config'] == name]
        means.append(dict(config=name,
            selected_train={m: float(np.mean([r['selected']['train'][m] for r in rows])) for m in ['auc', 'ap', 'bce']},
            selected_val={m: float(np.mean([r['selected']['val'][m] for r in rows])) for m in ['auc', 'ap', 'bce']},
            selected_reconstruction=float(np.mean([r['selected']['full_node_reconstruction']['mean'] for r in rows])),
            final_val={m: float(np.mean([r['final_step'][m] for r in rows])) for m in ['auc', 'ap']},
            final_minus_source={m: float(np.mean([r['final_minus_source'][m] for r in rows])) for m in ['auc', 'ap']},
            final_train_mode_pre_update_bce=float(np.mean([r['final_step']['train_mode_pre_update_bce'] for r in rows]))))
    assert sha256(HERE / 'selection.json') == selection_hash
    for path, digest in plan['source_hashes'].items(): assert sha256(path) == digest
    result = dict(status='passed', records=records, means=means, selection_sha256=selection_hash, selection_changed=False,
                  no_parameter_or_buffer_updates=True, no_lncrna=True, outer_validation_evaluated=False, test_evaluated=False,
                  limitation='Final checkpoint weights were not saved. Final metrics and pre-update train-mode losses come from frozen training curves; selected train/eval gaps and parameter shifts use replayed selected models only.')
    (HERE / 'component_diagnostics.json').write_text(json.dumps(result, indent=2))
    table = '\n'.join(f"| {r['config']} | {r['selected_train']['auc']:.6f}/{r['selected_train']['ap']:.6f} | {r['selected_val']['auc']:.6f}/{r['selected_val']['ap']:.6f} | {r['final_val']['auc']:.6f}/{r['final_val']['ap']:.6f} | {r['final_minus_source']['auc']:+.6f}/{r['final_minus_source']['ap']:+.6f} |" for r in means)
    (HERE / 'COMPONENT_DIAGNOSTICS.md').write_text(f'''# 选型冻结后的训练曲线诊断

| 分支 | 选中模型训练 AUC/AP | 选中模型内部留出 AUC/AP | 最后一步内部留出 AUC/AP | 最后一步相对源模型 ΔAUC/ΔAP |
|---|---:|---:|---:|---:|
{table}

选中模型的训练/留出指标、BCE、全节点标准化重构MSE和三组参数偏移见JSON。
最后一步指标来自已冻结曲线；没有保存最后模型权重，不声称重放过最后模型。
曲线BCE/MSE来自训练模式、更新前，不能与选中模型评估模式BCE直接作因果比较。
所有诊断均在选型冻结后执行，没有按最后一步、单折、单指标或曲线重选候选。
源模型已经在同一内部留出选轮，本结果只否定这一固定微调协议的推进资格，
不证明所有辅助任务衰减方法无效，也不证明主任务微调永远不能改善泛化。
''')


if __name__ == '__main__': main()
