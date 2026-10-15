"""Archive nonlinear-profile experiments and inspect validation complementarity."""
import csv,json,shutil
import numpy as np
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score,average_precision_score
from model import HERE,CONFIGS,sha256
from train import anchor_metrics,load_fold


def main():
    audit=json.loads((HERE/'verification.json').read_text());assert audit['status']=='passed' and len(audit['records'])==8
    rows=[]
    for name in CONFIGS:
        r=json.loads((HERE/'results'/f'{name}_s42.json').read_text())
        rows.append(dict(tag=r['tag'],val_auc=r['val_auc'],val_ap=r['val_ap'],best_epoch=r['best_epoch'],test_evaluated=r['test_evaluated']))
    with (HERE/'validation_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    reference=HERE.parent/'midrug_internal_20261006/fullgraph_drop/drop0.2_s42_val.npz'
    base=np.load(reference);records=[];ba,bp=anchor_metrics(load_fold('MiDrug',0))
    for name in ('only_pos256_d5','only_signed256_d5'):
        pred=np.load(HERE/'results'/f'{name}_s42_val.npz')
        np.testing.assert_array_equal(pred['y'],base['y']);np.testing.assert_array_equal(pred['pairs'],base['pairs'])
        for weight in (0.,.1,.25,.5,.75,1.):
            p=expit((1-weight)*logit(np.clip(base['probability'],1e-6,1-1e-6))+weight*logit(np.clip(pred['probability'],1e-6,1-1e-6)))
            a,b=roc_auc_score(base['y'],p),average_precision_score(base['y'],p)
            records.append(dict(tag=name,weight=weight,val_auc=a,val_ap=b,selection_score=min(a-ba,b-bp)))
    (HERE/'diagnostic_blends.json').write_text(json.dumps(dict(records=records,test_evaluated=False,
        reference_sha256=sha256(reference),purpose='Validation complementarity only; not an accepted M123 model'),indent=2))
    best=max(records,key=lambda r:r['selection_score'])
    table='\n'.join(f"| {r['tag']} | {r['val_auc']:.9f} | {r['val_ap']:.9f} | {r['best_epoch']} |" for r in rows)
    report='''# MiDrug 非线性关联重建分支

双指标 +0.02 目标尚未达到。固定 MiDrug 划分，只用原数据与训练监督。
完成 8 个验证配置；5 个联合分支、2 个独立分支诊断和 1 个 M123 对照。
全部新增联合分支均低于本轮 M123 对照，不纳入发布模型。

| 配置 | 验证 AUC | 验证 AP | 选中轮次 |
|---|---:|---:|---:|
'''+table+f'''

## 实现

对配对 (r,d)，先将正例邻接矩阵和负例指示矩阵的 (r,d) 单元清零，
再分别输入 RNA 行与药物列的两层 MLP。正例模式仅使用正例矩阵；双通道模式
额外使用训练抽样负例，负例不是生物学上的耐药/敏感方向标签。
两个目标特异读出平均形成重建 logit，作为 M1/M2/M3 的加性分支。
读出从零初始化，初始预测匹配同参数 M123 对照（最大浮点差约 3.6e-7）。

原始 lncRNA/miRNA/drug 图、全部原始特征、9,712 个训练监督对不变。
分支比较 128/256 隐层与 0/20/50% 输入丢弃，隐藏层丢弃 10%。
主干 RMSprop 1e-4，分支 3e-4、weight decay 1e-4，联合模式额外用 0.1 倍分支 BCE。
最多 1,000 epochs，最少 300，最优轮后 200 轮不改善即停。
按 min(验证 AUC 增益, 验证 AP 增益) 选轮，不使用测试标签选型。
分支最优轮普遍较早，只能排除这些已测训练配置，不能宣称所有重建方法无效。

独立分支与前一轮 M123 的诊断混合，最佳双指标选择结果：
{best['tag']}，权重 {best['weight']}，验证 AUC/AP {best['val_auc']:.9f}/{best['val_ap']:.9f}。
完整固定权重比较见 diagnostic_blends.json；这是验证诊断，不是最终达标模型。

## 核验与输入审计

initial_checks.json 验证初始零分支对照、非零读出下的 6 正/6 负监督记录删除、
12 个正例最终 logit 显式删边、验证/测试标签翻转不影响证据、有限非零梯度。
verification.json 复核全部 8 个已训练 checkpoint 的验证预测、指标、双指标选轮，
并重做分支与最终 logit 删除检查。所有检查通过。

追查原始 RNA 表达矩阵时，最初按 CSV 文件顺序比较，发现 miRNA 不同；
进一步按原代码与 miRNA_info 的细胞 ID 排序后，两类 RNA 冻结特征均精确复现，
938 个细胞顺序一致。这不是实体错配，也没有证据推翻此前对齐表达的实验结果。

后续检查 ../midrug_expression_20261006 的原始表达归一化：原按行 min-max 丢失
表达幅值，新候选只变换同一原始表达值，关联图和划分保留。目标仍 active。
未改发布模型或 PLMF 路径；PLMF 保护哈希复核见 ../midrug_attention_20261006/plmf_hash_recheck.json。

复现：check.py → launch.py → audit.py → summarize.py。
解释器 /data1/licheng/miniconda3/envs/ncRNAdrug/bin/python。
'''
    (HERE/'REPORT.md').write_text(report)
    target=json.loads((HERE.parent/'midrug_signedpaths_20261006/TARGET.json').read_text())
    (HERE/'TARGET.json').write_text(json.dumps(target,indent=2,ensure_ascii=False))
    (HERE/'CONTINUATION.json').write_text(json.dumps(dict(goal_status='active',completion_proven=False,
        configurations=8,test_evaluations=0,best_diagnostic_blend=best,next_action='Original RNA expression preprocessing in sibling experiment'),indent=2))
    snap=HERE/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
    print(json.dumps(dict(status='passed',best_diagnostic_blend=best)),flush=True)

if __name__=='__main__':main()
