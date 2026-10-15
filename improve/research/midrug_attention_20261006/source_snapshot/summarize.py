"""Archive validation-only attention and intrinsic-chemistry experiments."""
import csv
import json
import shutil
from model import HERE,RELEASE,sha256
from incrd_hg_m123.data import load_graph,split_path


def main():
    audit=json.loads((HERE/'verification.json').read_text())
    assert audit['status']=='passed' and len(audit['records'])==17
    rows=[]
    for family,directory in [('attention',HERE/'results'),('chemistry',HERE/'chemical_results')]:
        for p in sorted(directory.glob('*.json')):
            r=json.loads(p.read_text())
            if 'val_auc' not in r:continue
            rows.append(dict(family=family,tag=r['tag'],val_auc=r['val_auc'],val_ap=r['val_ap'],selection_score=r['selection_score'],
                best_epoch=r['best_epoch'],test_evaluated=r['test_evaluated']))
    assert len(rows)==17 and all(not r['test_evaluated'] for r in rows)
    with (HERE/'validation_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    previous=HERE.parent/'midrug_internal_20261006'
    protection=json.loads((previous/'plmf_protection.json').read_text())
    for relative,digest in protection['source_hashes'].items():assert sha256(RELEASE/relative)==digest
    for dataset in ('MDR','MDS'):load_graph(dataset)
    for r in protection['records']:
        assert sha256(split_path(r['dataset'],r['fold']))==r['split_sha256']
        cp=HERE.parent/'baselines_m123_20261004/results'/f"incrd_m123_{r['dataset']}_f{r['fold']}_s42.pt"
        assert sha256(cp)==r['checkpoint_sha256']
    (HERE/'plmf_hash_recheck.json').write_text(json.dumps(dict(status='passed',records=10,
        scope='Previous full PLMF replay remains applicable; source, frozen data and checkpoint hashes unchanged',
        prior_verification_sha256=sha256(previous/'plmf_protection.json')),indent=2))
    target=json.loads((HERE.parent/'midrug_signedpaths_20261006/TARGET.json').read_text())
    (HERE/'TARGET.json').write_text(json.dumps(target,indent=2,ensure_ascii=False))
    table='\n'.join(f"| {r['family']} | {r['tag']} | {r['val_auc']:.9f} | {r['val_ap']:.9f} |" for r in rows)
    report='''# MiDrug 查询相关注意力与原始分子结构实验

目标仍未达到。用户要求 AUC、AP 均相对原始 iNcRD-HG 绝对提高 0.02；
同一固定测试集的三种子均值目标为 AUC ≥0.9733929274、AP ≥0.9336833737。
本轮完成 17 个验证配置，全部 seed42，没有新候选测试评价，也未修改发布代码。

## 实验结果

| 实验族 | 配置 | 验证 AUC | 验证 AP |
|---|---|---:|---:|
'''+table+'''

六个学习注意力配置均低于静态权重对照。八个替代分子图也没有同时优于其原图对照：
Morgan2 k3 的 AUC 稍高、AP 较低；pharm2 k10 的 AP 稍高、AUC 较低。
这些结果不支持将上述变体纳入发布路径。

## 方法与边界

- 注意力：四个 M2 路径头使用原始表达谱/ChemBERTa 查询与键，
  权重为 exp(T*tanh(score))，查询投影从零开始，初始门与固定权重一致。
  只依赖节点原始协变量，不依赖当前查询标签；M1/M3 与原辅助图保留。
- 分子图：原 216 个药物 SMILES，重新计算 Morgan2/Morgan3/MACCS/feature-Morgan2
  Tanimoto，相似度 top-3/top-10 无向并集替换 DrugDrug 边。保留原自环重数，
  其他关系、全部节点特征、全部训练监督不变。没有下载外部数据或预训练模型。
- 来源核验：216 个 CID/SMILES/特征行顺序一致；原 ChemBERTa 行 min-max 与冻结
  药物特征完全相同；重新生成的原 Morgan2 count Tanimoto 与原 drugSimMat 最大误差
  2.956e-8。逐文件 SHA256 和 RDKit 版本见 chemical_provenance.json。
- 训练：RMSprop 1e-4，最多 700 epochs，最少 300；最优轮后 200 轮不改善即停。
  所有候选按 min(验证 AUC 增益, 验证 AP 增益) 选轮，锚点为原始模型同验证集
  三种子均值 0.9554707319/0.9181315885。

## 核验

verification.json：17 个已训练 checkpoint 预测重放、保存指标和双指标选轮重算通过。
每模型 12 对显式删边对照通过。注意力最终 logit 检查包含学习后的非均匀门；
通用中间池检查使用 forward 后已清除的门，不将其误述为学习门下的逐池核验。
初始检查还验证非零注意力梯度。分子图检查确认仅 DrugDrug 关系改变。
第一次审计在最后一个训练结果尚未落盘时停止；训练全部完成后完整重跑通过。

plmf_hash_recheck.json：先前已完整重推理的 10 个 PLMF checkpoint、冻结数据与发布代码
哈希均未改变。MDR/MDS 的既有测试均值仍为 0.9483092901/0.9436701562 与
0.9508978837/0.9490119744；这里是保留原路径的证据，不是新方法在 PLMF 上的验证。

## 继续方向

下一实验位于 ../midrug_profiles_20261006：从训练关联行/列中屏蔽当前配对后，
用非线性去噪重建分支预测目标，与 M1/M2/M3 联合训练。这个方向改变关联模式的
表示方式；不会继续放大本轮失败的注意力强度或分子邻居数。目标保持 active。

复现使用 /data1/licheng/miniconda3/envs/ncRNAdrug/bin/python。
check.py → launch.py；chemistry.py → launch_chem.py；audit.py → summarize.py。
已有训练结果拒绝覆盖；需在独立目录复跑。完整精度见 validation_summary.csv。
'''
    (HERE/'REPORT.md').write_text(report)
    snap=HERE/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
    state=dict(goal_status='active',completion_proven=False,configurations=17,test_evaluations=0,
        current_target=target,next_action='Nonlinear query-masked positive/negative interaction profiles in sibling directory',
        processes_expected_live=[])
    (HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2,ensure_ascii=False))
    print('PASS archive: 17 audited validation configurations; protected PLMF hashes unchanged',flush=True)

if __name__=='__main__':main()
