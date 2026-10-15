"""Frozen teacher lag, masked context coverage and latent prediction diagnostics."""
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,load_inner,seed_all,sha256
from model import CONFIGS,make_adapter,load_masks,encode,normalized_target


def main():
    frozen=sha256(HERE/'selection.json');assert json.loads((HERE/'verification.json').read_text())['status']=='passed'
    records=[]
    for fold in range(3):
        data=load_inner(fold);masks=load_masks(fold,'cuda:0')
        for name,cfg in CONFIGS.items():
            path=HERE/'results'/f'{name}_f{fold}_s42.pt';meta=json.loads(path.with_suffix('.json').read_text());assert sha256(path)==meta['checkpoint_sha256']
            seed_all(42);a=make_adapter(data,name,'cuda:0');a.model.load_state_dict(torch.load(path,map_location='cuda:0',weights_only=True));m=a.model;m.eval()
            before={k:v.detach().clone() for k,v in m.state_dict().items()};row=dict(config=name,fold=fold)
            with torch.no_grad():
                prob=a.predict('train');row['train']=dict(auc=float(roc_auc_score(data['train_y'],prob)),ap=float(average_precision_score(data['train_y'],prob)))
                row['val']=dict(auc=meta['inner_auc'],ap=meta['inner_ap'])
                teacher=encode(m.encoder,m.context_teacher.teacher,{n:a.graph[n].x for n in ['miRNA','drug']})
                student=encode(m.encoder,m.encoder.original,{n:a.graph[n].x for n in ['miRNA','drug']})
                row['teacher_student_cosine']={n:float((normalized_target(teacher[n])*normalized_target(student[n])).sum(-1).mean()) for n in teacher}
                row['teacher_centered_rms']={n:float((h-h.mean(0,keepdim=True)).square().mean().sqrt()) for n,h in teacher.items()}
                # Examine the paired masks containing the selected update, using
                # the post-update checkpoint; these are not historical losses.
                start=((meta['best_epoch']-1)//2)*2;view_records=[]
                for epoch_index in [start,start+1]:
                    masked={n:x[epoch_index] for n,x in masks[cfg['mask']].items()}
                    loss,typed=m.context_teacher.loss(m.encoder,a.graph,masked);coverage={}
                    for e in a.graph.edge_types:
                        visible=(~masked[e[0]]).float()[:,None]
                        count=torch.sparse.mm(m.encoder.adj[e],visible).view(-1)[masked[e[2]]]
                        coverage['|'.join(e)]=dict(empty_fraction=float((count==0).float().mean()),mean_visible_degree=float(count.mean()))
                    view_records.append(dict(mask_epoch=epoch_index+1,latent=float(loss),typed_latent={n:float(v) for n,v in typed.items()},visible_context=coverage))
                row['selected_checkpoint_masks']=view_records
                row['mean_latent']=float(np.mean([x['latent'] for x in view_records]))
            for k,v in before.items():torch.testing.assert_close(v,m.state_dict()[k],atol=0,rtol=0)
            records.append(row);print('PASS',name,fold,flush=True);del a,m,before;torch.cuda.empty_cache()
    means=[]
    for name in CONFIGS:
        rows=[r for r in records if r['config']==name];item=dict(config=name)
        for part in ['train','val']:item[part]={m:float(np.mean([r[part][m] for r in rows])) for m in ['auc','ap']}
        item['mean_latent']=float(np.mean([r['mean_latent'] for r in rows]))
        item['teacher_student_cosine']={n:float(np.mean([r['teacher_student_cosine'][n] for r in rows])) for n in ['miRNA','drug']}
        means.append(item)
    assert sha256(HERE/'selection.json')==frozen
    (HERE/'component_diagnostics.json').write_text(json.dumps(dict(status='passed',records=records,means=means,selection_sha256=frozen,
        selection_changed=False,no_parameter_or_buffer_updates=True,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))
    table='\n'.join(f"| {r['config']} | {r['train']['auc']:.6f}/{r['train']['ap']:.6f} | {r['val']['auc']:.6f}/{r['val']['ap']:.6f} | {r['mean_latent']:.6f} | {r['teacher_student_cosine']['miRNA']:.6f}/{r['teacher_student_cosine']['drug']:.6f} |" for r in means)
    (HERE/'COMPONENT_DIAGNOSTICS.md').write_text(f'''# 选型冻结后的教师/上下文诊断

| 配置 | 训练AUC/AP | 内部AUC/AP | 成对遮蔽潜在损失 | 教师/学生中心化表示cosine(miRNA/drug) |
|---|---:|---:|---:|---:|
{table}

损失使用已选更新后的权重和包含该更新的两张预定互补掩码，不是保存训练曲线的
更新前损失。JSON另记录每种关系的空可见邻域比例、平均可见度与教师中心化RMS。
不同目标/遮蔽下潜在损失不能直接当作泛化分数；当前模型对照理论上教师与学生一致。
没有修改参数、缓冲、选型、遮蔽日程或训练预算，未打开原验证/测试。
''')


if __name__=='__main__':main()
