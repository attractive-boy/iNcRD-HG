# -*- coding: utf-8 -*-
"""Ablation ledger as an Excel table, one row per change, plain-language
descriptions of how each change was actually implemented."""
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

HEADERS = ['阶段', '改动名称', '具体改法（详细）', '为什么有效',
           'LncDrug ΔAUC', 'LncDrug ΔAUPR',
           'MiDrug ΔAUC', 'MiDrug ΔAUPR', '显著性 / 备注']

ROWS = [
    ['① 作者本人的模型', 'base（不改任何东西）',
     '逐字复现作者的代码。数据划分用作者冻结的 RandomLinkSplit(验证20%/测试20%, 负样本2:1)。模型是两跳异构图卷积(GeneralConv, LncDrug用aggr=min、MiDrug用mean) + MLP解码器；通道数64，学习率5e-4，训练100个epoch，在测试集上选最优epoch记录。10个随机种子取平均。',
     '这是基线，所有改动的对比对象。复现结果已经超过论文打印值(LncDrug 0.9857→0.9866)，那属于复现误差，不算改进。',
     '—', '—', '—', '—',
     'LncDrug test 0.9866/0.9624；MiDrug test 0.9559/0.9155。'],
    ['② GNN 内部改动', 'posw2 正样本加权',
     '改损失函数：把二分类交叉熵(BCE)里正样本的权重设成负样本的2倍(pos_weight=2.0)，就改一行代码。损失=BCEWithLogitsLoss(pos_weight=2.0)。',
     '损失更重视"把真正的正样本排到最前面"。AUPR只统计排名最前面的部分，所以加权后AUPR直接上升。这是整个项目里唯一在两个ncRNA上都显著为正的机制。',
     '+0.0002', '−0.0002',
     '+0.0006', '+0.0021',
     'MiDrug S_test AUPR 10/10种子, p=1e-4；LncDrug S_val AUPR +0.0031(8/10, p=0.009)。'],
    ['② GNN 内部改动', 'dropedge_0.2 训练期随机删边',
     '训练时每个epoch以20%概率随机删掉部分消息传递的图边(用被删边的图训练，测试时仍用完整图)。相当于给图卷积加正则化，和Dropout类似但作用于边。',
     '防止模型死记图结构导致过拟合，训练更稳。这是第一个同时提升LncDrug和MiDrug的机制(之前所有机制都只对一个任务有效)。',
     '+0.0011', '+0.0038',
     '+0.0005', '+0.0013',
     'MiDrug 9/10种子 p=0.01；LncDrug 8/10 p≈0.05~0.06(边缘显著)。'],
    ['② GNN 内部改动', 'mc10 推理期MC-DropEdge',
     '推理时不用完整图算一次，而是随机生成10个删边图，每个图各算一次得分，最后取平均。消除"训练时缺边、推理时满边"的不一致(和MC Dropout补dropout缺口是同一个思路)。',
     '让推理时的图和训练时更一致，方差更小。只在MiDrug有效：MiDrug图稠密(每个miRNA约9条边)，删边有意义；LncDrug图太稀(每个lncRNA约1条边)，删边后图退化了。',
     '0', '−0.0011',
     '+0.0006', '+0.0013',
     '仅MiDrug, 10/10种子, p=0.002/0.004。'],
    ['② GNN 内部改动', 'norm 编码器LayerNorm',
     '在两层图卷积之间插入LayerNorm层归一化。',
     '归一化hidden state，稳定稠密图训练(LncDrug的每个节点聚合的度差异小所以无效，MiDrug度差异大所以有效)。这是整个项目里唯一有效的架构改动——之前试过的HAN/HGT/LightGCN/超图全部变差。',
     '−0.0011', '−0.0011',
     '+0.0012', '+0.0007',
     '仅MiDrug, 10/10种子, p=0.002/0.006。'],
    ['② GNN 内部改动', 'HP search 重做超参搜索',
     '把作者的find_best_params.py重跑，修三个bug：①每epoch都记录得分而不是只读最后一个epoch；②用3个种子平均而不是1个种子(666)；③把posw2加进搜索范围。找到LncDrug最优配置=通道256 / 学习率5e-4 / aggr=min。',
     '作者自己的搜索只看到64通道，而256通道的模型会先过拟合再收敛，作者的"只读最后epoch"规则把它错误判低。修正规则后找到更宽的模型，LncDrug四个指标全超过论文。',
     '+0.0017', '+0.0044',
     '+0.0005', '+0.0022',
     'LncDrug S_test +0.0017/+0.0044 超论文。'],
    ['② 融合改动', 'GBDT 树模型成员',
     '不用神经网络，改用梯度提升树。先手工构造19个"配对特征"：两个端点的度数、相似度加权的协同得分、3跳和5跳路径数、关联矩阵的SVD潜在因子、跨ncRNA桥计数等，然后喂给树打分。关键防泄漏：训练时把自己那条边从特征里删掉(K折)，否则验证集崩到0.4997(等于瞎猜)。',
     '这是第一个非GNN的模型类。树读的是显式的配对特征，GNN学不出来(如(A·Aᵀ·A)[r,d]这类路径计数)；两个模型犯的错不一样(Spearman 0.77-0.85)，把分数混合后8/8个格子都比GNN单模型强。',
     '+0.0028', '+0.0135',
     '+0.0019', '+0.0023',
     '混合后 val 0.9856/0.9642 · 0.9576/0.9218。'],
    ['② 融合改动', '6成员融合（验证集选权）',
     '把 gnn、gbdt、extratrees、logreg、mlp、xt35 六个评分器的分数各自转成排名(0~1)，再线性加权平均。权重在验证集上用作者自己的准则(AUC+AUPR)/2网格搜索得到。',
     '模型类多样性能互补：树、线性模型、MLP、GNN各自擅长不同配对。排名加权让量纲统一。',
     '—', '—',
     '—', '—',
     '绝对值：LncDrug test 0.9921/0.9856；MiDrug test 0.9604/0.9228。'],
    ['② 融合改动', '额外负样本 5倍',
     '原来每个树只用冻结的2:1标注集(LncDrug只有3801行)。现在额外从"训练关联矩阵中不存在的配对"里采样5倍于正样本数的负样本(严格保证不碰验证/测试集的信息)，一起喂给树成员训练。',
     'LncDrug每个lncRNA只有约1条边，协同特征估计得不准，负样本不够导致分类边界欠定；补足后边界更准。MiDrug每节点约9条边已够，所以无效。',
     '+0.0008', '+0.0025',
     '+0.0002', '+0.0005',
     '仅LncDrug显著：test AUPR 95%CI [+0.0005,+0.0050], P(Δ>0)=0.997。'],
    ['② 融合改动', 'NCN 公共邻居池化成员',
     '新加一个小的神经网络成员(来自NCNC论文, ICLR 2024)。对每一对(ncRNA, drug)计算"公共邻居"：①与这个drug化学相似的其他drug中、该ncRNA已关联过的，②与此ncRNA相似的ncRNA中、已关联这个drug的。把这些邻居的向量加权平均(池化)，连上两个端点的原始属性，过一个小MLP打分。防泄漏同GBDT(K折+相似矩阵对角为0)。',
     '公共邻居计数是强配对信号。消融发现真正起作用的是"计数"这个数字+MLP的非线性表达，而不是池化/消息传递本身；它和GBDT里已有的collab特征是同一个数，本质是给已有特征加一个非线性MLP成员，多样性高(Spearman 0.78-0.86)，犯的错和树/线性成员都不同，融入后MiDrug大涨。',
     '+0.0028', '+0.0033',
     '+0.0110', '+0.0053',
     'AUC bootstrap p=0.997(Lnc)/1.000(Mi)，可靠；AUPR的置信区间碰到0(p≈0.92)，只能报AUC。'],
    ['③ 所有改动合并的最大值', 'NCN 融合（5成员）',
     '在作者模型基础上叠加前面全部改动后的最终报告模型：extratrees .45/logreg .25/ncn .30(LncDrug)；logreg .20/gbdt .10/gnn .25/ncn .45(MiDrug)，验证集选权。',
     '把模型类多样性(树+线性+GNN+非线性MLP)和训练技巧(posw2、dropedge、额外负样本)全部叠起来。',
     '+0.0070', '+0.0225',
     '+0.0151', '+0.0115',
     'LncDrug test 0.9936/0.9849；MiDrug test 0.9710/0.9270。MiDrug已超过"读测试标签"的oracle上界0.9563。'],
    ['② 图构造改动', 'top-k 邻居 + 连续边权',
     '把相似网络从全局阈值 |sim|>0.3 换成"每节点保留 top-k 最相似邻居"(k=2/3/5)，同时把连续相似度 |sim| 作为 edge_attr 加权消息传递(weighted_edges.py --edge-set knn --mode raw)。期间修了 kNN 路径的 NaN 漏洞：LncSimMat_ken 每行有 2 个 NaN，argpartition 会让 NaN 进 top-k，加权后产出 NaN 预测(已改为在分区前把 NaN 按 -inf 排除)。',
     '否定的。阈值图度分布(6.6/26.8/3.9)虽不均，但 top-k 在验证集更讨喜(S_val 涨)、测试集反而跌，k=2/3/5 都不行；带上连续边权也救不回来——权重对 top-k 拓扑是平到负(与 knn3-bin 比 S_test AUPR -0.0003)。这证实并推广了早先"kNN 别用"的结论。',
     '−0.0005', '+0.0019',
     '−0.0009', '−0.0017',
     'S_test(knn3-raw vs 发布版，n=3/2)。S_val 三格为正(3/3)，是验证集拟合。'],
    ['② 图构造改动', 'signed 保留相关符号（区分正/负相关）',
     '去掉发布版对相似矩阵的 np.abs()(heter_data_prepare.py line 27)：--mode signed 把带符号 sim 直接作边权(负权重=互斥/抗调控邻居，反向消息)；--mode split 把同一套边集按符号拆成 _pos/_neg 两个关系、各带 |sim|。因边集与 |sim| 权重完全相同，signed vs raw 是纯符号效应。LncLnc 里有 234 条 sim<-0.3 的反相关对，发布版一直把它们当强共表达。',
     'MiDrug 是整条图构造轴里唯一测试集有方向的结果：S_test AUPR +0.0025(6/7 种子)，靠 LncLnc 的反相关边经共享 LncMi 桥传导到药物嵌入(MiMi 自己只有 2 条反相关对)。LncDrug 纯符号效应落在噪声内；split 拆两类关系只是验证集效应，补种子后消失。',
     '+0.0003', '−0.0022',
     '+0.0017', '+0.0025',
     'S_test：MiDrug AUPR 6/7 种子(p≈0.06，量小且集中在 0/2/3 号种子)，LncDrug n=5。signed 只动 234 条边的符号，量级微弱，不是头条。'],
    ['② 图构造改动', 'ncRNA 节点特征：表达统计量 + 压缩',
     '把发布的 938 维 CCLE 表达谱整向量换成(a)3 个表达统计量(行均值/行标准差/零比例, expr_stats3)、(b)PCA-64+3 统计、(c)938→256→64 自编码器瓶颈+3 统计(expr_ae64)；全部只在训练集节点上拟合、替换(非追加)输入。另加 expr_mlp 模型：消息传递前加一层可训练 Linear 投影(对 938 个细胞系维度做学习加权，即 B2 的细胞系上下文注意力)。',
     '938 维输入对 GNN 是噪声/高方差——行级表达特征(均值/稀疏度)模型自己从 938 个相关细胞系列里学不出来，3 个标量直接给到。这是全文件第一个测试集上双任务都正的特征侧改动(MiDrug test AUPR 5/5)。',
     '+0.0018', '+0.0113',
     '+0.0021', '+0.0044',
     'S_test(AE-64，n=5，5/5 种子)。机制其实是 expr_stats3(+0.0085/+0.0041)；PCA/AE 在 3 统计之上只加 LncDrug +0.001~0.003、MiDrug≈0，不能单独归因于压缩。expr_mlp 可训练投影在 raw 上有 +0.006/+0.002，但在压缩输入上为负。RNA-FM 序列通道已测为 null。'],
]

wb = Workbook()
ws = wb.active
ws.title = '消融总账'

header_fill = PatternFill('solid', fgColor='4472C4')
header_font = Font(bold=True, color='FFFFFF', size=11)
stage_fill = {
    '①': PatternFill('solid', fgColor='FFF2CC'),
    '②': PatternFill('solid', fgColor='E2EFDA'),
    '③': PatternFill('solid', fgColor='FCE4D6'),
}

for c, h in enumerate(HEADERS, 1):
    cell = ws.cell(row=1, column=c, value=h)
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)

for r, row in enumerate(ROWS, 2):
    for c, v in enumerate(row, 1):
        cell = ws.cell(row=r, column=c, value=v)
        cell.alignment = Alignment(vertical='top', wrap_text=True)
    stage = row[0][0]
    ws.cell(row=r, column=1).fill = stage_fill[stage]

widths = [14, 20, 70, 55, 13, 13, 13, 13, 40]
for i, w in enumerate(widths, 1):
    ws.column_dimensions[get_column_letter(i)].width = w
ws.freeze_panes = 'A2'

out = 'abl_ledger.xlsx'
wb.save(out)
print('saved', out)