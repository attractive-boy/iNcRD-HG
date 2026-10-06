"""Build a verified bibliography and distinguish paper evidence from proposals."""
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from verify_sources import SELECTED, FULL_TEXT


ROOT = Path(__file__).resolve().parent
ORDER = ['mirbind2', 'mirnapex', 'parnet', 'lnc_rbp_occupancy', 'm2umol',
         'selformermm', 'forge', 'dispa', 'mphgnn', 'mifndra', 'mdadgan',
         'rgdagmae', 'mirbench', 'pulsnar', 'bin_pu', 'fourierdrug',
         'simple_controls', 'perturbation_benchmark', 'slim', 'mechanism_benchmark']

# All summaries are paraphrases. "project_transfer" is our proposed experiment,
# not a result reported in the cited paper or a claim of improvement here.
NOTES = {
 'mirbind2': dict(locators='相关全文：§2.2、§3.2；本地编号段落 0034–0048、0071–0078。',
  paper_finding='区分 50 nt 结合位点任务与完整 3′UTR 的 miRNA 干预后 mRNA log2FC 任务；后者迁移结合编码器并加入空间注意力和回归头。',
  limitation='基因抑制模型的训练使用 HeLa 优势 3′UTR；长度截到 3000 nt。测试 Pearson 约 0.30、AUC 约 0.60，不能把结合概率当作强功能效应，更没有直接验证 ncRNA–药物反应。',
  project_transfer='先保留现有位点到基因的对应关系；补齐正确 UTR 与转录本信息后，才比较 binding 与 repression 特征。'),
 'mirnapex': dict(locators='相关全文：段落 0012–0014、0059–0067、0071–0081。',
  paper_finding='联合靶基因表达变化和 APA/3′UTR 使用变化预测 miRNA 表达差异；提出有效靶位点剂量随异构体和状态改变。',
  limitation='APA 的总体增量有限且 miRNA 依赖；bulk 数据混合细胞状态，靶集和 PDUI 均为代理测量。作者明确不据此推断 APA 与 miRNA 稳定性的因果方向。',
  project_transfer='在有正确 UTR 和异构体使用数据的子集建模位点可用性；现有 FASTA 缓存不能直接提供这些变量。'),
 'parnet': dict(locators='复用预印本全文文本：第 62、143–153、185–189 行。',
  paper_finding='用 223 个 eCLIP 实验、150 个 RBP 的序列到结合信号任务学习 RNA 表示；冻结表示可用于多种 RNA 功能任务。',
  limitation='预印本；训练主要来自 K562/HepG2，覆盖的 RBP 和状态有限。论文的 lncRNA 染色质解离任务不同于核/胞质定位，也不同于药物反应。',
  project_transfer='当前通用均值/SD 表示已经验证过；新的假设应是具名 RBP 组成、局部单元或明确功能输出，不能再次把换 PARNET 本身当新实验。'),
 'lnc_rbp_occupancy': dict(locators='相关全文：§2.2–2.5、§3、§4.1–4.4；段落 0012–0039。',
  paper_finding='实测 eCLIP 的 RBP 组成在长度和总结合量之外增加定位预测信息；ridge 模型、染色体分组验证、跨 K562/HepG2 迁移。扩展协变量后仍有增量。',
  limitation='纯计算重分析，无新增干预验证；仅两个细胞系。占据定义为整个基因座（含内含子）上的实验峰重叠，不能等同成熟 lncRNA 序列上的 PARNET 预测。',
  project_transfer='先验证显式 RBP 组成对独立定位读出的增量，再考虑药物任务；核/胞质定位与染色质停留需分开。'),
 'm2umol': dict(locators='相关全文：段落 0012、0021、0038–0047、0069–0118；代码 README §3、§5.1。',
  paper_finding='将 3D、文本、药物类别/靶点/酶知识通过模态适配器迁移给 2D 分子编码器；预训练允许部分模态缺失，下游可只输入 SMILES 对应的 2D 图。',
  limitation='验证任务为分子属性、DDI、DTI，不能外推 ncRNA–药物收益。生成的模态表示不是实测数据；预训练文本含作用机制和临床结局描述，Bio 含靶点知识，需披露与下游证据重叠。',
  project_transfer='公开 README 给出冻结编码例子与权重链接；作为一次固定维度、固定预算的药物编码器对照。本轮未下载或运行权重。'),
 'selformermm': dict(locators='相关全文：§2、§3.3、§4；段落 0025–0047、0074–0089。',
  paper_finding='对齐 SELFIES、Uni-Mol 结构、SciBERT 文本与 CROssBARv2/DMGI 知识图谱表示；提供检查点和预计算表示。',
  limitation='正文消融显示单一补充模态有时优于全部模态；Text/KG 覆盖不齐。正文不同段落对结构有 2D/3D 表述差异，具体实现采用 Uni-Mol；不把所有视图当独立实验信息。',
  project_transfer='适合作为备选模型和缺失模态对照设计依据，优先级低于更直接的 M2UMol 冻结编码试验；避免重训约 2.46 亿参数模型。'),
 'forge': dict(locators='相关全文：§Discussion、§Methods；段落 0071–0075、0079–0105。',
  paper_finding='稀疏共享矩阵分解使基线表达同时解释 CRISPR gene dependency 和药物 IC50；输入特征的相关性筛选在训练侧进行。',
  limitation='主要按药物–靶点建模，外部 PDX 样本小，受表达尺度和批次影响。所用 dependency 概率与 Chronos GeneEffect 不是同一量；基因敲除也不是 ncRNA 调节。',
  project_transfer='在可准确对应的小药物–基因–细胞系集合上加真正 dependency 信息，先用稀疏线性/低秩模型检验，再决定是否增加图节点。'),
 'dispa': dict(locators='相关全文：§2.3–2.5、§4；段落 0023–0044、0085–0089。',
  paper_finding='用双向差分交叉注意力让药物子结构和通路表达相互条件化；报告 Drug-blind 与 Disjoint-set 等划分。',
  limitation='主要训练评价来自单一药理资源；单细胞/空间迁移缺少真实药敏标签；作者明确注意力不是分子相互作用的因果证据。',
  project_transfer='借鉴局部配对表示；先与同信息均值、线性/双线性交互比较，不仅与弱拼接模型比较。'),
 'mphgnn': dict(locators='相关全文：Materials and method；段落 0014–0034。',
  paper_finding='已构建 miRNA–gene–gene–drug 元路径，结合表达、ChemBERTa、HumanNet 与异构图/图自编码器。',
  limitation='论文任务与数据过滤、随机负样本协议有自身边界；其性能数值不能与本地不同数据直接排名。本文已足以证明增加 gene 节点/元路径不是独立新颖贡献。',
  project_transfer='贡献应放在位点级功能、作用类型、细胞依赖性与来源分组的验证，而不是节点数量。'),
 'mifndra': dict(locators='相关全文：§Feature representation learning、§Model architecture；段落 0044–0059。',
  paper_finding='已组合 SpliceBERT、GIN、跨 ncRNA 连接、PCA/KNN、带残差 GraphSAGE、多模态融合与 triplet 对比损失。',
  limitation='组合这些组件本身已有近邻先例；摘要/全文中的性能不构成本地表示增益的证据。ncRNA–药物负类语义仍需按具体来源审查。',
  project_transfer='降低通用预训练拼接、KNN、残差与通用对比损失作为主创新的优先级。'),
 'mdadgan': dict(locators='摘要与 Crossref/PubMed/Europe PMC 元数据；未阅读全文。',
  paper_finding='摘要明确描述 miRNA–lncRNA–drug 图、异构卷积、扩散对抗训练与代价敏感学习。',
  limitation='只据摘要判断技术先例；没有核验完整划分、消融或独立机制验证，不能据其宣称收益选择本地模型。',
  project_transfer='跨 RNA 类型扩图、扩散去噪和类别加权已有直接任务先例；进一步工作要有独立信息或更强验证。'),
 'rgdagmae': dict(locators='摘要与 Crossref/PubMed/Europe PMC 元数据；未阅读全文。',
  paper_finding='路由引导的全局低秩交互与门控 graph masked autoencoder 用于 miRNA–药物关联预测。',
  limitation='本轮仅读摘要，不能评定完整实现质量。PubMed 电子发表为 2026-07-13，Crossref 印刷卷期为 2026-10，在线发表早于检索截止日。',
  project_transfer='GraphMAE、门控和路由替换不宜独立包装为前沿突破；自监督应针对缺失功能或可核验关系。'),
 'mirbench': dict(locators='相关全文：§2.1–2.2；段落 0022–0039。',
  paper_finding='用仅 miRNA 序列频率相关的简单分类器诊断类别频率偏差；按 miRNA 家族匹配生成负例，并按位点染色体拆分。',
  limitation='它研究结合位点分类，不直接证明本项目所有成绩由频率造成；生成负例也不是逐条实验确认阴性。',
  project_transfer='保留实体频率与覆盖对照，报告药物内 AP 和长尾结果，区分跨药物分数重排与真正候选排序改善。'),
 'pulsnar': dict(locators='Crossref 摘要与元数据；本轮未阅读全文。',
  paper_finding='针对正例并非随机被标注的 SNAR 情形，按正例子群估计未标注集合中的正例比例与校准。',
  limitation='不能在完全无假设的情况下从 PU 数据识别真实发生率；本轮仅读摘要，未核查其全部理论条件。',
  project_transfer='把非随机收录作为方法假设显式报告，预先设置类别先验/收录机制敏感性分析，不直接把已观察比例当真实先验。'),
 'bin_pu': dict(locators='相关全文：§2.2、§3.2–3.4；段落 0023–0047、0051–0053。',
  paper_finding='将未标注 CPI 分箱训练，按集成评分生成伪正/伪负，再对已知正例增加损失权重；有外部 CYP 数据验证。',
  limitation='正文 §3.2 的交叉验证测试混入计算产生的伪标签；其内部 F1 不等同对实验阴性的评价。外部细菌文献和 human CYP 阴性是不同验证层次。',
  project_transfer='可借鉴置信度分层，但必须把伪负、未标注和真实阴性分开；不以模型自己的伪标签证明改进。'),
 'fourierdrug': dict(locators='相关全文：§2、§4；段落 0018–0047、0076–0078。',
  paper_finding='多来源域泛化、梯度反转，以及在编码后的潜在向量上施加 Fourier 变换和敏感/耐药非对称约束；训练不使用目标域数据。',
  limitation='标签来自真实 IC50 或临床响应，不是耐药株表达描述；把癌种域信息全消去可能也删除真实药敏条件。Fourier 输入是潜在特征，不能误述为沿基因排序直接做 FFT。',
  project_transfer='借鉴来源留出、状态保留和来源偏差控制；不立即复制 Fourier 模块，也不直接套用敏感样本聚集假设。'),
 'simple_controls': dict(locators='相关全文：§2.2–2.3、§3–4；段落 0019–0026、0048–0055、0061–0065。',
  paper_finding='在若干 K562 扰动任务，知道干预靶基因的简单均值对照可达到很强成绩；检查预训练权重与注意力的增量并修正数据标注。',
  limitation='模型知道 CRISPR 靶基因且依赖训练分布均值，不能泛化成所有药物/未知靶点任务均优于深度模型；多个种子也不是多个独立生物数据集。',
  project_transfer='对每种新功能信息同时给简单可解释的线性/均值基线，拆开信息贡献与复杂算子贡献。'),
 'perturbation_benchmark': dict(locators='PubMed/Europe PMC 摘要与 Crossref 元数据；本轮未阅读全文。',
  paper_finding='摘要报告 13 种方法/基线、25 个数据集与 24 项指标；跨细胞类型没有始终获胜的方法，表达相似度与差异基因恢复刻画不同目标。',
  limitation='只核读摘要；不将跨任务总结当作本项目数据上的性能估计，亦不从摘要断言某模型普遍最好。',
  project_transfer='按实际目标分别评价关联排序、描述方向和干预效应；保留简单基线与细胞/来源分组。'),
 'slim': dict(locators='Europe PMC 摘要与 Crossref 元数据；未读预印本全文。',
  paper_finding='摘要提出 64 维 STRING 表示和闭式双线性 ridge 模型，并报告小参数与低计算成本。',
  limitation='预印本、摘要级证据；结果来自若干数据集内的遗传扰动预测，不是 ncRNA–药物或跨研究泛化，论文计时不代表本地计时。',
  project_transfer='用相同基因信息的简单双线性模型作为增加 gene 节点之前的对照。'),
 'mechanism_benchmark': dict(locators='Europe PMC 摘要与 Crossref 元数据；未读预印本全文。',
  paper_finding='摘要提出独立机制注释和 Mechanism Fidelity Score，指出表达重构相似性与方向性关键基因恢复不一致，来源上下文影响评价。',
  limitation='预印本且本轮只读摘要；MFS 的实施细节和注释偏差尚未独立核验，不能当成熟的通用标准。',
  project_transfer='增加来自独立实验/论文的机制小集，评价关键基因方向与上下文一致性；不能用输入图谱本身再证明机制正确。'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    return ' '.join(html.unescape(re.sub('<[^>]+>', ' ', value or '')).split())


def main():
    if (ROOT / 'references.json').exists():
        raise ValueError('Refuse to overwrite archived bibliography')
    discovery = json.loads((ROOT / 'discovery_unique.json').read_text())
    discovered = {(r.get('doi') or '').lower(): r for r in discovery if r.get('doi')}
    reused = {r['key']: r for r in json.loads((ROOT / 'reused_sources.json').read_text())}
    pubmed = {}
    for article in ET.parse(ROOT / 'pubmed_selected.xml').getroot().findall('PubmedArticle'):
        identifiers = {e.attrib.get('IdType'): ''.join(e.itertext()) for e in article.findall('./PubmedData/ArticleIdList/ArticleId')}
        if identifiers.get('doi'):
            dates = []
            for d in article.findall('./MedlineCitation/Article/ArticleDate'):
                dates.append(dict(type=d.attrib.get('DateType'), year=d.findtext('Year'), month=d.findtext('Month'), day=d.findtext('Day')))
            abstract = ' '.join(''.join(e.itertext()) for e in article.findall('./MedlineCitation/Article/Abstract/AbstractText'))
            pubmed[identifiers['doi'].lower()] = dict(identifiers=identifiers, article_dates=dates, abstract=abstract)
    records, ris, note_lines = [], [], [
        '**20 篇文献的技术证据与迁移边界（截至 2026-09-14）**', '',
        '“论文发现”概括已读来源；“项目用途”是待验证建议。全文条目只表示核读相关方法、主要结果和讨论，不表示逐字阅读全部补充材料或复现实验。预印本均单独标记。', '']
    for number, key in enumerate(ORDER, 1):
        doi = SELECTED[key]
        cr_path = ROOT / ('crossref_retry_' + key + '.json')
        if not cr_path.exists():
            cr_path = ROOT / ('crossref_' + key + '.json')
        crossref = json.loads(cr_path.read_text())['message']
        assert crossref['DOI'].lower() == doi.lower()
        d, pm = discovered.get(doi.lower(), {}), pubmed.get(doi.lower(), {})
        preprint = crossref.get('type') == 'posted-content'
        raw_path = None
        if key in FULL_TEXT:
            raw_path = ROOT / ('full_' + key + '.xml')
            text_path = raw_path.with_suffix('.txt')
            depth = '全文相关方法/结果/讨论（本轮获取）'
        elif key in reused:
            raw_path = Path(reused[key]['reused_local_raw'])
            text_path = ROOT / reused[key]['readable_file']
            depth = '全文相关方法/结果/讨论（复用 2026-09-13 缓存）'
        elif key == 'parnet':
            text_path = ROOT.parent / 'frontier_reassessment_20260913/parnet_full_readable.txt'
            depth = '预印本全文相关段落（复用 2026-09-13 缓存）'
        else:
            text_path = ROOT / ('crossref_pulsnar.json' if key == 'pulsnar' else 'discovery_unique.json')
            depth = '摘要与元数据；未阅读全文'
        title = clean(crossref['title'][0])
        journal = clean((crossref.get('container-title') or ['bioRxiv' if preprint else ''])[0])
        year = crossref['published']['date-parts'][0][0]
        abstract = clean(d.get('abstract') or pm.get('abstract') or crossref.get('abstract'))
        if depth.startswith('摘要'):
            assert abstract, 'Missing actual abstract: ' + key
        citations = dict(crossref=dict(file=cr_path.name, sha256=sha(cr_path),
            url='https://api.crossref.org/works/' + doi, doi_match=True),
            reading_file=os.path.relpath(text_path, ROOT), reading_file_sha256=sha(text_path))
        if raw_path:
            citations.update(full_text_raw=os.path.relpath(raw_path, ROOT), full_text_sha256=sha(raw_path))
        row = dict(id=f'R{number:02}', key=key, title=title, doi=doi, year=year, journal=journal,
            authors=crossref.get('author', []), publication_type='preprint' if preprint else 'journal_article',
            doi_url='https://doi.org/' + doi, reading_depth=depth,
            publication_dates=dict(crossref_published=crossref.get('published'),
                crossref_online=crossref.get('published-online'), crossref_print=crossref.get('published-print'),
                crossref_posted=crossref.get('posted'), epmc_first_publication=d.get('date'),
                pubmed_article_dates=pm.get('article_dates', [])),
            pmid=pm.get('identifiers', {}).get('pubmed') or d.get('pmid'),
            pmcid=pm.get('identifiers', {}).get('pmc') or d.get('pmcid'),
            abstract=abstract, evidence=NOTES[key], sources=citations)
        if key == 'rgdagmae':
            row['date_resolution'] = 'PubMed ArticleDate Electronic=2026-07-13; Crossref 2026-10 is the print issue, not future online evidence.'
        if key == 'mirnapex':
            row['date_resolution'] = 'Crossref online=2026-04-23; Europe PMC firstPublicationDate=2026-06-16. Retain both source fields, cite year only.'
        if key == 'selformermm':
            row['date_resolution'] = 'Crossref online=2026-08-21; print issue date=2026-08-01. Cite year only.'
        records.append(row)
        ris.extend(['TY  - ' + ('UNPB' if preprint else 'JOUR'), 'ID  - ' + row['id'], 'TI  - ' + title])
        for author in row['authors']:
            name = ', '.join(v for v in [author.get('family'), author.get('given')] if v) or author.get('name')
            if name:
                ris.append('AU  - ' + name)
        ris.extend(['PY  - ' + str(year), 'T2  - ' + journal, 'DO  - ' + doi,
                    'UR  - ' + row['doi_url'], 'N1  - ' + depth + ('；预印本，未经同行评审。' if preprint else '')])
        if abstract:
            ris.append('AB  - ' + abstract)
        ris.extend(['ER  - ', ''])
        note_lines.extend([f'**{row["id"]} · {key} · {year} · {"预印本" if preprint else journal}**', '',
            f'[{title}]({row["doi_url"]})', '', f'阅读深度：{depth}。定位：{NOTES[key]["locators"]}', '',
            f'论文发现：{NOTES[key]["paper_finding"]}', '', f'限制：{NOTES[key]["limitation"]}', '',
            f'项目用途（待验证）：{NOTES[key]["project_transfer"]}', '',
            f'本地证据：[{text_path.name}]({os.path.relpath(text_path, ROOT)})；[Crossref 元数据]({cr_path.name})。', ''])
    strict = {}
    search_manifest = json.loads((ROOT / 'search_manifest.json').read_text())
    for result in search_manifest['results']:
        for row in result.get('records', []):
            doi = (row.get('doi') or '').strip().lower()
            key = ('doi', doi) if doi else ('title_author',
                re.sub(r'[^\w]+', ' ', row.get('title') or '').lower().strip(),
                (row.get('authors') or '').split(',')[0].strip().lower())
            if key not in strict or (not strict[key].get('pmcid') and row.get('pmcid')):
                strict[key] = row
    (ROOT / 'discovery_strict_unique.json').write_text(json.dumps(list(strict.values()), ensure_ascii=False, indent=2) + '\n')
    manifest = dict(as_of='2026-09-14', created_at_utc=datetime.now(timezone.utc).isoformat(),
        selection='Targeted 20-paper reading set; not a systematic review or performance meta-analysis.',
        discovery_deduplication='DOI first; normalized title plus first author when DOI absent. All selected references have distinct verified DOIs.',
        preliminary_discovery_records=len(discovery), strict_discovery_records=len(strict),
        selected_count=len(records), verified_crossref_doi_count=len(records),
        full_text_relevant_sections=sum(not r['reading_depth'].startswith('摘要') for r in records),
        abstract_only_count=sum(r['reading_depth'].startswith('摘要') for r in records),
        preprint_count=sum(r['publication_type'] == 'preprint' for r in records),
        supplements_newly_downloaded=0, references=records)
    assert len(records) == len({r['doi'] for r in records}) == 20
    (ROOT / 'references.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    (ROOT / 'references.ris').write_text('\n'.join(ris))
    (ROOT / 'READING_NOTES.md').write_text('\n'.join(note_lines))
    print(json.dumps({k: v for k, v in manifest.items() if k != 'references'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
