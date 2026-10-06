"""Compile a bounded evidence index and BibTeX from verified primary records."""

from pathlib import Path
import hashlib
import json
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
NS = {"a": "http://www.w3.org/2005/Atom"}


def write_json(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atom_records():
    extra = ROOT / "additional_arxiv_metadata.xml"
    if not extra.is_file():
        url = "https://export.arxiv.org/api/query?" + urlencode({
            "id_list": "2501.07556v1,2602.08592v1,2510.22538v1",
        })
        try:
            with urlopen(Request(url, headers={"User-Agent": "AcademicEvidenceAudit/1.0"}), timeout=25) as response:
                raw = response.read()
            ET.fromstring(raw)
            extra.write_bytes(raw)
            write_json("additional_arxiv_fetch.json", {
                "url": url, "status": "ok", "sha256": hashlib.sha256(raw).hexdigest(),
            })
        except Exception as exc:
            write_json("additional_arxiv_fetch.json", {"url": url, "error": str(exc)})
            raise
    records = {}
    for path in [ROOT / "arxiv_metadata.xml", extra]:
        for entry in ET.fromstring(path.read_bytes()).findall("a:entry", NS):
            identifier = entry.findtext("a:id", namespaces=NS).split("/abs/")[-1]
            records[re.sub(r"v\d+$", "", identifier)] = {
                "arxiv_id": identifier,
                "title": " ".join(entry.findtext("a:title", namespaces=NS).split()),
                "authors": [a.findtext("a:name", namespaces=NS) for a in entry.findall("a:author", NS)],
                "date": entry.findtext("a:published", namespaces=NS)[:10],
                "updated": entry.findtext("a:updated", namespaces=NS),
            }
    return records


JOURNALS = [
    ("mirbind2_2026", "10.1093/bioinformatics/btag410", "2026-08-21",
     "primary_mirna_mechanism", "verified_abstract_and_official_readme",
     "核苷酸两两交互表示；从结合位点任务迁移到转录本抑制预测。",
     "出版社全文未取得；无 ncRNA–药物或 lncRNA 迁移证据。",
     ["btag410_crossref.json", "miRBind_2.0_README.md", "fulltext_fetch.json"]),
    ("mirformer_2026", "10.1093/bioinformatics/btag222", "2026-07-07",
     "mirna_local_resolution", "verified_abstract_and_official_readme",
     "成熟 miRNA/mRNA 双编码器、窗口注意力及位点定位。",
     "未做全文评价；靶位点任务不是药物关联任务。",
     ["btag222_crossref.json", "verified_pubmed.xml", "miRformer_README.md"]),
    ("mirbench_2025", "10.1093/bioinformatics/btaf233", "2025-07-15",
     "evaluation_bias", "verified_abstract_and_doi",
     "miRNA 频率类别偏置与负采样可以损害泛化；提供去偏基准。",
     "本地药物数据是否存在对应捷径仍需实测。", ["btaf233_crossref.json"]),
    ("protein_sae_2025", "10.1073/pnas.2506316122", "2025-08-19",
     "exploratory_interpretability", "verified_abstract_and_doi",
     "蛋白质语言模型的稀疏特征与可解释生物类别有关。",
     "不是 ncRNA–药物准确率或因果概念的证据。", ["pnas_2506316122_crossref.json"]),
    ("mifndra_2026", "10.1093/bib/bbag322", "2026-06-17",
     "overlapping_prior_work", "verified_abstract_and_doi",
     "预训练序列/药物特征、多模态融合、跨 ncRNA 知识及图学习。",
     "尚未做完整全文的新颖性比较。", ["bbag322_crossref.json"]),
    ("dcmetamg_2026", "10.1109/JBHI.2026.3717803", "2026-07-28",
     "overlapping_prior_work", "pubmed_abstract_electronic_date_and_crossref_title",
     "双粒度、静态结合/动态调控及两类 ncRNA–药物评价。",
     "机制措辞为作者主张；未独立验证因果解释或阅读全文。",
     ["JBHI_2026_3717803_crossref.json", "verified_pubmed.xml", "verified_pubmed_dates.json"]),
    ("rgdagmae_2026", "10.1016/j.compbiolchem.2026.109236", "2026-07-13",
     "overlapping_prior_work", "pubmed_abstract_electronic_date_and_crossref_title",
     "miRNA–药物路由注意力及图掩码重建。",
     "10 月卷期与 7 月电子发表不同；路由和掩码重建已经有先例。",
     ["j_compbiolchem_2026_109236_crossref.json", "verified_pubmed.xml", "verified_pubmed_dates.json"]),
]

PREPRINTS = [
    ("matchanything_2025", "2501.07556", "cross_domain_transfer_design",
     "fulltext_selected_methods_ablations_limitations",
     "合成跨模态监督与多来源预训练，使同一组权重用于八种以上未见图像模态匹配任务。",
     "无 RNA 证据；不将最优传输或无匹配项归于这篇论文；仅按 arXiv 版本引用。",
     ["additional_arxiv_metadata.xml", "matchanything_html.html"]),
    ("sae_rna_2025", "2510.02734", "demoted_exploratory_only",
     "fulltext_selected_methods_discussion_conclusion",
     "探索 RiNALMo 表示的稀疏特征分解。",
     "v2 明确提出特征稳定性、长度归一化及信噪验证尚未解决，不能承担可靠下游结论。",
     ["arxiv_metadata.xml", "sae_rna.pdf", "sae_rna.txt"]),
    ("ribospan_2026", "2608.22849", "optional_long_rna_encoder",
     "verified_metadata_and_abstract",
     "1.61B 参数 RNA 编码器，原生 10240 nt 上下文。",
     "仅摘要筛选；仍需完整输入序列；更换编码器本身不是新架构。", ["arxiv_metadata.xml"]),
    ("tfmlinker_2026", "2602.08592", "overlapping_graph_icl",
     "verified_primary_abstract_this_round_prior_local_method_review",
     "原型增强上下文与拓扑链接编码用于表格基础模型图 ICL。",
     "通用图 ICL 与原型检索不能在这里声称为新。",
     ["additional_arxiv_metadata.xml", "architecture_prior_check.json"]),
    ("isonetpp_2025", "2510.22538", "counterexample_operator_tradeoff",
     "verified_primary_metadata_and_abstract",
     "迭代 early interaction 改进先前的 late interaction 子图检索方法。",
     "不是 ncRNA 任务的实证；提示需要同预算的早期交互对照。",
     ["additional_arxiv_metadata.xml", "architecture_prior_check.json"]),
]


def escape(value):
    text = str(value)
    for source, target in [("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")]:
        text = text.replace(source, target)
    return text


def main():
    arxiv = atom_records()
    dois = {row["doi"]: row
            for name in ["verified_dois.json", "additional_verified_dois.json"]
            for row in json.loads((ROOT / name).read_text()) if "error" not in row}
    selected = []
    for key, doi, date, role, depth, finding, boundary, files in JOURNALS:
        row = dois[doi]
        authors = row.get("author", [])
        if not authors:
            for name in files:
                if name.endswith("_crossref.json"):
                    raw = json.loads((ROOT / name).read_text())
                    if raw.get("DOI", "").casefold() == doi.casefold():
                        authors = raw.get("author", [])
                        break
        if not authors:
            raise ValueError(f"Missing verified authors for {doi}")
        selected.append({
            "key": key, "title": row["title"][0], "doi": doi,
            "url": "https://doi.org/" + doi, "venue": row["venue"][0],
            "status": "peer_reviewed_journal", "date": date,
            "authors": [a.get("family", "") + ", " + a.get("given", "") for a in authors],
            "role": role, "read_depth": depth, "supported_finding": finding,
            "boundary": boundary, "evidence_files": files,
        })
    for key, aid, role, depth, finding, boundary, files in PREPRINTS:
        row = arxiv[aid]
        selected.append({
            "key": key, **row, "url": "https://arxiv.org/abs/" + row["arxiv_id"],
            "venue": "arXiv", "status": "cited_as_preprint_version",
            "role": role, "read_depth": depth, "supported_finding": finding,
            "boundary": boundary, "evidence_files": files,
        })
    meta = ET.fromstring((ROOT / "duplex_jats.xml").read_bytes()).find("./front/article-meta")
    selected.append({
        "key": "duplexfm_2026",
        "title": " ".join("".join(meta.find("./title-group/article-title").itertext()).split()),
        "doi": "10.64898/2026.08.06.743413", "url": "https://doi.org/10.64898/2026.08.06.743413",
        "venue": "bioRxiv", "status": "preprint_not_peer_reviewed", "date": "2026-08-12",
        "authors": [n.findtext("surname") + ", " + n.findtext("given-names")
                    for n in meta.findall("./contrib-group/contrib/name")],
        "role": "primary_transfer_mechanism",
        "read_depth": "fulltext_selected_methods_results_discussion_and_official_readme",
        "supported_finding": "冻结 miRNA 交互表示，以小型头迁移到 siRNA 效率；六个设置均优于描述符对照。",
        "boundary": "共同 Argonaute 机制支持 miRNA–siRNA 迁移；未验证 lncRNA；OligoFormer 在两个数据集更强。",
        "evidence_files": ["targeted_followup.json", "duplex_jats.xml", "duplex_readme.md"],
    })
    selected.append({
        "key": "lilac_2025",
        "title": "LILaC: Late Interacting in Layered Component Graph for Open-domain Multimodal Multihop Retrieval",
        "doi": "10.18653/v1/2025.emnlp-main.1037", "url": "https://aclanthology.org/2025.emnlp-main.1037/",
        "venue": "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
        "status": "verified_conference_proceedings", "date": "2025-11",
        "authors": ["Yun, Joohyung", "Lee, Doyup", "Han, Wook-Shin"],
        "role": "primary_cross_domain_operator",
        "read_depth": "official_venue_metadata_and_fulltext_selected_methods_ablations_limitations",
        "supported_finding": "细粒度子查询与子组件 late interaction 用于图边评分，在五个检索基准评价。",
        "boundary": "检索性能不证明生物学有效；本方案不引入其图遍历与 LLM 查询分解。",
        "evidence_files": ["lilac_venue_verification.json", "lilac_html.html", "lilac_README.md"],
    })
    for row in selected:
        row["evidence_sha256"] = {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in row["evidence_files"]
        }
    write_json("selected_sources.json", {
        "as_of": "2026-09-12",
        "scope": "Bounded multi-source search; not a systematic review or proof of novelty. No new-model performance claims.",
        "main_search_deduplicated_records": 95, "selected": selected,
    })
    bibs = []
    for row in selected:
        status = row["status"]
        kind = "article" if status == "peer_reviewed_journal" else "inproceedings" if status == "verified_conference_proceedings" else "misc"
        fields = {"title": "{" + row["title"] + "}", "author": " and ".join(row["authors"]),
                  "year": row["date"][:4], "url": row["url"]}
        if row.get("doi"):
            fields["doi"] = row["doi"]
        if kind == "article":
            fields.update(journal=row["venue"], note="First online date verified: " + row["date"])
        elif kind == "inproceedings":
            fields["booktitle"] = row["venue"]
        else:
            fields["note"] = row["venue"] + " preprint; cited version screened as of 2026-09-12"
        if row.get("arxiv_id"):
            fields.update(eprint=row["arxiv_id"], archivePrefix="arXiv")
        lines = ["  " + key + " = {" + (str(value) if key in ["url", "doi", "eprint"] else escape(value)) + "}"
                 for key, value in fields.items() if value]
        bibs.append("@" + kind + "{" + row["key"] + ",\n" + ",\n".join(lines) + "\n}")
    (ROOT / "references.bib").write_text("\n\n".join(bibs) + "\n")
    print(json.dumps({"selected_sources": len(selected), "bib_entries": len(bibs),
                      "keys": [row["key"] for row in selected]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
