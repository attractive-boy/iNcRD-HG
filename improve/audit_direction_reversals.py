"""Audit existing direction records as candidates for a context-reversal task.

This reads source annotations only, not model test predictions. Contradictions
are not automatically reclassified as biological context effects. Cell-line
aliases require a unique Cellosaurus accession before entering strict counts.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import pandas as pd

from improve.first_round_features import sha256, write_json


def normalize(name):
    return re.sub('[^a-z0-9]', '', str(name).lower())


def cell_accessions(path, wanted):
    aliases, details = {}, {}
    accession, canonical, names, disease = None, None, [], None
    with open(path, encoding='utf-8') as source:
        for line in source:
            if line.startswith('//'):
                for name in names:
                    key = normalize(name)
                    if key in wanted and accession is not None:
                        aliases.setdefault(key,set()).add(accession)
                        details[accession] = dict(name=canonical,disease=disease)
                accession, canonical, names, disease = None, None, [], None
            elif line.startswith('AC   '):
                accession = line[5:].strip()
            elif line.startswith('ID   '):
                canonical = line[5:].strip()
                names.append(canonical)
            elif line.startswith('SY   '):
                names.extend(name.strip() for name in line[5:].split(';') if name.strip())
            elif line.startswith('DI   ') and disease is None:
                parts = line[5:].split(';')
                disease = parts[2].strip() if len(parts)>2 else None
    return aliases, details


def pair_set(index):
    return set((str(row[0]),int(row[1])) for row in index)


def audit(records, context, aliases, details, destination):
    annotated = records.reset_index().rename(columns={'index':'record'})
    merged = context.merge(annotated,on='record',suffixes=('','_source'),validate='many_to_one')
    if len(merged)!=len(context) or set(context.record)!=set(annotated.record):
        raise ValueError('Context-to-source record mapping is incomplete')
    for name in ('ncRNA','cid','effect'):
        if not merged[name].eq(merged[name+'_source']).all():
            raise ValueError('Context annotation disagrees with source: '+name)
    named = merged[merged.cell_line.notna()].copy()
    named['alias_key'] = named.cell_line.map(normalize)
    named['alias_matches'] = named.alias_key.map(lambda key:len(aliases.get(key,set())))
    named['cell_accession'] = named.alias_key.map(lambda key:next(iter(aliases[key])) if len(aliases.get(key,set()))==1 else None)
    trusted = named[named.cell_accession.notna()].copy()
    trusted['canonical_cell'] = trusted.cell_accession.map(lambda key:details[key]['name'])
    # Every PMID contributes at most one observation per pair/context/effect,
    # regardless of duplicate rows or the number of synonyms in Condition.
    evidence = trusted.drop_duplicates(['ncRNA','cid','cell_accession','effect','pmid']).copy()
    evidence.to_csv(destination/'evidence.csv',index=False)
    keys = ['ncRNA','cid','cell_accession']
    per_context = evidence.groupby(keys).agg(effects=('effect','nunique'),
        papers=('pmid','nunique'),effect=('effect','first'),canonical_cell=('canonical_cell','first'))
    per_context.to_csv(destination/'pair_contexts.csv')
    pair_summary = per_context.reset_index().groupby(['ncRNA','cid']).agg(
        contexts=('cell_accession','nunique'),conflicting_contexts=('effects',lambda x:int((x>1).sum())))
    overall = evidence.groupby(['ncRNA','cid']).effect.nunique()
    pair_summary['directions'] = overall
    strict = pair_set(pair_summary[(pair_summary.directions>1)&(pair_summary.conflicting_contexts==0)].index)
    replicated = per_context[(per_context.effects==1)&(per_context.papers>=2)].reset_index().groupby(['ncRNA','cid']).effect.nunique()
    replicated = pair_set(replicated[replicated>1].index) & strict
    study_context = evidence.groupby(['ncRNA','cid','pmid','cell_accession']).effect.nunique()
    bad_studies = set(study_context[study_context>1].reset_index()[['ncRNA','cid','pmid']].itertuples(index=False,name=None))
    study = evidence.groupby(['ncRNA','cid','pmid']).agg(directions=('effect','nunique'),contexts=('cell_accession','nunique'))
    same_study = study[(study.directions>1)&(study.contexts>1)].reset_index()
    study_mask = pd.Series([tuple(row) not in bad_studies for row in same_study[['ncRNA','cid','pmid']].itertuples(index=False,name=None)], index=same_study.index, dtype=bool)
    same_study = same_study.loc[study_mask]
    same_study.to_csv(destination/'same_study_candidates.csv',index=False)
    same_study_pairs = pair_set(same_study[['ncRNA','cid']].itertuples(index=False,name=None))
    pair_summary['strict_candidate'] = [tuple(index) in strict for index in pair_summary.index]
    pair_summary['same_study_support'] = [tuple(index) in same_study_pairs for index in pair_summary.index]
    pair_summary['both_directions_replicated'] = [tuple(index) in replicated for index in pair_summary.index]
    pair_summary.to_csv(destination/'pair_audit.csv')
    raw_pair_effects = records.groupby(['ncRNA','cid']).effect.nunique()
    return dict(source_records=len(records),source_pairs=len(raw_pair_effects),
        source_discordant_pairs=int((raw_pair_effects>1).sum()),source_papers=int(records.pmid.nunique()),
        max_records_per_paper=int(records.groupby('pmid').size().max()),
        expanded_context_rows=len(context),named_context_rows=len(named),
        unique_accession_context_rows=len(trusted),mapped_source_records=int(trusted.record.nunique()),
        ambiguous_alias_rows=int((named.alias_matches>1).sum()),unmapped_alias_rows=int((named.alias_matches==0).sum()),
        deduplicated_study_context_evidence=len(evidence),canonical_cell_lines=int(evidence.cell_accession.nunique()),
        same_cell_conflicting_pair_contexts=int((per_context.effects>1).sum()),
        within_study_conflicting_pair_contexts=int((study_context>1).sum()),
        strict_opposite_direction_pairs=len(strict),
        same_study_opposite_context_pairs=len(same_study_pairs),
        strict_pairs_with_same_study_support=len(strict & same_study_pairs),
        strict_pairs_with_both_directions_replicated=len(replicated))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='/data1/licheng/incrd-hg/incrd-run')
    parser.add_argument('--out-dir',required=True)
    args=parser.parse_args()
    root,out=Path(args.root),Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    inputs,frames,wanted={}, {},set()
    for task in ('LncDir','MiDir'):
        rp,cp=root/'11.direction'/(task+'_records.csv'),root/'13.context'/(task+'_context.csv')
        frames[task]=(pd.read_csv(rp),pd.read_csv(cp))
        wanted.update(frames[task][1].cell_line.dropna().map(normalize))
        inputs[str(rp)]=sha256(rp);inputs[str(cp)]=sha256(cp)
    cell_path=root/'13.context'/'cellosaurus.txt'
    aliases,details=cell_accessions(cell_path,wanted)
    inputs[str(cell_path)]=sha256(cell_path)
    result=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha256(__file__),
                input_sha256=inputs,definition='annotation candidates, not verified biological reversals',tasks={})
    for task,(records,context) in frames.items():
        destination=out/task
        destination.mkdir(exist_ok=True)
        result['tasks'][task]=audit(records,context,aliases,details,destination)
        print(task,json.dumps(result['tasks'][task],ensure_ascii=False),flush=True)
    write_json(out/'AUDIT.json',result)


if __name__=='__main__':
    main()
