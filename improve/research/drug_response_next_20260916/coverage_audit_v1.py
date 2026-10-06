"""Coverage audit for a cell-line drug-response channel (GDSC release 8.5).

Read-only: loads the already-downloaded GDSC cell-line catalogue, the screened
compound list, the DepMap model table and the project expression matrices. It
does not read any ncRNA-drug association label and does not train anything.
"""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

ROOT = Path('/data1/licheng/incrd-hg')
GDSC = ROOT / 'data_external/drug_response_20260916/gdsc'
NET = ROOT / 'incrd-run/2.construct_network'
DEPMAP_MODEL = ROOT / 'data_external/pathway_20260913/depmap/Model.csv'
OUT = Path(__file__).resolve().parent


def expression_cells(path):
    with path.open() as handle:
        reader = csv.reader(handle)
        next(reader)
        return [row[0] for row in reader]


def main():
    cells = expression_cells(NET / 'MiExp.csv')
    lnc_cells = expression_cells(NET / 'LncExp.csv')
    project_cells = sorted(set(cells) | set(lnc_cells))

    model = pd.read_csv(DEPMAP_MODEL, low_memory=False)
    model['COSMICID'] = pd.to_numeric(model['COSMICID'], errors='coerce')
    cosmic_to_ach = {int(c): a for c, a in zip(model['COSMICID'], model['ModelID']) if pd.notna(c)}
    name_to_ach = {}
    for name, stripped, ach in zip(model['CellLineName'], model['StrippedCellLineName'], model['ModelID']):
        for candidate in (name, stripped):
            if isinstance(candidate, str):
                name_to_ach.setdefault(candidate.upper().replace('-', ''), ach)

    gdsc_cells = pd.read_excel(GDSC / 'Cell_Lines_Details.xlsx')
    gdsc_cells.columns = [c.strip() for c in gdsc_cells.columns]
    cosmic_col = next(c for c in gdsc_cells.columns if 'COSMIC' in c.upper())
    name_col = next(c for c in gdsc_cells.columns if c.strip().upper() in ('SAMPLE NAME', 'CELL LINE NAME'))
    gdsc_cosmic = set(pd.to_numeric(gdsc_cells[cosmic_col], errors='coerce').dropna().astype(int))
    matched_ach = {cosmic_to_ach[c] for c in gdsc_cosmic if c in cosmic_to_ach}
    overlap = sorted(set(project_cells) & matched_ach)
    by_name = sorted({name_to_ach[str(n).upper().replace('-', '')] for n in gdsc_cells[name_col]
                      if str(n).upper().replace('-', '') in name_to_ach})
    overlap_name = sorted(set(project_cells) & set(by_name))

    with (NET / 'Drug.csv').open() as handle:
        reader = csv.reader(handle)
        next(reader)
        drugs = [row[0].strip().upper() for row in reader]
    compounds = pd.read_csv(GDSC / 'screened_compounds_rel_8.5.csv')
    gdsc_drugs = {str(n).strip().upper() for n in compounds['DRUG_NAME']}
    drug_overlap = sorted(set(drugs) & gdsc_drugs)

    result = dict(accessed_at_utc=datetime.now(timezone.utc).isoformat(),
        expression_cells=len(project_cells),
        gdsc_cell_lines=len(gdsc_cells), gdsc_cosmic_ids=len(gdsc_cosmic),
        matched_cosmic_to_ach=len(matched_ach), cell_overlap_by_cosmic=len(overlap),
        cell_overlap_by_name=len(overlap_name),
        project_drugs=len(drugs), gdsc_screened_compounds=len(compounds),
        drug_overlap=len(drug_overlap), drug_overlap_names=drug_overlap,
        cell_overlap_sample=overlap[:15])
    (OUT / 'COVERAGE_V1.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'drug_overlap_names'}, indent=2))
    print('drug overlap:', ', '.join(drug_overlap[:60]))


if __name__ == '__main__':
    main()
