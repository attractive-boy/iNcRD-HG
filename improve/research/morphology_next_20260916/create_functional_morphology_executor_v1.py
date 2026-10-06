"""Derive the narrower protocol while preserving all training/prediction code."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
source = ROOT / 'anchor_morphological_alignment_v1.py'
target = ROOT / 'anchor_functional_morphology_v1.py'
report = HERE / 'FUNCTIONAL_MORPHOLOGY_EXECUTOR_DERIVATION_V1.json'
if target.exists() or report.exists():
    raise ValueError('Preserve executor derivation')
original = source.read_text()
text = original.replace('anchor_morphological_alignment_v1.py', 'anchor_functional_morphology_v1.py')
text = text.replace('morphological_alignment_v1.py', 'functional_morphology_v1.py')
text = text.replace('test_morphological_alignment_v1.py', 'test_functional_morphology_v1.py')
tree = ast.parse(text)
own = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'OWN' for x in node.targets))
names = tuple(dict.fromkeys(ast.literal_eval(own.value) + ('morphological_alignment_v1.py', 'anchor_morphological_alignment_v1.py',
                 'test_morphological_alignment_v1.py', 'test_functional_morphology_v1.py', 'adopt_functional_morphology_v1.py')))
lines = text.splitlines(keepends=True)
lines[own.lineno-1:own.end_lineno] = ['OWN = ' + repr(names) + '\n']
text = ''.join(lines)
tree = ast.parse(text)
prepare = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'prepare')
descriptions = dict(
    likelihood='BCE on covered nested-OOF reference queries plus a bounded learned morphological residual. External positive DTI batches and their loss are computed with a zero multiplier in both arms solely to preserve the already observed parent full training exactly; there is no DTI-supervised gradient.',
    training_budget='400 AdamW updates and 400 backward evaluations; identical frozen profiles, network, dropout/private RNG, zero auxiliary weight and twenty checkpoint evaluations. Full maps to the parent no_alignment arm. The only control changes RNA-to-gene identity by the frozen global permutation. Exact parent screen adoption is separately audited.',
    diagnostic_budget='Observed parent no_alignment performance motivated this same-family revision. Adopt exactly matching completed screen models; fit the new mandatory control and subsequent seeds. No adopted record counts as a new neural fit.')
edits = []
for node in ast.walk(prepare):
    if isinstance(node, ast.keyword) and node.arg in descriptions:
        value = node.value
        if not isinstance(value, ast.Constant) or value.lineno != value.end_lineno:
            raise ValueError('Unexpected protocol literal')
        edits.append((value.lineno, value.col_offset, value.end_col_offset, repr(descriptions[node.arg])))
lines = text.splitlines(keepends=True)
for line, start, end, replacement in sorted(edits, reverse=True):
    lines[line-1] = lines[line-1][:start] + replacement + lines[line-1][end:]
text = ''.join(lines)
ast.parse(text)
def bodies(value):
    rows = value.splitlines(keepends=True)
    return {node.name: ''.join(rows[node.lineno-1:node.end_lineno]) for node in ast.parse(value).body if isinstance(node, ast.FunctionDef)}
old, new = bodies(original), bodies(text)
shared = ['library', 'offset_arrays', 'setup', 'predict', 'cpu_copy', 'fit', 'score_arrays', 'run', 'select']
if any(old[name] != new[name] for name in shared):
    raise ValueError('Parent fitting/prediction behavior changed')
target.write_text(text)
report.write_text(json.dumps(dict(parent=str(source), parent_sha256=hashlib.sha256(original.encode()).hexdigest(),
    output=str(target), output_sha256=hashlib.sha256(text.encode()).hexdigest(), unchanged_function_bodies=shared,
    only_protocol_descriptions_and_source_registry_changed=True), indent=2) + '\n')
print('FUNCTIONAL_MORPHOLOGY_EXECUTOR_DERIVED', flush=True)
