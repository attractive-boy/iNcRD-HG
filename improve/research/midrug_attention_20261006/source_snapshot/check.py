"""Audit nonuniform path gates against explicit graph deletion."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from model import HERE,CONFIGS,make_adapter,AttentivePairStruct,load_fold,seed_all
import check_modules


def main():
    data=load_fold("MiDrug",0)
    seed_all(42);base=make_adapter(data,CONFIGS["degree05"],"cuda:0")
    seed_all(42);attn=make_adapter(data,CONFIGS["attn16_deg"],"cuda:0")
    base.model.eval();attn.model.eval()
    with torch.no_grad():
        p1,p2=base.predict("val"),attn.predict("val")
    np.testing.assert_allclose(p1,p2,rtol=1e-6,atol=1e-6)
    reports={}
    for name in ("attn16_t3","attn64_deg"):
        seed_all(42);adapter=make_adapter(data,CONFIGS[name],"cuda:0")
        with torch.no_grad():
            adapter.model.attention.qr.weight.normal_(std=.02)
            adapter.model.attention.qd.weight.normal_(std=.02)
        power=CONFIGS[name]["power"]
        check_modules.PairStruct=lambda D,p,c:AttentivePairStruct(D,p,c,power)
        checks=check_modules.check(adapter,count=12)
        adapter.model.train()
        loss=F.binary_cross_entropy_with_logits(adapter.logits("train"),adapter.y)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in adapter.model.parameters())
        assert adapter.model.attention.qr.weight.grad.abs().max()>0
        reports[name]=checks
    (HERE/"initial_checks.json").write_text(json.dumps(dict(status="passed",unit_gate_matches_control=True,
        nonuniform_gate_explicit_deletion=reports,finite_attention_gradients=True),indent=2))
    print("PASS unit-gate control equivalence, nonuniform-gate explicit deletion and finite gradients",flush=True)


if __name__=="__main__":main()
