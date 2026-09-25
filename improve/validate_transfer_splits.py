"""Fixed-recipe sensitivity validation on three new development-pool resplits.

These are new pair partitions, not new biological data. Hyperparameters come
from historical inner selection and are never selected on these outcomes.
Graph preparation and GNN/TabM training run in the historical PyG environment;
TabICLv2 inference runs in the isolated TabICL environment.
"""
import argparse
from datetime import datetime, timezone
import gc
from pathlib import Path
import pickle
import shutil
import time

import numpy as np

from improve import tabular_foundation as common

ROOT = Path(__file__).resolve().parent / "results"
DEFAULT_OUT = ROOT / "transfer_splits_20260914"
GRAPH = ROOT / "backbone_replacement_20260910"
TABM = ROOT / "graph_tabm_20260910"
ICL = ROOT / "tabicl_20260912"
CONTEXT = ROOT / "tabicl_pathway_context_20260913"
TASKS = ("LncDrug", "MiDrug")
SPLITS = (42, 2026, 31415)
SEED = 20260914


def protocol(out):
    p = common.check_seal(common.read_json(Path(out) / "protocol.json"))
    for name, sha in p["source_sha256"].items():
        if common.digest(Path(__file__).with_name(name)) != sha:
            raise ValueError("Frozen implementation changed: " + name)
    for path, sha in p["input_sha256"].items():
        if common.digest(path) != sha:
            raise ValueError("Frozen input changed: " + path)
    return p


def prepare(out, smoke=False):
    import torch
    from sklearn.model_selection import train_test_split
    from improve.graph_tabm_modules import PairFeatureEncoder, crossfit_features, fit_transform
    from improve.tabicl_graph_context_features import build_views
    out = Path(out).resolve()
    if out.exists():
        raise ValueError("Choose a new output directory")
    gp = common.read_json(GRAPH / "protocol.json")
    mp = common.read_json(TABM / "protocol.json")
    ip = common.read_json(ICL / "protocol.json")
    cp = common.read_json(CONTEXT / "protocol.json")
    gf = common.read_json(GRAPH / "REFIT.json")["chosen"]
    mf = common.read_json(TABM / "REFIT.json")["chosen"]
    sources = ("validate_transfer_splits.py", "graph_tabm_modules.py", "graph_tabm.py",
               "backbone_replacement.py", "tabicl_graph_context_features.py",
               "tabicl_context_program_modules.py", "tabicl_pathway_modules.py",
               "network_feature_ablation.py", "tabular_foundation.py")
    inputs = [GRAPH / "protocol.json", GRAPH / "REFIT.json", TABM / "REFIT.json",
              ICL / "protocol.json", CONTEXT / "protocol.json",
              Path(cp["inputs_dir"]) / "INPUTS.json", Path(ip["checkpoint"])]
    for task in TASKS:
        inputs += [GRAPH / "inputs" / task / name for name in ("outer.npz", "train_graph.pth")]
    p = common.seal(dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(), smoke=smoke,
        tasks=list(TASKS), split_seeds=[42] if smoke else list(SPLITS), model_seed=SEED,
        holdout_fraction=.2, arms=["gnn", "tabm", "tabicl", "graph_context", "tiered"],
        selection="None. All five recipes and historical fixed refit epochs evaluated and reported.",
        gnn={task:gf[task]["base"] for task in TASKS},
        tabm={task:mf[task]["tabm"] for task in TASKS},
        graph_training=gp, tabm_training=mp, icl_protocol=ip,
        graph_recipes=cp["graph_recipes"], input_configs=cp["input_configs"],
        evidence_dir=cp["inputs_dir"], members=2 if smoke else 32,
        source_sha256={name:common.digest(Path(__file__).with_name(name)) for name in sources},
        input_sha256={str(path):common.digest(path) for path in inputs},
        graph_policy="Fixed historical PCA/circRNA side information; target adjacency rebuilt using only split fitting positives; feature fold query positives absent in both directions.",
        gate="MiDrug mean AUC and AP exceed GNN, each in >=2/3 new splits; LncDrug mean AUC/AP and each split do not decrease more than 0.001 absolute. This is a descriptive engineering gate, not a statistical noninferiority test.",
        limitations=["New partitions of previously used training labels, not independent biological data or a new external test.",
                     "Three overlapping resplits are not independent datasets. One fixed new model seed per split; historical three-seed results reported separately.",
                     "Training and synthetic pretraining budgets differ between model families.",
                     "Other-task side associations retain the historical single-task transductive policy.",
                     "Global association ranking does not establish functional mechanism or causal drug response."],
        original_test_access=False, historical_outer_labels_access=False))
    out.mkdir(parents=True)
    common.write_json(out / "protocol.json", p)
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    for name in sources:
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    for task in TASKS:
        directory = GRAPH / "inputs" / task
        graph = torch.load(directory / "train_graph.pth", map_location="cpu", weights_only=False)
        with np.load(directory / "outer.npz", allow_pickle=False) as z:
            # Only previously available FITTING labels are deserialized.
            pairs, labels = z["pairs_train"].copy(), z["y_train"].copy()
            for nt in graph.node_types:
                key = ("pca_x_" if nt in ("lncRNA", "miRNA") else "x_") + nt
                graph[nt].x = torch.from_numpy(z[key].copy())
        assert len(set(map(tuple, pairs.T))) == len(labels)
        rna = "lncRNA" if task == "LncDrug" else "miRNA"
        target, reverse = (rna, task, "drug"), ("drug", "rev_" + task, rna)
        for split in p["split_seeds"]:
            fit, query = train_test_split(np.arange(len(labels)), test_size=.2,
                                         random_state=split, stratify=labels)
            fit, query = np.sort(fit), np.sort(query)
            if smoke:
                fit = fit[common.balanced_subset(labels[fit], 160, split)]
                query = query[common.balanced_subset(labels[query], 40, split)]
            a = dict(pairs_train=pairs[:,fit], y_train=labels[fit],
                     pairs_val=pairs[:,query], y_val=labels[query])
            g = graph.clone()
            support = a["pairs_train"][:, a["y_train"] == 1]
            g[target].edge_index = torch.from_numpy(support.copy())
            g[reverse].edge_index = g[target].edge_index.flip(0).clone()
            g[target].edge_label_index = torch.from_numpy(a["pairs_train"])
            g[target].edge_label = torch.from_numpy(a["y_train"])
            # Historical graph labels outside the target store are not used.
            for key in g.edge_types:
                if key != target:
                    for attr in ("edge_label", "edge_label_index"):
                        if attr in g[key]:
                            del g[key][attr]
            forbidden = set(map(tuple, a["pairs_val"].T))
            assert not forbidden.intersection(map(tuple, support.T))
            assert torch.equal(g[target].edge_index.flip(0), g[reverse].edge_index)
            raw, raw_val, names, folds = crossfit_features(
                g, task, a["pairs_train"], a["y_train"], a["pairs_val"], 5, 20260910)
            views, view_names, checks, error = build_views(
                PairFeatureEncoder(g,task), a["pairs_train"], a["y_train"],
                folds, a["pairs_val"], raw, raw_val)
            assert names == view_names and len(names) == 36
            x, xv, transform = fit_transform(raw, raw_val)
            a.update(raw_train=raw, raw_val=raw_val, raw_query_views=views,
                     x_train=x, x_val=xv, fold_ids=folds,
                     x_query_views=transform.transform(views.reshape(-1,36)).astype(np.float32).reshape(views.shape))
            dest = out / "inputs" / task / str(split)
            dest.mkdir(parents=True)
            np.savez_compressed(dest / "arrays.npz", **a)
            torch.save(g, dest / "graph.pt")
            with (dest / "transform.pkl").open("wb") as stream:
                pickle.dump(transform, stream, protocol=4)
            common.write_json(dest / "manifest.json", dict(
                protocol_sha256=p["sha256"], names=names, fitting_pool_rows=fit.tolist(),
                holdout_pool_rows=query.tolist(), target_overlap=0, fold_audits=checks,
                full_view_reconstruction_error=error,
                files={name:common.digest(dest/name) for name in ("arrays.npz","graph.pt","transform.pkl")}))
            print("PREPARED",task,split,len(fit),len(query),flush=True)


def arrays(out,p,task,split):
    dest = Path(out) / "inputs" / task / str(split)
    m = common.read_json(dest / "manifest.json")
    assert m["protocol_sha256"] == p["sha256"]
    for name, sha in m["files"].items():
        assert common.digest(dest / name) == sha
    with np.load(dest / "arrays.npz", allow_pickle=False) as z:
        a = {name:z[name].copy() for name in z.files}
    common.validate_arrays(a)
    a["feature_manifest_sha256"] = common.digest(dest / "manifest.json")
    return a, dest


def done(dest,p):
    if not (dest / "result.json").exists():
        return False
    r = common.read_json(dest / "result.json")
    assert r["protocol_sha256"] == p["sha256"]
    for name, sha in r["files"].items():
        assert common.digest(dest / name) == sha
    return True


def save_result(dest,p,a,task,split,arm,score,details):
    np.savez_compressed(dest / "scores.npz", pred=score,y=a["y_val"],pairs=a["pairs_val"])
    common.write_json(dest / "result.json",dict(
        task=task,split=split,arm=arm,seed=p["model_seed"],protocol_sha256=p["sha256"],
        **common.metrics(a["y_val"],score,a["pairs_val"]),**details,
        files={str(f.relative_to(dest)):common.digest(f) for f in dest.rglob("*") if f.is_file()}))


def train(out,task,device):
    import torch
    from improve import backbone_replacement as gnn
    from improve import graph_tabm as tabm
    out=Path(out); p=protocol(out)
    for split in p["split_seeds"]:
        a, source=arrays(out,p,task,split)
        graph=torch.load(source/"graph.pt",map_location="cpu",weights_only=False)
        weights={key:torch.ones(edge.shape[1]) for key,edge in graph.edge_index_dict.items()}
        for arm in ("gnn","tabm"):
            dest=out/"runs"/task/str(split)/arm
            if done(dest,p):
                continue
            dest.mkdir(parents=True,exist_ok=True)
            recipe=p[arm][task]; epochs=2 if p["smoke"] else recipe["epochs"]
            cfg=recipe["config"]
            if arm=="gnn":
                row,predictions,trace,state=gnn.train(a,graph,weights,task,"base",cfg,p["model_seed"],
                    "outer",epochs,p["graph_training"],device)
                score=predictions["val"]
                torch.save(state,dest/"model.pt")
                model,g,w=gnn.make_model(graph,weights,task,"base",cfg,p["model_seed"],device)
                model.load_state_dict(torch.load(dest/"model.pt",map_location=device,weights_only=False))
                replay=gnn.predict(model,g,w,a["pairs_val"],None,None,p["graph_training"]["batch_size"])
            else:
                row,score,trace,state=tabm.train(a,task,"tabm",cfg,p["model_seed"],
                    "outer",epochs,p["tabm_training"],device)
                torch.save(state,dest/"model.pt")
                model=tabm.model_for("tabm",36,cfg,p["model_seed"],device)
                model.load_state_dict(torch.load(dest/"model.pt",map_location=device,weights_only=False))
                replay=tabm.predict(model,torch.from_numpy(a["x_val"]).to(device),p["tabm_training"]["batch_size"])
            np.testing.assert_allclose(replay,score,atol=5e-5,rtol=5e-5)
            common.write_json(dest/"trace.json",trace)
            save_result(dest,p,a,task,split,arm,score,dict(epochs=epochs,config=cfg,
                replay_max_abs=float(np.max(np.abs(replay-score))),seconds=row["elapsed_seconds"]))
            print("VERIFIED",task,split,arm,common.metrics(a["y_val"],score,a["pairs_val"]),flush=True)
            del model
            gc.collect(); torch.cuda.empty_cache()


def infer(out,task,device):
    import torch
    from tabicl import TabICLClassifier
    from improve.tabicl_context_program_modules import ContextPrograms
    out=Path(out); p=protocol(out)
    evidence=ContextPrograms.load(p["evidence_dir"])
    torch.cuda.set_device(device)
    for split in p["split_seeds"]:
        a,_=arrays(out,p,task,split)
        prefix="raw" if p["input_configs"][task]["input_view"]=="raw" else "x"
        graph_arm=p["graph_recipes"][task]["arm"]
        assert graph_arm in ("query_bag","fold_match")
        graph_scores={}
        for variant in ("graph","tiered"):
            component_scores={}; start=time.monotonic(); errors=[]
            dest=out/"runs"/task/str(split)/("graph_context" if variant=="graph" else "tiered")
            if done(dest,p):
                with np.load(dest/"scores.npz") as z:
                    graph_scores[variant]=z["pred"].copy()
                continue
            dest.mkdir(parents=True,exist_ok=True)
            extra_train=evidence.transform(task,a["pairs_train"],"tiered") if variant=="tiered" else None
            extra_val=evidence.transform(task,a["pairs_val"],"tiered") if variant=="tiered" else None
            kinds=["full"]+(["fold"+str(i) for i in range(5)] if graph_arm=="fold_match" else [])
            for kind in kinds:
                full=kind=="full"
                keep=np.arange(len(a["y_train"])) if full else np.flatnonzero(a["fold_ids"]==int(kind[-1]))
                x=a[prefix+"_train"][keep]
                q=(np.concatenate([a[prefix+"_val"][None],a[prefix+"_query_views"]])
                   if full and graph_arm=="query_bag" else a[prefix+"_val"][None] if full
                   else a[prefix+"_query_views"][int(kind[-1]):int(kind[-1])+1])
                if variant=="tiered":
                    x=np.concatenate([x,extra_train[keep]],axis=1).astype(np.float32)
                    q=np.concatenate([q,np.broadcast_to(extra_val,(len(q),*extra_val.shape))],axis=2).astype(np.float32)
                model=common.fit_context(common.estimator(dict(n_estimators=p["members"]),p["model_seed"],p["icl_protocol"],device),x,a["y_train"][keep])
                parameter_sha=common.parameter_digest(model)
                pred=common.query_scores(model,q.reshape(-1,x.shape[1])).reshape(q.shape[:2])
                model.save(dest/(kind+".pkl"),save_model_weights=False,save_training_data=True,save_kv_cache=False)
                del model; gc.collect(); torch.cuda.empty_cache()
                model=TabICLClassifier.load(dest/(kind+".pkl"),device=device)
                assert common.parameter_digest(model)==parameter_sha
                replay=common.query_scores(model,q.reshape(-1,x.shape[1])).reshape(q.shape[:2])
                np.testing.assert_allclose(replay,pred,atol=1e-6,rtol=1e-6)
                errors.append(float(np.max(np.abs(pred-replay))))
                component_scores[kind]=pred
                del model; gc.collect(); torch.cuda.empty_cache()
            original=component_scores["full"][0]
            component=(component_scores["full"][1:].mean(0) if graph_arm=="query_bag"
                       else np.stack([component_scores["fold"+str(i)][0] for i in range(5)]).mean(0))
            score=.5*original+.5*component
            if variant=="graph":
                base_dest=out/"runs"/task/str(split)/"tabicl"
                base_dest.mkdir(exist_ok=True)
                save_result(base_dest,p,a,task,split,"tabicl",original,dict(
                    replay_max_abs=max(errors),parameters_updated=False,shared_model="../graph_context/full.pkl"))
                graph_scores[variant]=score
            else:
                score=.5*graph_scores["graph"]+.5*score
            np.savez_compressed(dest/"components.npz",**component_scores)
            save_result(dest,p,a,task,split,"graph_context" if variant=="graph" else "tiered",score,
                dict(replay_max_abs=max(errors),parameters_updated=False,parameter_sha256=parameter_sha,
                     seconds=time.monotonic()-start,members=p["members"]))
            print("VERIFIED",task,split,variant,common.metrics(a["y_val"],score,a["pairs_val"]),flush=True)


def report(out):
    import pandas as pd
    out=Path(out); p=protocol(out); rows=[]
    for task in p["tasks"]:
        for split in p["split_seeds"]:
            baseline=None
            for arm in p["arms"]:
                dest=out/"runs"/task/str(split)/arm
                assert done(dest,p), "Missing registered result: "+str(dest)
                r=common.read_json(dest/"result.json")
                with np.load(dest/"scores.npz") as z:
                    actual=common.metrics(z["y"],z["pred"],z["pairs"])
                    ids=z["pairs"].copy(); y=z["y"].copy()
                if baseline is None:
                    baseline=(ids,y)
                np.testing.assert_array_equal(ids,baseline[0]); np.testing.assert_array_equal(y,baseline[1])
                for name,value in actual.items():
                    if value is not None:
                        assert abs(value-r[name])<1e-12
                rows.append({k:v for k,v in r.items() if not isinstance(v,(dict,list))})
    df=pd.DataFrame(rows); df.to_csv(out/"metrics.csv",index=False)
    summary=df.groupby(["task","arm"])[["auc","ap","macro_drug_ap"]].agg(["mean","std"])
    summary.to_csv(out/"summary.csv")
    contrasts=[]
    for arm,control in (("tabm","gnn"),("tabicl","gnn"),("graph_context","gnn"),("tiered","gnn"),
                        ("tabicl","tabm"),("graph_context","tabicl"),("tiered","graph_context")):
        for task in p["tasks"]:
            a=df[(df.task==task)&(df.arm==arm)].set_index("split")
            b=df[(df.task==task)&(df.arm==control)].set_index("split")
            row=dict(task=task,arm=arm,control=control)
            for metric in ("auc","ap","macro_drug_ap"):
                d=a[metric]-b[metric]
                row.update({metric+"_delta":float(d.mean()),metric+"_wins":int((d>0).sum()),
                            metric+"_min_delta":float(d.min()),metric+"_paired_sd":float(d.std(ddof=1))})
            contrasts.append(row)
    pd.DataFrame(contrasts).to_csv(out/"contrasts.csv",index=False)
    gates={}
    for arm in p["arms"][1:]:
        mi=next(r for r in contrasts if r["arm"]==arm and r["control"]=="gnn" and r["task"]=="MiDrug")
        lnc=next(r for r in contrasts if r["arm"]==arm and r["control"]=="gnn" and r["task"]=="LncDrug")
        gates[arm]=(not p["smoke"] and all(mi[m+"_delta"]>0 and mi[m+"_wins"]>=2
                    and lnc[m+"_delta"]>=-.001 and lnc[m+"_min_delta"]>=-.001 for m in ("auc","ap")))
    common.write_json(out/"COMPLETION.json",dict(protocol_sha256=p["sha256"],results=len(rows),
        smoke=p["smoke"],gates_vs_gnn=gates,all_metrics_recomputed=True,all_saved_models_replayed=True,
        max_replay_error=float(df.replay_max_abs.max()),new_external_test=False,
        counts={"partitions_per_task":len(p["split_seeds"]),"models_seeds_per_partition":1},
        artifacts={name:common.digest(out/name) for name in ("metrics.csv","summary.csv","contrasts.csv")}))
    print(summary.to_string()); print(pd.DataFrame(contrasts).to_string(index=False)); print(gates)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","train","infer","report"))
    parser.add_argument("--out",default=str(DEFAULT_OUT))
    parser.add_argument("--task",choices=TASKS)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    if args.command=="prepare": prepare(args.out,args.smoke)
    elif args.command=="train": train(args.out,args.task,args.device)
    elif args.command=="infer": infer(args.out,args.task,args.device)
    else: report(args.out)


if __name__=="__main__":
    main()
