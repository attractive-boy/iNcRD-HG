"""Independent explicit edge-deletion checks for the generalized QMP port."""
import copy
import json
from types import SimpleNamespace
import numpy as np
import torch
from full_adapter import FullAdapter, OriginalQueryEncoder, structure_data, HERE, load
from engine import seed_all


def check(adapter, count=6):
    model=adapter.model.eval(); enc=model.encoder
    graph=adapter.graph; D=adapter.D
    rows=[]
    with torch.no_grad():
        original=enc(graph.x_dict, graph.edge_index_dict)
        zr,zd=enc.forward_dense(*adapter.dense)
        for node, actual in [('miRNA',zr),('drug',zd)]:
            torch.testing.assert_close(actual,original[node],atol=3e-4,rtol=2e-5)
        pos=graph['miRNA','MiDrug','drug'].edge_index.t()
        indices=np.linspace(0,len(pos)-1,count,dtype=int)
        for index in indices:
            pair=pos[index:index+1].t(); r,d=map(int,pair[:,0])
            q=adapter.ps.A[r,d].reshape(1)
            z=enc.forward_dense(*adapter.dense,*pair,q)
            f1=model(graph,pair,adapter.ps,(adapter.ps.qis(*pair)-adapter.mu)/adapter.sd,
                     adapter.ps.ease(*pair),adapter.dense)
            g2=graph.clone()
            for edge,first,second in [(('miRNA','MiDrug','drug'),r,d),(('drug','rev_MiDrug','miRNA'),d,r)]:
                idx=g2[edge].edge_index
                g2[edge].edge_index=idx[:,~((idx[0]==first)&(idx[1]==second))]
            d2=structure_data(g2)
            ps2=adapter.module.PairStruct(d2,g2['miRNA','MiDrug','drug'].edge_index.t())
            z2=enc(g2.x_dict,g2.edge_index_dict)
            torch.testing.assert_close(z[2][0],z2['miRNA'][r],atol=3e-4,rtol=2e-5)
            torch.testing.assert_close(z[3][0],z2['drug'][d],atol=3e-4,rtol=2e-5)
            pools1=adapter.ps.qncn(*pair,{'miRNA':z[0],'drug':z[1]},z[4])
            pools2=ps2.qncn(*pair,z2)
            for x,y in zip(pools1,pools2):torch.testing.assert_close(x,y,atol=3e-4,rtol=2e-5)
            torch.testing.assert_close(adapter.ps.qis(*pair),ps2.qis(*pair),atol=2e-5,rtol=2e-5)
            torch.testing.assert_close(adapter.ps.ease(*pair),ps2.ease(*pair),atol=3e-4,rtol=3e-4)
            old_graph,old_adj=enc.graph,enc.adj
            enc.graph=g2
            enc.adj=dict(old_adj)
            for edge in [('miRNA','MiDrug','drug'),('drug','rev_MiDrug','miRNA')]:
                idx=g2[edge].edge_index
                enc.adj[edge]=torch.sparse_coo_tensor(idx.flip(0),torch.ones(idx.shape[1],device=idx.device),old_adj[edge].shape).coalesce()
            try:
                f2=model(g2,pair,ps2,(ps2.qis(*pair)-adapter.mu)/adapter.sd,ps2.ease(*pair),
                         (d2.xr,d2.xd,d2.Sr1,d2.Sd1,ps2.A))
            finally:
                enc.graph,enc.adj=old_graph,old_adj
            torch.testing.assert_close(f1,f2,atol=3e-4,rtol=3e-4)
            rows.append({'m':r,'d':d,'logit_abs_error':float((f1-f2).abs().max()),
                'encoder_abs_error':max(float((z[2][0]-z2['miRNA'][r]).abs().max()),float((z[3][0]-z2['drug'][d]).abs().max()))})
    return rows


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--trained',action='store_true');args=p.parse_args()
    torch.cuda.set_device(0); reports={}
    for ds in ('MiDrug','MDR','MDS'):
        seed_all(42)
        a=SimpleNamespace(dataset=ds,device='cuda:0')
        data=load(ds,0);adapter=FullAdapter(data,a)
        if args.trained:
            adapter.model.load_state_dict(torch.load(HERE/f'results/incrd_m123_{ds}_f0_s42.pt',map_location='cuda:0'))
            adapter.model.eval()
            saved=np.load(HERE/f'results/incrd_m123_{ds}_f0_s42_predictions.npz')
            with torch.no_grad():
                for part in ('val','test'):
                    np.testing.assert_allclose(adapter.predict(part),saved[part+'_probability'],rtol=2e-5,atol=2e-6)
        rows=check(adapter)
        if not args.trained:
            opt=torch.optim.RMSprop(adapter.model.parameters(),lr=1e-4)
            adapter.model.train();loss=adapter.train_epoch(opt,1)
            assert np.isfinite(loss)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in adapter.model.parameters())
            # Rebuilding with permuted held-out labels changes no graph or pair channel.
            changed=dict(data)
            for part in ('val','test'):changed[part+'_y']=1-data[part+'_y']
            seed_all(42);other=FullAdapter(changed,SimpleNamespace(dataset=ds,device='cuda:0'))
            for e in adapter.graph.edge_types:torch.testing.assert_close(adapter.graph[e].edge_index,other.graph[e].edge_index)
            for part in ('train','val','test'):
                torch.testing.assert_close(adapter.qis[part],other.qis[part])
                torch.testing.assert_close(adapter.ease[part],other.ease[part])
            del other
        reports[ds]={'relations':len(adapter.graph.edge_types),'n_queries':len(rows),'checks':rows}
        print('PASS',ds,'explicit deletion, encoder/M2/M3/fused logits',flush=True)
        del adapter
    path=HERE/('module_checks_trained.json' if args.trained else 'module_checks_initial.json')
    path.write_text(json.dumps(reports,indent=2))

if __name__=='__main__':main()
