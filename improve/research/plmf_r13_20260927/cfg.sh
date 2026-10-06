BASE="--enc mask --exact --qis --qis_ext --qncn --ncn_rel co,knn,dknn,seed7,seed6,pre,fam --rel seed7,seed6,pre,fam --ease 10,50 --ease_side kmer"
FIN="$BASE --val_graph --hardneg 2 --hardneg_pool 8 --hardneg_mode both --hardneg_loss bpr --unl 1 --unl_w 1 --ppr --ppr_norm"
