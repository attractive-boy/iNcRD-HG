
### MDR


#### MDR · PLMF Table-2 冷启动药物协议（11–20 条关联的药物，5 个负采样 seed）

| 方法 | 配置 | AUC | AUPR | macroAUC |
|---|---:|---:|---:|---:|
| ctxknn_warm | ctx,knn / warm / lr:0.3 | 0.8280 ± 0.0097 | 0.8175 ± 0.0215 | 0.8391 |
| ctx_warm | ctx / warm / lr:0.3 | 0.8232 ± 0.0138 | 0.8079 ± 0.0244 | 0.8310 |
| transfer_ctx | ctx / scorer=transfer / lr:0.3 | 0.8151 ± 0.0104 | 0.8142 ± 0.0217 | 0.8272 |
| transfer_ctxdiff | ctx,diff / drugcv / lr:0.3 | 0.8142 ± 0.0120 | 0.8111 ± 0.0259 | 0.8270 |
| knn_drugcv | knn / drugcv / lr:0.3 | 0.8137 ± 0.0262 | 0.8012 ± 0.0351 | 0.8338 |
| ctxdiff_warm | ctx,diff / warm / lr:0.3 | 0.8094 ± 0.0127 | 0.7995 ± 0.0270 | 0.8221 |
| ctxknn_lodo | ctx,knn / lodo / lr:0.3 | 0.8050 ± 0.0267 | 0.8038 ± 0.0374 | 0.8602 |
| score_knn_tan0_mean | knn / scorer=knn_tan0_mean / lr:0.3 | 0.8007 ± 0.0099 | 0.8124 ± 0.0142 | 0.8206 |
| ctxdiff_lodo | ctx,diff / lodo / lr:0.3 | 0.7996 ± 0.0109 | 0.8015 ± 0.0133 | 0.8349 |
| rankavg_degr_knn_attr | ctx,knn / scorer=rankavg:degree_r+knn_tan0_mean+drug_attribute_support_sum / lr:0.3 | 0.7943 ± 0.0115 | 0.8064 ± 0.0202 | 0.8129 |
| rankavg_degr_knn | ctx,knn / scorer=rankavg:degree_r+knn_tan0_mean / lr:0.3 | 0.7936 ± 0.0115 | 0.8055 ± 0.0211 | 0.8085 |
| ctxdiffknnmech_drugcv | ctx,diff,knn,mech / drugcv / lr:0.3 | 0.7935 ± 0.0261 | 0.7762 ± 0.0232 | 0.8306 |
| transfer10_ctxdiff | ctx,diff / scorer=transfer:10 / lr:0.3 | 0.7933 ± 0.0089 | 0.7949 ± 0.0105 | 0.8471 |
| score_drug_attr_support | ctx / scorer=drug_attribute_support_sum / lr:0.3 | 0.7922 ± 0.0126 | 0.8010 ± 0.0202 | 0.8169 |
| ctx_lodo | ctx / lodo / lr:0.3 | 0.7895 ± 0.0091 | 0.7954 ± 0.0183 | 0.8116 |
| ctxdiff_drugcv | ctx,diff / drugcv / lr:0.3 | 0.7890 ± 0.0146 | 0.7805 ± 0.0193 | 0.8269 |
| rankavg_degr_tan_cb | ctx,knn / scorer=rankavg:degree_r+knn_tan0_mean+knn_cb0_mean / lr:0.3 | 0.7887 ± 0.0117 | 0.8016 ± 0.0218 | 0.8046 |
| ctxdiffknn_lodo | ctx,diff,knn / lodo / lr:0.3 | 0.7858 ± 0.0256 | 0.7913 ± 0.0339 | 0.8492 |
| ctxdiffknn_drugcv | ctx,diff,knn / drugcv / lr:0.3 | 0.7821 ± 0.0250 | 0.7716 ± 0.0248 | 0.8276 |
| score_degree_r | ctx / scorer=degree_r / lr:0.3 | 0.7813 ± 0.0131 | 0.7837 ± 0.0229 | 0.7944 |
| score_knn_cb0_mean | knn / scorer=knn_cb0_mean / lr:0.3 | 0.7801 ± 0.0121 | 0.7954 ± 0.0222 | 0.7965 |
| ctx_drugcv | ctx / drugcv / lr:0.3 | 0.7778 ± 0.0089 | 0.7767 ± 0.0193 | 0.8041 |
| ctxdiffknnmech_drugcv_hgb | ctx,diff,knn,mech / drugcv / hgb | 0.7645 ± 0.0179 | 0.7616 ± 0.0269 | 0.8222 |
| ctxdiffcdiffalignknn_drugcv | ctx,diff,cdiff,align,knn / drugcv / lr:0.3 | 0.7461 ± 0.0306 | 0.7403 ± 0.0354 | 0.8146 |
| mech_drugcv | mech / drugcv / lr:0.3 | 0.7000 ± 0.0188 | 0.6842 ± 0.0113 | 0.7100 |
| score_collab_d | ctx / scorer=collab_d / lr:0.3 | 0.6685 ± 0.0119 | 0.6628 ± 0.0067 | 0.7170 |
| kron3 | ctx / scorer=kron:3.0 / lr:0.3 | 0.6517 ± 0.0154 | 0.6198 ± 0.0136 | 0.7057 |
| syntax_kron | ctx / scorer=kron:1.0 / lr:0.3 | 0.6463 | 0.6152 | 0.7215 |
| score_knn_tan5_mean | knn / scorer=knn_tan5_mean / lr:0.3 | 0.6352 ± 0.0094 | 0.6395 ± 0.0069 | 0.6601 |
| kron1 | ctx / scorer=kron:1.0 / lr:0.3 | 0.6319 ± 0.0121 | 0.6066 ± 0.0084 | 0.7003 |
| kron03 | ctx / scorer=kron:0.3 / lr:0.3 | 0.6088 ± 0.0127 | 0.5899 ± 0.0087 | 0.6796 |
| score_knn_tan0_rpc | knn / scorer=knn_tan0_rpc / lr:0.3 | 0.5661 ± 0.0095 | 0.5928 ± 0.0153 | 0.6358 |
| score_mech_b0_norm | mech / scorer=mech_b0_norm / lr:0.3 | 0.5033 ± 0.0024 | 0.5031 ± 0.0031 | 0.5028 |

#### MDR · frozen TabICLv2 in-context（同划分）

| 方法 | 配置 | AUC | AUPR | macroAUC |
|---|---:|---:|---:|---:|
| tab_ctx_warm | ctx / warm | 0.7874 ± 0.0105 | 0.7992 ± 0.0236 | 0.8090 |
| tab_ctxknn_warm | ctx,knn / warm | 0.7858 ± 0.0081 | 0.8010 ± 0.0187 | 0.8154 |
| tab_ctxdiff_warm | ctx,diff / warm | 0.7788 ± 0.0127 | 0.7900 ± 0.0244 | 0.7967 |

#### MDR · few-shot 冷启动药物（每个测试药物先给 k 条已知正例）

| 方法 | 配置 | AUC | AUPR | macroAUC |
|---|---:|---:|---:|---:|
| ctxdiff_warm_k1 | ctx,diff / warm / k=1 | 0.8198 ± 0.0206 | 0.7935 ± 0.0122 | 0.8424 |
| ctxdiff_warm_k3 | ctx,diff / warm / k=3 | 0.8514 ± 0.0243 | 0.8429 ± 0.0290 | 0.8563 |
| ctxdiff_warm_k5 | ctx,diff / warm / k=5 | 0.8686 ± 0.0194 | 0.8657 ± 0.0267 | 0.8736 |
| ctxdiff_warm_k10 | ctx,diff / warm / k=10 | 0.8764 ± 0.0302 | 0.8732 ± 0.0372 | 0.8901 |
| ctxdiffknn_warm_k1 | ctx,diff,knn / warm / k=1 | 0.8224 ± 0.0227 | 0.7987 ± 0.0149 | 0.8473 |
| ctxdiffknn_warm_k3 | ctx,diff,knn / warm / k=3 | 0.8529 ± 0.0250 | 0.8483 ± 0.0303 | 0.8585 |
| ctxdiffknn_warm_k5 | ctx,diff,knn / warm / k=5 | 0.8691 ± 0.0110 | 0.8659 ± 0.0177 | 0.8727 |
| ctxdiffknn_warm_k10 | ctx,diff,knn / warm / k=10 | 0.8756 ± 0.0249 | 0.8707 ± 0.0336 | 0.8987 |

#### MDR · few-shot（TabICLv2）

| 方法 | 配置 | AUC | AUPR | macroAUC |
|---|---:|---:|---:|---:|
| tab_ctxdiff_warm_k1 | TabICLv2 ctx,diff / k=1 | 0.7951 ± 0.0147 | 0.7898 ± 0.0060 | 0.8021 |
| tab_ctxdiff_warm_k3 | TabICLv2 ctx,diff / k=3 | 0.8509 ± 0.0210 | 0.8541 ± 0.0279 | 0.8538 |
| tab_ctxdiff_warm_k5 | TabICLv2 ctx,diff / k=5 | 0.8726 ± 0.0216 | 0.8729 ± 0.0278 | 0.8651 |

#### MDR · PLMF Table-2 冷启动 miRNA 协议（5–6 条关联的 miRNA；论文 PLMF-MDA：MDR 0.9115/0.8977，MDS 0.9190/0.9037）

| 方法 | 配置 | AUC | AUPR | macroAUC |
|---|---:|---:|---:|---:|
| mcold_ctx_warm | ctx / warm | 0.9327 ± 0.0049 | 0.9200 ± 0.0091 | 0.9298 |
| mcold_ctxdiff_warm | ctx,diff / warm | 0.9283 ± 0.0052 | 0.9141 ± 0.0092 | 0.9250 |
| mcold_score_degree_d | ctx / scorer=degree_d | 0.9242 ± 0.0051 | 0.9049 ± 0.0106 | 0.9228 |

#### MDR · 全药物 leave-drugs-out 5 折（按度分层，3 seeds）

| 方法 | 配置 | AUC(折均值) | pooled AUC | pooled AUPR | macroAUC |
|---|---:|---:|---:|---:|---:|
| ctx_warm | ctx / warm | 0.6979 ± 0.0375 | 0.6929 | 0.6855 | 0.8496 |
| score_degree_r | ctx / scorer=degree_r | 0.6925 ± 0.0415 | 0.6853 | 0.6788 | 0.8425 |
| ctx_drugcv | ctx / drugcv | 0.5968 ± 0.1258 | 0.5790 | 0.5858 | 0.8411 |
| score_knn_tan0_mean | knn / scorer=knn_tan0_mean | 0.6758 ± 0.0681 | 0.6643 | 0.6797 | 0.8385 |
| ctxdiff_drugcv | ctx,diff / drugcv | 0.6200 ± 0.1054 | 0.6079 | 0.6003 | 0.8360 |
| ctxdiffknnmech_drugcv | ctx,diff,knn,mech / drugcv | 0.6089 ± 0.1061 | 0.5866 | 0.5929 | 0.8314 |
| ctxdiffknn_drugcv | ctx,diff,knn / drugcv | 0.6119 ± 0.1155 | 0.5887 | 0.5952 | 0.8270 |

### MDS


#### MDS · PLMF Table-2 冷启动药物协议（11–20 条关联的药物，5 个负采样 seed）

| 方法 | 配置 | AUC | AUPR | macroAUC |
|---|---:|---:|---:|---:|
| transfer_ctxdiff | ctx,diff / scorer=transfer / lr:0.3 | 0.7924 ± 0.0259 | 0.7794 ± 0.0275 | 0.7997 |
| transfer_ctx | ctx / scorer=transfer / lr:0.3 | 0.7908 ± 0.0249 | 0.7791 ± 0.0246 | 0.7970 |
| ctxknn_warm | ctx,knn / warm / lr:0.3 | 0.7768 ± 0.0247 | 0.7811 ± 0.0244 | 0.7959 |
| ctx_warm | ctx / warm / lr:0.3 | 0.7718 ± 0.0253 | 0.7726 ± 0.0305 | 0.7861 |
| rankavg_degr_knn | ctx,knn / scorer=rankavg:degree_r+knn_tan0_mean / lr:0.3 | 0.7714 ± 0.0234 | 0.7804 ± 0.0257 | 0.7802 |
| rankavg_degr_tan_cb | ctx,knn / scorer=rankavg:degree_r+knn_tan0_mean+knn_cb0_mean / lr:0.3 | 0.7703 ± 0.0241 | 0.7799 ± 0.0269 | 0.7795 |
| rankavg_degr_knn_attr | ctx,knn / scorer=rankavg:degree_r+knn_tan0_mean+drug_attribute_support_sum / lr:0.3 | 0.7693 ± 0.0230 | 0.7789 ± 0.0255 | 0.7782 |
| score_knn_tan0_mean | knn / scorer=knn_tan0_mean / lr:0.3 | 0.7692 ± 0.0215 | 0.7802 ± 0.0233 | 0.7769 |
| score_degree_r | ctx / scorer=degree_r / lr:0.3 | 0.7686 ± 0.0244 | 0.7641 ± 0.0267 | 0.7791 |
| knn_drugcv | knn / drugcv / lr:0.3 | 0.7682 ± 0.0214 | 0.7549 ± 0.0230 | 0.7901 |
| ctxdiff_warm | ctx,diff / warm / lr:0.3 | 0.7682 ± 0.0280 | 0.7678 ± 0.0315 | 0.7804 |
| score_knn_cb0_mean | knn / scorer=knn_cb0_mean / lr:0.3 | 0.7672 ± 0.0246 | 0.7773 ± 0.0273 | 0.7774 |
| score_drug_attr_support | ctx / scorer=drug_attribute_support_sum / lr:0.3 | 0.7639 ± 0.0230 | 0.7742 ± 0.0253 | 0.7773 |
| transfer10_ctxdiff | ctx,diff / scorer=transfer:10 / lr:0.3 | 0.7551 ± 0.0204 | 0.7473 ± 0.0178 | 0.8029 |
| ctxknn_lodo | ctx,knn / lodo / lr:0.3 | 0.7485 ± 0.0260 | 0.7430 ± 0.0219 | 0.8006 |
| ctxdiffknnmech_drugcv_hgb | ctx,diff,knn,mech / drugcv / hgb | 0.7484 ± 0.0154 | 0.7307 ± 0.0177 | 0.7744 |
| ctxdiff_lodo | ctx,diff / lodo / lr:0.3 | 0.7481 ± 0.0226 | 0.7349 ± 0.0230 | 0.7922 |
| ctxdiff_drugcv | ctx,diff / drugcv / lr:0.3 | 0.7452 ± 0.0291 | 0.7391 ± 0.0323 | 0.7922 |
| ctxdiffknn_lodo | ctx,diff,knn / lodo / lr:0.3 | 0.7427 ± 0.0252 | 0.7362 ± 0.0229 | 0.7952 |
| ctx_drugcv | ctx / drugcv / lr:0.3 | 0.7403 ± 0.0315 | 0.7434 ± 0.0370 | 0.7812 |
| ctxdiffknn_drugcv | ctx,diff,knn / drugcv / lr:0.3 | 0.7352 ± 0.0329 | 0.7256 ± 0.0342 | 0.7939 |
| ctxdiffcdiffalignknn_drugcv | ctx,diff,cdiff,align,knn / drugcv / lr:0.3 | 0.7290 ± 0.0272 | 0.7108 ± 0.0355 | 0.8001 |
| ctx_lodo | ctx / lodo / lr:0.3 | 0.7280 ± 0.0281 | 0.7269 ± 0.0342 | 0.7714 |
| ctxdiffknnmech_drugcv | ctx,diff,knn,mech / drugcv / lr:0.3 | 0.7269 ± 0.0369 | 0.7162 ± 0.0351 | 0.7943 |
| mech_drugcv | mech / drugcv / lr:0.3 | 0.6704 ± 0.0267 | 0.6531 ± 0.0228 | 0.6817 |
| score_collab_d | ctx / scorer=collab_d / lr:0.3 | 0.6626 ± 0.0178 | 0.6525 ± 0.0150 | 0.6681 |
| score_knn_tan5_mean | knn / scorer=knn_tan5_mean / lr:0.3 | 0.6425 ± 0.0125 | 0.6341 ± 0.0127 | 0.6324 |
| kron3 | ctx / scorer=kron:3.0 / lr:0.3 | 0.6202 ± 0.0227 | 0.6055 ± 0.0181 | 0.6807 |
| kron1 | ctx / scorer=kron:1.0 / lr:0.3 | 0.6109 ± 0.0200 | 0.5976 ± 0.0141 | 0.6695 |
| kron03 | ctx / scorer=kron:0.3 / lr:0.3 | 0.6010 ± 0.0181 | 0.5910 ± 0.0129 | 0.6477 |
| score_knn_tan0_rpc | knn / scorer=knn_tan0_rpc / lr:0.3 | 0.5740 ± 0.0213 | 0.5843 ± 0.0165 | 0.5902 |
| score_mech_b0_norm | mech / scorer=mech_b0_norm / lr:0.3 | 0.5013 ± 0.0012 | 0.5013 ± 0.0012 | 0.5011 |

#### MDS · frozen TabICLv2 in-context（同划分）

| 方法 | 配置 | AUC | AUPR | macroAUC |
|---|---:|---:|---:|---:|
| tab_ctx_warm | ctx / warm | 0.7863 ± 0.0205 | 0.7812 ± 0.0289 | 0.7921 |
| tab_ctxknn_warm | ctx,knn / warm | 0.7817 ± 0.0251 | 0.7796 ± 0.0272 | 0.7940 |
| tab_ctxdiff_warm | ctx,diff / warm | 0.7719 ± 0.0280 | 0.7716 ± 0.0338 | 0.7852 |

#### MDS · few-shot 冷启动药物（每个测试药物先给 k 条已知正例）

| 方法 | 配置 | AUC | AUPR | macroAUC |
|---|---:|---:|---:|---:|
| ctxdiff_warm_k1 | ctx,diff / warm / k=1 | 0.7961 ± 0.0235 | 0.7773 ± 0.0240 | 0.8009 |
| ctxdiff_warm_k3 | ctx,diff / warm / k=3 | 0.8130 ± 0.0289 | 0.7932 ± 0.0315 | 0.8117 |
| ctxdiff_warm_k5 | ctx,diff / warm / k=5 | 0.8266 ± 0.0332 | 0.8125 ± 0.0364 | 0.8196 |
| ctxdiff_warm_k10 | ctx,diff / warm / k=10 | 0.8504 ± 0.0290 | 0.8409 ± 0.0355 | 0.8255 |
| ctxdiffknn_warm_k1 | ctx,diff,knn / warm / k=1 | 0.8015 ± 0.0289 | 0.7841 ± 0.0300 | 0.8104 |
| ctxdiffknn_warm_k3 | ctx,diff,knn / warm / k=3 | 0.8138 ± 0.0318 | 0.7948 ± 0.0354 | 0.8167 |
| ctxdiffknn_warm_k5 | ctx,diff,knn / warm / k=5 | 0.8297 ± 0.0332 | 0.8166 ± 0.0357 | 0.8264 |
| ctxdiffknn_warm_k10 | ctx,diff,knn / warm / k=10 | 0.8532 ± 0.0308 | 0.8450 ± 0.0334 | 0.8331 |

#### MDS · few-shot（TabICLv2）

| 方法 | 配置 | AUC | AUPR | macroAUC |
|---|---:|---:|---:|---:|
| tab_ctxdiff_warm_k1 | TabICLv2 ctx,diff / k=1 | 0.7856 ± 0.0167 | 0.7688 ± 0.0242 | 0.7898 |
| tab_ctxdiff_warm_k3 | TabICLv2 ctx,diff / k=3 | 0.8113 ± 0.0265 | 0.7954 ± 0.0253 | 0.8095 |
| tab_ctxdiff_warm_k5 | TabICLv2 ctx,diff / k=5 | 0.8257 ± 0.0288 | 0.8165 ± 0.0312 | 0.8232 |

#### MDS · PLMF Table-2 冷启动 miRNA 协议（5–6 条关联的 miRNA；论文 PLMF-MDA：MDR 0.9115/0.8977，MDS 0.9190/0.9037）

| 方法 | 配置 | AUC | AUPR | macroAUC |
|---|---:|---:|---:|---:|
| mcold_ctx_warm | ctx / warm | 0.9360 ± 0.0064 | 0.9235 ± 0.0108 | 0.9385 |
| mcold_ctxdiff_warm | ctx,diff / warm | 0.9354 ± 0.0069 | 0.9223 ± 0.0122 | 0.9373 |
| mcold_score_degree_d | ctx / scorer=degree_d | 0.9331 ± 0.0054 | 0.9197 ± 0.0093 | 0.9340 |

#### MDS · 全药物 leave-drugs-out 5 折（按度分层，3 seeds）

| 方法 | 配置 | AUC(折均值) | pooled AUC | pooled AUPR | macroAUC |
|---|---:|---:|---:|---:|---:|
| ctx_warm | ctx / warm | 0.7082 ± 0.0285 | 0.7038 | 0.6948 | 0.8242 |
| ctxdiff_drugcv | ctx,diff / drugcv | 0.6841 ± 0.0798 | 0.6737 | 0.6640 | 0.8228 |
| ctxdiffknnmech_drugcv | ctx,diff,knn,mech / drugcv | 0.6541 ± 0.0778 | 0.6342 | 0.6305 | 0.8179 |
| ctx_drugcv | ctx / drugcv | 0.6673 ± 0.0824 | 0.6635 | 0.6591 | 0.8176 |
| ctxdiffknn_drugcv | ctx,diff,knn / drugcv | 0.6501 ± 0.0765 | 0.6307 | 0.6278 | 0.8155 |
| score_degree_r | ctx / scorer=degree_r | 0.6861 ± 0.0327 | 0.6784 | 0.6760 | 0.8141 |
| score_knn_tan0_mean | knn / scorer=knn_tan0_mean | 0.6792 ± 0.0469 | 0.6666 | 0.6918 | 0.8103 |