import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.diff_cross_attn import Path2SubDifferCrossMHA, Drug2PathDifferCrossMHA
from src.modules.cross_attn import Path2SubCrossMHA, Drug2PathCrossMHA
from src.modules.ffn_layer import CelllineFFN, DrugFFN
from src.modules.mlp_head import DeepRegressorHead
    
#  DRUG RESPONSE MODEL
class DrugResponseModel(nn.Module):
    def __init__(self, pathway_gene_indices,
                 gene_ffn_output_dim, drug_ffn_output_dim,
                 cross_attn_dim, final_dim,
                 max_gene_slots, gene_input_dim, drug_input_dim=768, isDiffer=True,
                 gene_ffn_hidden_dim=1024, drug_ffn_hidden_dim=1024,
                 gene_ffn_dropout=0.5, drug_ffn_dropout=0.5,
                 num_heads=4, depth=2, mlp_dropout=0.3, final_dim_reduction_factor=2,
                 regressor_hidden_dims=[512, 512, 512, 256, 256],
                 regressor_dropout=0.15,
                 regressor_norm="batch",
                 regressor_act="gelu",
                 regressor_dropouts=None,
                 regressor_norms=None,
                 regressor_acts=None):
        super(DrugResponseModel, self).__init__()
        
        self.register_buffer('pathway_gene_indices', pathway_gene_indices)
        self.max_gene_slots = max_gene_slots
        self.gene_value_norm = nn.LayerNorm([94, gene_ffn_output_dim])
        
        # FFN Modules Initialization
        self.gene_ffn = CelllineFFN(input_dim=gene_input_dim, output_dim=gene_ffn_output_dim, 
                                       hidden_dim=gene_ffn_hidden_dim, dropout_rate=gene_ffn_dropout)
        self.drug_ffn = DrugFFN(input_dim=drug_input_dim, output_dim=drug_ffn_output_dim,
                                         hidden_dim=drug_ffn_hidden_dim, dropout_rate=drug_ffn_dropout)
        self.sub_ffn = DrugFFN(input_dim=drug_input_dim, output_dim=drug_ffn_output_dim,
                                         hidden_dim=drug_ffn_hidden_dim, dropout_rate=drug_ffn_dropout)
                
        # Cross-Attention Modules Initialization
        if isDiffer:
            self.Path2Sub_cross_attention = Path2SubDifferCrossMHA(
                pathway_embed_dim=gene_ffn_output_dim, 
                drug_embed_dim=drug_ffn_output_dim, 
                attention_dim=cross_attn_dim, 
                num_heads=num_heads, 
                depth=depth
            )
            self.Drug2Path_cross_attention = Drug2PathDifferCrossMHA(
                drug_embed_dim=drug_ffn_output_dim, 
                pathway_embed_dim=gene_ffn_output_dim, 
                attention_dim=cross_attn_dim, 
                num_heads=num_heads, 
                depth=depth
            )
        else:
            self.Path2Sub_cross_attention = Path2SubCrossMHA(
                pathway_embed_dim=gene_ffn_output_dim,
                drug_embed_dim=drug_ffn_output_dim,
                attention_dim=cross_attn_dim,
                num_heads=num_heads
            )
            self.Drug2Path_cross_attention = Drug2PathCrossMHA(
                drug_embed_dim=drug_ffn_output_dim,
                pathway_embed_dim=gene_ffn_output_dim,
                attention_dim=cross_attn_dim,
                num_heads=num_heads
            )
            
        self.head = DeepRegressorHead(
            input_dim=3 * cross_attn_dim,
            hidden_dims=regressor_hidden_dims,
            dropout=regressor_dropouts if regressor_dropouts is not None else regressor_dropout,
            norm=regressor_norms if regressor_norms is not None else regressor_norm,
            act=regressor_acts if regressor_acts is not None else regressor_act,
            residual_every=2,
            residual_proj=False,
            out_dim=1,
            last_dropout=False,
            final_norm=False
        )

    def forward(self, gene_embeddings_input, drug_embeddings_input, drug_substructure_embeddings, drug_multitoken_masks):
        B = gene_embeddings_input.size(0)
        P = self.pathway_gene_indices.size(0)     

        # Gene embeddings
        genes = gene_embeddings_input              
        indices = self.pathway_gene_indices.to(torch.long)     
        pad_mask = (indices == -1)
        safe_idx = indices.clone()
        safe_idx[pad_mask] = 0
        expanded_genes   = genes.unsqueeze(1).expand(-1, P, -1)         
        expanded_indices = safe_idx.unsqueeze(0).expand(B, -1, -1)      

        pathway_specific_genes = torch.gather(
            expanded_genes, 2, expanded_indices.to(expanded_genes.device)
        )  

        pathway_specific_genes = pathway_specific_genes.masked_fill(
            pad_mask.unsqueeze(0).to(pathway_specific_genes.device), 0.0
        )  

        # Gene FFN Module
        ffn_output = self.gene_ffn(pathway_specific_genes)
        gene_embedded_value = ffn_output.view(B, P, -1)
        pathway_embeddings = gene_embedded_value
        pathway_embeddings = self.gene_value_norm(gene_embedded_value)  

        # Multi-Token Drug FFN Module
        sub_embeddings = self.sub_ffn(drug_substructure_embeddings, drug_multitoken_masks)  
        drug_embeddings_input = drug_embeddings_input.unsqueeze(1)  
        drug_embeddings = self.drug_ffn(drug_embeddings_input)  

        # Path2Sub Cross-Attention
        path2sub_out, path2sub_weights = self.Path2Sub_cross_attention(
            query=pathway_embeddings,    
            key=sub_embeddings,        
            key_mask=drug_multitoken_masks  
        )

        # Drug2Path Cross-Attention
        drug2path_out, drug2path_weights = self.Drug2Path_cross_attention(
            query=drug_embeddings,       
            key=pathway_embeddings,      
        )
        
        # Mean pooling
        final_pathway_embedding = path2sub_out.mean(dim=1)
        final_drug_embedding = drug2path_out.mean(dim=1)
        final_sub_embedding = sub_embeddings.mean(dim=1)

        # Concatenate embeddings and MLP
        combined_embedding = torch.cat((final_pathway_embedding, final_drug_embedding, final_sub_embedding), dim=-1)
        y = self.head(combined_embedding) 

        return y, path2sub_weights, drug2path_weights, pathway_embeddings