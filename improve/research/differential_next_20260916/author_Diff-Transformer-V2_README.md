# Differential Transformer V2 (DIFF V2)

[Read the blog post here](https://spiky-homegrown-4cb.notion.site/Differential-Transformer-V2-2e7baa052def80ecaa93d4d67d125417)

The implementation is provided in `multihead_flashdiffv2.py`.

## TL;DR

We introduce **Differential Transformer V2** (DIFF V2), an improved version of [Differential Transformer](https://arxiv.org/abs/2410.05258) (DIFF V1). This revision focuses on inference efficiency, training stability for production-level LLMs, and architectural elegance.

### Key Improvements

1. **Faster Inference & No Need of Custom Attention Kernels**  
   Instead of forcing the attention parameter count to match the baseline Transformer (as in DIFF V1), we introduce additional parameters for $Q_2$. This design allows DIFF V2 to match the baseline Transformer’s decoding speed and directly use [FlashAttention](https://github.com/Dao-AILab/flash-attention) without custom kernels.
   
2. **Improved Training Stability**  
   We remove the per-head RMSNorm after differential attention. We find the per-head RMSNorm can lead to instability in later stages of large-scale pretraining of LLM.

3. **Simpler Parameterization & Initialization**  
   We replace the globally shared $\lambda$ with a token-specific, head-wise projected $\lambda$. This eliminates the exponential re-parameterization and initialization complexity of $\lambda$ in V1.

## Implementation Details

### Pseudocode

In the script, `h` represents number of query heads, `h_kv`  represents number of key-value heads, and `d` means head dimension. The $\lambda$ in DIFF V2 is projected from $X$ for each token each head.

(For simplicity, we omit the batch dimension and assume that both the input and output of the following `flash_attn_func` are three-dimensional tensors `(tokens, heads, head dimension)`. Heads belonging to the same GQA group are arranged contiguously in the output)

```python
def DiffAttnV2(
	q, k, v, lam
):
   """
   q:   (N, 2h, d)
   k:   (N, h_kv, d)
   v:   (N, h_kv, d)
   lam: (N, h, 1)
   """

   attn = flash_attn_func(q, k, v)
   attn1, attn2 = (attn[:, 0::2], 
                     attn[:, 1::2])

   lam_val = sigmoid(lam)
   attn = attn1 - lam_val * attn2
   return attn
```

### Note

DIFF V2 subtracts two heads that are **in the same GQA group, which means they share the same key and value**.

```python
# Subtraction of two heads that are **not** in the same GQA group
# ❌ Wrong Implementation of DIFF V2!
...
attn = flash_attn_func(q, k, v)
nh = attn.size(1)
attn1, attn2 = (attn[:, :nh//2], 
		          attn[:, nh//2:])
# similarly, also wrong implementation:
# attn1, attn2 = attn.chunk(2, dim=1)
...
```

```python
# DIFF V2: Subtraction of two heads that are **in** the same GQA group
# ✅ Correct Implementation of DIFF V2
...
attn = flash_attn_func(q, k, v)

attn1, attn2 = (attn[:, 0::2], 
		          attn[:, 1::2])
...
```
