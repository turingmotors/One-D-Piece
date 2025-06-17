# Imported from:
# https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/models/gpt.py
# https://github.com/FoundationVision/LlamaGen/blob/main/utils/drop_path.py

# Modified from:
#   VQGAN:    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/transformer/mingpt.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py  
#   nanoGPT:  https://github.com/karpathy/nanoGPT/blob/master/model.py
#   llama:    https://github.com/facebookresearch/llama/blob/main/llama/model.py
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
#   PixArt:   https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
from modeling.modules.base_model import BaseModel
from dataclasses import dataclass
from typing import Optional, List
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torch.nn import functional as F

# 一旦関係ない--------------------------------------------------------------------------------> 
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(torch.nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


# 関係ある-------------------------------------------------------------------------------->
# mask_tokenの位置から知っている場所を特定し，知っている場所(known)が空いたマスクを作る.
def bidirectional_attn_mask(maskable_mask: torch.Tensor, dtype):
    # maskを反転させて知っている部分を作成する
    known = ~maskable_mask
    # 見えている部分についてandをとってattention maskを作成する
    visible = known[:, :, None] & known[:, None, :]
    # visibleは見えている部分をTrueにしているので、maskable_maskの部分はFalseにする
    attn_mask = (~visible).to(dtype) * torch.finfo(dtype).min
    return attn_mask.unsqueeze(1)  # add head dim for multi-head attention

def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)
# -------------------------------------------------------------------------------------->

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    
    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0

    num_classes: int = 1000
    caption_dim: int = 2048
    class_dropout_prob: float = 0.1
    model_type: str = 'c2i'

    vocab_size: int = 16384
    cls_token_num: int = 1
    block_size: int = 256
    max_batch_size: int = 32
    max_seq_len: int = 2048


# ここは一旦放置できる--------------------------------------------------------------------------------->
#################################################################################
#                      Embedding Layers for Class Labels                        #
#################################################################################
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings


#################################################################################
#                      Embedding Layers for Text Feature                        #
#################################################################################
class CaptionEmbedder(nn.Module):
    """
    Embeds text caption into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, in_channels, hidden_size, uncond_prob, token_num=120):
        super().__init__()
        self.cap_proj = MLP(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size)
        self.register_buffer("uncond_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None], self.uncond_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        embeddings = self.cap_proj(caption)
        return embeddings


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

# ------------------------------------------------------------------------------------------------>
#################################################################################
#                                  GPT Model                                    #
#################################################################################
# 一旦ここも考えない
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


# FFNはいじらない
class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# 一旦無効にする(呼び出していないことを確認)
class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2], f"Check the shape: {input_pos.shape} and {k_val.shape}"
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor = None, 
        input_pos: Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None
    ):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        
        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None and input_pos is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        output = F.scaled_dot_product_attention(
            xq, keys, values, 
            attn_mask=mask, 
            # ここはFalseにする
            is_causal=False, # is_causal=False is for KV cache
            dropout_p=self.attn_dropout_p if self.training else 0)            
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, drop_path: float):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None):
        h = x + self.drop_path(self.attention(self.attention_norm(x), freqs_cis, start_pos, mask))
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


# 変えるところ--------------------------------------------------------------------------------->
class Transformer(BaseModel):
    """
    Changes from original LlamaGen implementation:
    * Change freqs_cis_2d to normal freqs_cis, because titok is one-d tokenizer
    * Removed loss calculation from forward, because it is not needed in titok training implementation. It is handled in the training script instead.
    * Removed support for t2i model type, because it is not supported in our experiments.
    """
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        # 普通のとーくんは0からvocab_size-1までのIDを持つのでmaskは一番後ろのやつ+1にする
        self.mask_id = self.vocab_size
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.num_classes = config.num_classes
        self.model_type = config.model_type
        self.cls_token_num = config.cls_token_num
        # clsembeddingを取得する
        if self.model_type == 'c2i':
            self.cls_embedding = LabelEmbedder(config.num_classes, config.dim, config.class_dropout_prob)
        else:
            raise Exception("please check model type")
        # TODO: tok_embeddingsを一個増やして初期化するためにどうするか考える
        # 一旦一つのtoken分のembeddingにする.tokenで引いたらembeddingになる
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.tok_dropout = nn.Dropout(config.token_dropout_p)

        # pretrained weightsを読み込んでいるかどうかのフラグ．Falseで初期化しておく
        self._pretrained_loaded = False

        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(TransformerBlock(config, dpr[layer_id]))

        # output layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        # 出力をvocab_sizeに広げて定義
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # MARK: Changed: use precompute_freqs_cis instead of precompute_freqs_cis_2d
        self.freqs_cis = precompute_freqs_cis(self.block_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num)
        
        # 重みの初期化.
        self.initialize_weights()


    def add_mask_token(self):
        """
        mask tokenを語彙に追加する
        """
        # もしターゲットが過去のvocab_sizeよりも大きい場合はmask tokenを追加する
        if self.tok_embeddings.num_embeddings < self.vocab_size + 1:
            self.resize_token_embeddings(self.vocab_size + 1)


    def resize_token_embeddings(self, new_num_tokens: int):
        """
        よしなにtoken_embeddingsを初期化する
        """
        old_num_tokens = self.tok_embeddings.num_embeddings
        
        if new_num_tokens <= old_num_tokens:
            return
            
        # Resize input embeddings
        old_embeddings = self.tok_embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embeddings.embedding_dim)
        new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)
        
        # Initialize new embeddings
        self._init_weights(new_embeddings)
        
        # Copy old embeddings
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]
        
        self.tok_embeddings = new_embeddings
        
        # Resize output layer
        old_lm_head = self.output
        new_lm_head = nn.Linear(old_lm_head.in_features, new_num_tokens, bias=old_lm_head.bias is not None)
        new_lm_head.to(old_lm_head.weight.device, dtype=old_lm_head.weight.dtype)
        
        # Initialize new output layer
        self._init_weights(new_lm_head)
        
        # Copy old weights
        new_lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]
        if old_lm_head.bias is not None:
            new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]
            
        self.output = new_lm_head
    

    def load_pretrained_ar(self, checkpoints_path: str):
        """
        Load pretrained weights and handle vocabulary expansion for mask token
        """
        if isinstance(checkpoints_path, str):
            sd = torch.load(checkpoints_path, map_location='cpu')
        else:
            sd = checkpoints_path
        
        # Load the pretrained state dict first
        miss, unexp = self.load_state_dict(sd, strict=False)
        
        current_vocab_size = self.tok_embeddings.num_embeddings
        target_vocab_size = self.vocab_size + 1  # original vocab + mask token
        
        if current_vocab_size < target_vocab_size:
            print(f"Resizing embeddings from {current_vocab_size} to {target_vocab_size}")
            self.resize_token_embeddings(target_vocab_size)
        
        self._pretrained_loaded = True
        print('loaded pretrained weights from', checkpoints_path)
        print(f'Missing keys: {miss}')
        print(f'Unexpected keys: {unexp}')

    def initialize_weights(self):        
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # Zero-out output layers:
        nn.init.constant_(self.output.weight, 0)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def forward(
        self, 
        idx: torch.Tensor, 
        cond_idx: torch.Tensor,  # cond_idx_or_embed
        input_pos:  Optional[torch.Tensor] = None, 
        mask: Optional[torch.Tensor] = None,
    ):
        if idx is not None and cond_idx is not None: # training or naive inference
            cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
            token_embeddings = self.tok_embeddings(idx)
            if (idx == self.mask_id).any(): # mask token
                # tok_embeddingsからmask_idの埋め込みを引いてくる
                mask_embed = self.tok_embeddings.weight[self.mask_id : self.mask_id + 1]
                token_embeddings = torch.where((idx == self.mask_id).unsqueeze(-1), mask_embed, token_embeddings)
            token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)
            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis.to(h.device)
        else:
            if cond_idx is not None: # prefill in inference
                token_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
            else: # decode_n_tokens(kv cache) in inference
                token_embeddings = self.tok_embeddings(idx)            
            bs = token_embeddings.shape[0]
            mask = self.causal_mask[:bs, None, input_pos]

            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis
        if self.training:
            freqs_cis = self.freqs_cis[:token_embeddings.shape[1]]
        else:
            freqs_cis = self.freqs_cis[input_pos]

        # transformer blocks
        for layer in self.layers:
            print(mask.shape)
            print(mask)
            # visualization
            plt.figure(figsize=(6, 6))

            h = layer(h, freqs_cis, input_pos, mask)
        
        # output layers
        h = self.norm(h)
        logits = self.output(h).float()
        
        logits = logits[:, self.cls_token_num - 1:].contiguous()
        return logits

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)
    
    @torch.no_grad()
    def generate(self, condition, guidance_scale=1.0, top_k=0, top_p=1.0, temperature=1.0, **sampling_kwargs):
        """
        generated_tokens = generator.generate(
            condition=labels,
            guidance_scale=guidance_scale,
            guidance_decay=guidance_decay,
            guidance_scale_pow=guidance_scale_pow,
            randomize_temperature=randomize_temperature,
            softmax_temperature_annealing=softmax_temperature_annealing,
            num_sample_steps=num_sample_steps
        )
        """
        # remove unuseable kwargs
        return generate(
            # TODO: hyperparameters should be passed in config
            self, 
            cond=condition,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
        )



# Modified from:
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch._dynamo.config
import torch._inductor.config
import copy
# torch._inductor.config.coordinate_descent_tuning = True
# torch._inductor.config.triton.unique_kernel_names = True
# torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future


### from https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html
def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample(logits, temperature: float=1.0, top_k: int=0, top_p: float=1.0, sample_logits=True):        
    logits = logits[:, -1, :] / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=-1)
    if sample_logits:
        idx = torch.multinomial(probs, num_samples=1)
    else:
        _, idx = torch.topk(probs, k=1, dim=-1)
    return idx, probs


def logits_to_probs(logits, temperature: float = 1.0, top_p: float=1.0, top_k: int = None, **kwargs):
    logits = logits / max(temperature, 1e-5)
    if top_k > 0 or top_p < 1.0:
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def prefill(model, cond_idx: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, **sampling_kwargs):
    if cfg_scale > 1.0:
        logits = model(None, cond_idx, input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0)
        logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
    else:
        logits = model(None, cond_idx, input_pos)

    return sample(logits, **sampling_kwargs)[0]


def decode_one_token(model, x: torch.Tensor, input_pos: torch.Tensor, cfg_scale: float, cfg_flag: bool, **sampling_kwargs):
    assert input_pos.shape[-1] == 1
    if cfg_scale > 1.0:
        x_combined = torch.cat([x, x])
        logits = model(x_combined, cond_idx=None, input_pos=input_pos)
        logits_combined = logits
        cond_logits, uncond_logits = torch.split(logits_combined, len(logits_combined) // 2, dim=0) 
        if cfg_flag:
            logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
        else:
            logits = cond_logits
    else:
        logits = model(x, cond_idx=None, input_pos=input_pos)
    return sample(logits, **sampling_kwargs)


def decode_n_tokens(
    model, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, 
    cfg_scale: float, cfg_interval: int,
    **sampling_kwargs):
    new_tokens, new_probs = [], []
    cfg_flag = True
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
            if cfg_interval > -1 and i > cfg_interval:
                cfg_flag = False
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, cfg_scale, cfg_flag, **sampling_kwargs
            )
            input_pos += 1
            new_tokens.append(next_token.clone())
            new_probs.append(next_prob.clone())
            cur_token = next_token.view(-1, 1)
    return new_tokens, new_probs


@torch.no_grad()
# diffusion style
def generate(
    self, cond: torch.Tensor, steps: int = 8, top_p: float = 0.9, top_k: int =0, temperature: float = 1.0
):
    B = cond.size(0)
    device = cond.device
    # fix
    ids = torch.full((B, self.block_size), self.mask_id, dtype=torch.long, device=device)
    known = torch.zeros_like(ids, dtype=torch.bool)

    for t in range(steps, 0, -1):
        maskable_latent = ~known
        print(f'step{t}: known ->', known)

        zero_cls        = torch.zeros_like(maskable_latent[:, :1])
        maskable_full   = torch.cat([zero_cls, maskable_latent], dim=1)

        attn = bidirectional_attn_mask(maskable_full, dtype=torch.float16)
        attn = attn.repeat_interleave(self.config.n_head, dim=1)

        save_attention_mask(attn, step_idx=t, head_dim=self.config.n_head)

        seq_len_full = self.cls_token_num + self.block_size
        full_pos = torch.arange(seq_len_full, device=device)

        logits = self(ids, cond, input_pos=full_pos, mask=attn)
        logits = logits / max(temperature, 1e-5)
        logits = top_k_top_p_filtering(logits, top_p=top_p, top_k=top_k)
        probs = F.softmax(logits, dim=-1)
        # ここまで普通のサンプル
        preds = torch.distributions.Categorical(probs).sample()
        # print(preds.shape)
        preds = preds[:, :-1]
        ids = torch.where(known, ids, preds)
        
        
        newly = (~known) & (torch.rand_like(ids.float()) < 1.0 / t)
        known = known | newly
    print('step0: known ->', known)
    # 以下マスクを可視化するためのコード
    maskable_latent = ~known
    zero_cls        = torch.zeros_like(maskable_latent[:, :1])
    maskable_full   = torch.cat([zero_cls, maskable_latent], dim=1)
    attn = bidirectional_attn_mask(maskable_full, dtype=torch.float16)
    attn = attn.repeat_interleave(self.config.n_head, dim=1)
    save_attention_mask(attn, step_idx=0, head_dim=self.config.n_head)
    return ids


def save_attention_mask(attn, step_idx, head_dim, save_dir="attn_viz"):
    """
    attn: torch.Tensor of shape (B, num_heads, L, L)
    """
    os.makedirs(save_dir, exist_ok=True)
    B, _, L, _ = attn.shape

    for b in range(B):
        for h in range(head_dim):
            attn_mask = attn[b, h].cpu().numpy()
            plt.figure(figsize=(6, 6))
            plt.imshow(attn_mask, cmap="gray", interpolation="nearest", vmin=-65504, vmax=0)
            plt.title(f"Step {step_idx} | Sample {b} | Head {h}")
            # plt.colorbar()
            plt.tight_layout()
            filename = os.path.join(save_dir, f"step{step_idx}_b{b}_h{h}.png")
            plt.savefig(filename)
            plt.close()
            if h == 0:
                break

#################################################################################
#                      Rotary Positional Embedding Functions                    #
#################################################################################
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py 
def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, cls_token_num=120):
    # print(seq_len, 'seq_len')
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs) # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1) # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache 


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2) # (bs, seq_len, n_head, head_dim//2, 2)
    # print(xshaped.shape, 'xshaped')
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2) # (1, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack([
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)



#################################################################################
#                                GPT Configs                                    #
#################################################################################
### text-conditional
def GPT_7B(config):
    return Transformer(ModelArgs(n_layer=32, n_head=32, dim=4096, **dict(config.model.generator.model_args))) # 6.6B

def GPT_3B(config):
    return Transformer(ModelArgs(n_layer=24, n_head=32, dim=3200, **dict(config.model.generator.model_args))) # 3.1B

def GPT_1B(config):
    return Transformer(ModelArgs(n_layer=22, n_head=32, dim=2048, **dict(config.model.generator.model_args))) # 1.2B

### class-conditional
def GPT_XXXL(config):
    return Transformer(ModelArgs(n_layer=48, n_head=40, dim=2560, **dict(config.model.generator.model_args))) # 3.9B

def GPT_XXL(config):
    return Transformer(ModelArgs(n_layer=48, n_head=24, dim=1536, **dict(config.model.generator.model_args))) # 1.4B

def GPT_XL(config):
    return Transformer(ModelArgs(n_layer=36, n_head=20, dim=1280, **dict(config.model.generator.model_args))) # 775M

def GPT_L(config):
    return Transformer(ModelArgs(n_layer=24, n_head=16, dim=1024, **dict(config.model.generator.model_args))) # 343M

def GPT_B(config):
    return Transformer(ModelArgs(n_layer=12, n_head=12, dim=768, **dict(config.model.generator.model_args))) # 111M
        

GPT_models = {
    'GPT-B': GPT_B, 'GPT-L': GPT_L, 'GPT-XL': GPT_XL, 'GPT-XXL': GPT_XXL, 'GPT-XXXL': GPT_XXXL,
    'GPT-1B': GPT_1B, 'GPT-3B': GPT_3B, 'GPT-7B': GPT_7B, 
}