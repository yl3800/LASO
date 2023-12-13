import math
import copy
from typing import Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F

class PositionEmbeddingSine1D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self,temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask, num_pos_feats):
        # x = tensor_list.tensors # [B, C, T]
        # mask = tensor_list.mask # [B, T]
        assert mask is not None
        x_embed = mask.cumsum(1, dtype=torch.float32)  # [B, T]
        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t  # [B, T, C]
        # n,c,t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        # pos = pos_x.permute(0, 2, 1)    # [B, C, T]
        return pos_x
    

class MultiheadAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 out_dim=None,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 q_project=True,
                 k_project=True,
                 v_project=True,
                 proj_after_att=True,
                 lan_dim=None):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.attn_drop = nn.Dropout(attn_drop)
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        out_dim = out_dim or dim

        assert self.dim % self.num_heads == 0

        self.q_lin = nn.Linear(in_features=lan_dim if lan_dim else self.dim, out_features=self.dim, bias=qkv_bias) if q_project else None
        self.k_lin = nn.Linear(in_features=self.dim, out_features=self.dim, bias=qkv_bias) if k_project else None
        self.v_lin = nn.Linear(in_features=self.dim, out_features=self.dim, bias=qkv_bias) if v_project else None
        
        self.out_lin = nn.Sequential(nn.Linear(dim, out_dim), nn.Dropout(proj_drop)) if proj_after_att else None

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, output_attentions=False):
        """
        Parameters
        ----------
        query: torch.tensor(bs, seq_length, dim)
        key: torch.tensor(bs, seq_length, dim)
        value: torch.tensor(bs, seq_length, dim)
        key_padding_mask: torch.tensor(bs, seq_length)
        Outputs
        -------
        weights: torch.tensor(bs, num_heads, seq_length, seq_length)
            Attention weights
        context: torch.tensor(bs, seq_length, dim)
            Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.num_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x):
            """ separate heads """
            return x.view(bs, -1, self.num_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """ group heads """
            return (
                x.transpose(1, 2).contiguous().view(bs, -1, self.num_heads * dim_per_head)
            )

        q = shape(self.q_lin(query)) if self.q_lin else shape(query) # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key)) if self.k_lin else shape(key)  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  if self.v_lin else shape(value) # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        attn = torch.matmul(q, k.transpose(2, 3)) * self.scale  # (bs, n_heads, q_length, k_length)
        if key_padding_mask is not None:
            key_padding_mask = ((~key_padding_mask).view(mask_reshp).expand_as(attn))  # (bs, n_heads, q_length, k_length)
            attn.masked_fill_(key_padding_mask, -float("inf"))  # (bs, n_heads, q_length, k_length)

        attn = nn.Softmax(dim=-1)(attn)  # (bs, n_heads, q_length, k_length)
        attn = self.attn_drop(attn)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if attn_mask is not None:
            attn = attn * attn_mask

        out = torch.matmul(attn, v)  # (bs, n_heads, q_length, dim_per_head)
        out = unshape(out)  # (bs, q_length, dim)
        if self.out_lin:
            out = self.out_lin(out)  # (bs, q_length, dim)
        return out



class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nheads, dim_feedforward=2048, dropout=0,
                 activation="relu", **kwargs):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nheads)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nheads, dim_feedforward=2048, dropout=0,
                 activation="relu", normalize_before=False, **kwargs):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nheads)
        self.multihead_attn = MultiheadAttention(d_model, nheads)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):

 
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask,
                                   output_attentions = True)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output
    


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                output_attentions = False):
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
            
        if output_attentions:
            return output,  c_att
        else:
            return output
        

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output