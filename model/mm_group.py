import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from model.attention import MultiheadAttention
# from attention import MultiheadAttention


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self,
                 in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    

class MLPMixerLayer(nn.Module):
    def __init__(self,
                 num_patches,
                 embed_dims,
                 patch_expansion,
                 channel_expansion,
                 drop_out,
                 **kwargs):

        super().__init__()

        patch_mix_dims = int(patch_expansion * embed_dims) # 16
        channel_mix_dims = int(channel_expansion * embed_dims) # 128

        self.patch_mixer = nn.Sequential(
            nn.Linear(num_patches, patch_mix_dims, bias=False), # try here
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(patch_mix_dims, num_patches, bias=False),
            nn.Dropout(drop_out)
        )

        self.channel_mixer = nn.Sequential(
            nn.Linear(embed_dims, channel_mix_dims),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(channel_mix_dims, embed_dims),
            nn.Dropout(drop_out)
        )

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)

    def forward(self, x):
        # x_mask = (x.sum(-1)!=0).to(x.dtype)
        x = x + self.patch_mixer(self.norm1(x).transpose(1,2)).transpose(1,2)
        x = x + self.channel_mixer(self.norm2(x))
        # x *= x_mask
        return x


class MLPMixer(nn.Module):
    def __init__(self,
                 num_patches,
                 embed_dims,
                 patch_expansion=0.5,
                 channel_expansion=4.0,
                 depth=1,
                 drop_out=0.,
                #  init_cfg=None,
                 **kwargs):
        super().__init__()
        layers = [
            MLPMixerLayer(num_patches, embed_dims, patch_expansion, channel_expansion, drop_out)
            for _ in range(depth)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    

class FullAttnCatBlock(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 ffn_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 key_is_query=False,
                 value_is_key=False,
                 q_project=True,
                 with_cp=False,
                 **kwargs):
        super().__init__()
        self.with_cp = with_cp

        self.norm_query = nn.LayerNorm(embed_dims)

        if not key_is_query:
            self.norm_key = nn.LayerNorm(embed_dims)
        else:
            self.norm_key = None
        self.key_is_query = key_is_query

        if not value_is_key:
            self.norm_value = nn.LayerNorm(embed_dims)
        else:
            self.norm_value = None
        self.value_is_key = value_is_key

        self.attn = MultiheadAttention(
            embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            q_project=q_project)

        self.ffn = Mlp(in_features=embed_dims, hidden_features=int(embed_dims * ffn_ratio), drop=drop)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.drop = nn.Dropout(drop)
        # self.proj = nn.Linear(embed_dims * 2, embed_dims, bias=True)

    def forward(self, query, key, value, key_padding_mask=None):
        def _inner_forward(query, key, value, key_padding_mask):
            q = self.norm_query(query)
            k = q if self.key_is_query else self.norm_key(key)
            v = k if self.value_is_key else self.norm_value(value)

            x = self.attn(q, k, v, key_padding_mask) + self.drop(query)
            # x = self.proj(x)
            x = self.ffn(self.norm2(x)) + x
            return x

        if self.with_cp:
            return cp.checkpoint(_inner_forward, query, key, value, key_padding_mask)
        else:
            return _inner_forward(query, key, value, key_padding_mask)
        
    
class LightGroupAttnBlock(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 ffn_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 key_is_query=False,
                 value_is_key=False,
                 with_cp=False,
                 lan_dim = 768):
        super().__init__()
        self.drop = nn.Dropout(drop)
        self.with_cp = with_cp

        self.norm_query = nn.GELU()

        if not key_is_query:
            self.norm_key = nn.LayerNorm(embed_dims)
        else:
            self.norm_key = None
        self.key_is_query = key_is_query

        if not value_is_key:
            self.norm_value = nn.LayerNorm(embed_dims)
        else:
            self.norm_value = None
        self.value_is_key = value_is_key

        self.attn = MultiheadAttention(
            embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            q_project=True,
            k_project=True,
            v_project=False,
            proj_after_att=False,
            lan_dim = lan_dim)


    def forward(self, query, key, value, q_mask=None):
        def _inner_forward(query, key, value):
            q = self.norm_query(query)
            k = q if self.key_is_query else self.norm_key(key)
            v = k if self.value_is_key else self.norm_value(value)
            x = self.attn(q, k, v, q_mask) + self.drop(q)
            return x

        if self.with_cp:
            return cp.checkpoint(_inner_forward, query, key, value)
        else:
            return _inner_forward(query, key, value)
        


class GPBlock(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_group_token,
                 depth=1,
                 num_group_heads=4,
                 num_ungroup_heads=4,
                 lan_dim=768,
                 ffn_ratio=4.,
                 qkv_bias=True,
                 group_qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 with_cp=False,
                 group_att_cfg=dict(),
                 fwd_att_cfg=dict(),
                 ungroup_att_cfg=dict(),
                 **kwargs):

        super().__init__()

        self.embed_dims = embed_dims
        self.num_group_token = num_group_token
        self.with_cp = with_cp

        # self.group_token = nn.Parameter(torch.zeros(1, num_group_token, embed_dims))
        # trunc_normal_(self.group_token, std=.02)
        self.drop = nn.Dropout(drop)

        _group_att_cfg = dict(
            embed_dims=embed_dims,
            num_heads=num_group_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            qk_scale=group_qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            key_is_query=False,
            value_is_key=True,
            with_cp=with_cp,
            lan_dim = lan_dim)
        _group_att_cfg.update(group_att_cfg)
        self.group_layer = LightGroupAttnBlock(**_group_att_cfg)

        _mixer_cfg = dict(
            num_patches=num_group_token,
            embed_dims=embed_dims,
            patch_expansion=0.5,
            channel_expansion=4.0,
            depth=depth)
        _mixer_cfg.update(fwd_att_cfg)
        self.mixer = MLPMixer(**_mixer_cfg)

        _ungroup_att_cfg = dict(
            embed_dims=embed_dims,
            num_heads=num_ungroup_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            qk_scale=None,
            drop=drop,
            attn_drop=attn_drop,
            key_is_query=False,
            value_is_key=True,
            with_cp=with_cp)
        _ungroup_att_cfg.update(ungroup_att_cfg)
        self.un_group_layer = FullAttnCatBlock(**_ungroup_att_cfg)

        # self.dwconv = torch.nn.Sequential(
        #     nn.Conv2d(embed_dims, embed_dims, kernel_size=(3,3), padding=(1,1), bias=False, groups=embed_dims),
        #     nn.BatchNorm2d(num_features=embed_dims),
        #     nn.ReLU(True))


    def forward(self, q, x, q_mask=None):
        """
        Args:
            x: image tokens, shape [B, L, C]
            hw_shape: tuple or list (H, W)
        Returns:
            proj_tokens: shape [B, L, C]
        """
        # x = x
        # q_mask = q.sum(-1) != 0
        # print(q.shape, x.shape)
        gt = self.group_layer(query=q, key=x, value=x)
        if q_mask is not None:
            gt *= q_mask.unsqueeze(-1)
        gt = self.mixer(gt) + self.drop(gt)
        ungroup_tokens = self.un_group_layer(query=x, key=gt, value=gt, key_padding_mask=q_mask)
        # ungroup_tokens = ungroup_tokens.permute(0,2,1).contiguous().reshape(B, C, hw_shape[0], hw_shape[1])
        # proj_tokens = self.dwconv(ungroup_tokens).view(B, C, -1).permute(0,2,1).contiguous().view(B, L, C)

        return ungroup_tokens
    


if __name__ == '__main__':
    import numpy as np

    q = np.load('/storage_fast/ycli/3d_affordance/AffordanceNetQ_v1/question_bert.npy')
    q = torch.from_numpy(q[:2,:,:])
    v = torch.rand(2, 2048, 32) # torch.rand(2, 64, 1024)
    print(q.shape)
    _gp = {     'embed_dims':32,
                'depth':1,
                'num_group_heads':4,
                'num_ungroup_heads':4,
                'num_group_token':41
            }
    
    model = GPBlock(**_gp)
    out = model(q, v)
    print(out.shape)
