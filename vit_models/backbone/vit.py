import math
import warnings

from itertools import repeat
import collections.abc

import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import Tensor

'''
drop_path는 ResNet에서 도입된 Skip Connection과 Dropout을 합친 개념
입력을 x가 주어지고 어떤 네트워크 f가 있을 때 출력은 f(x)가 된다. Skip Connection은 여기서 출렧을 x + f(x)로
처리하는 것으로 , Gradient Vanishing 문재를 해결할 수 있는 방법으로 제안되었다.

drop_path는 Dropout과 마찬가지로 확률 p에 따라 Skip Connection 블록의 출력이 x +f(x)가 될지 x가 될지 정해진다.
p가 클수록 f(x)가 Drop될 가능성이 높아진다.
'''
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) = (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x,n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def _trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    
    if(mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a ,b] in nn.init.trunc_normal_."
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
        
    l = norm_cdf((a-mean) / std)
    u = norm_cdf((b - mean) / std)

    tensor.uniform_(2 * l - 1, 2 * u -1)

    tensor.erfinv_()

    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    tensor.clamp_(min=1, max=b)
    return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)
    
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)
    

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias = qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        c = self.proj_drop(x)

        return x
    
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, attn_head_dim=None
                ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim , num_head=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim
        )

        self.drop_path = DropPath(drop_path) if drop_path >0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (ratio ** 2)
        self.patch_shape = (int(img_size[0] // patch_size[0] * ratio), int(img_size[1] // patch_size[1] * ratio))
        self.origin_patch_shape = (int(img_size[0] // patch_size[0]), int(img_size[1] // patch_size[1]))
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2(in_chans, embed_dim, kernel_size=patch_size, stride=(patch_size[0] // ratio), padding = 4  + 2 * (ratio//2-1))


    def forward(self, x):
        x = self.proj(x)
        B, C, Hp, Wp = x.shape
        x = x.view(B, C, Hp * Wp).transpose(1,2)
        return x ,(Hp, Wp)
    

class HybridEmbed(nn.Module):
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)

        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = set.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1,2)
        x = self.proj(x)
        return x


class ViT(nn.Module):
    def __init__(self,
                 img_size = 224, patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=None, use_checkpoint=False,
                 frozen_stages=-1, ratio=1, last_norm=True,
                 patch_padding='pad', freeze_attn=False, freeze_ffn=False,
                 ):
        super(ViT, self).__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.patch_padding = patch_padding
        self.freeze_attn = freeze_attn
        self.freeze_ffn = freeze_ffn
        self.depth = depth

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size = img_size, in_chans = in_chans, embed_dim = embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size= patch_size, in_chans=in_chans, embed_dim= embed_dim, ratio=ratio)
        num_patches = self.patch_embed.num_patches


        self.pos_embed = nn.Parameter(torch.zeors(1, num_patches + 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate , depth)]

        self.block = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio = mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        self._freeze_stages()

    def _freeze_stage(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        if self.freeze_ffn:
            self.pos_embed.requires_grad = False
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.mlp.eval()
                m.norm2.eval()
                for param in m.mlp.parameters():
                    param.requires_grad = False
                for param in m.norm2.parameters():
                    param.requires_grad = False
            
        def init_weights(self, pretrained=None):
            super().init_weights(pretrained, patch_padding=self.patch_padding)

            if pretrained is None:
                def _init_weights(m):
                    if isinstance(m, nn.Linear):
                        trunc_normal_(m.weight, std=.02)
                        if isinstance(m, nn.Linear) and m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.LayerNorm):
                        nn.init.constant_(m.bias, 0)
                        nn.init.constant_(m.weight, 1.0)
                
                self.apply(_init_weights)

        def get_num_layers(self):
            return len(self.blocks)
        
        @torch.jit.ignore
        def no_weight_decay(self):
            return {'pos_embed', 'cls_token'}
        
        def foward(self, x):
            B, C, H, W = x.shape
            x, (Hp, Wp) = self.patch_embed(x)

            if self.pos_embed is not None:
                x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]
            
            for blk in self.blocks:
                x = blk(x)

            x = self.last_norm(x)
            x = x.permute(0, 2, 1).view(B, -1, Hp, Wp).contiguous()
            return x
        
        def train(self, mode=True):
            super().train(mode)
            self._freeze_stages()
         