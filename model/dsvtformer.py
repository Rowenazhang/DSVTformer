

import torch
import torch.nn as nn
from functools import partial
from einops import rearrange
from timm.models.layers import DropPath

from common.opt import opts

opt = opts().parse()
device = torch.device("cuda")

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
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, v_dim, kv_num, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.kv_num = kv_num
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(v_dim, v_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(v_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xk, xv):
        B, N, C = xq.shape
        v_dim = xv.shape[-1]
        q = self.wq(xq).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                   3)
        k = self.wk(xk).reshape(B, self.kv_num, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                             3)
        v = self.wv(xv).reshape(B, self.kv_num, self.num_heads, v_dim // self.num_heads).permute(0, 2, 1,
                                                                                                 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N,
                                               v_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, kv_num, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.2,
                 attn_drop=0.2, drop_path=0.2, act_layer=nn.GELU, norm_layer=nn.LayerNorm, has_mlp=True):
        super().__init__()
        self.normq = norm_layer(q_dim)
        self.normk = norm_layer(k_dim)
        self.normv = norm_layer(v_dim)
        self.kv_num = kv_num
        self.attn = CrossAttention(q_dim, v_dim, kv_num=kv_num, num_heads=num_heads, qkv_bias=qkv_bias,
                                   qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(q_dim)
            mlp_hidden_dim = int(q_dim * mlp_ratio)
            self.mlp = Mlp(in_features=q_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, xq, xk, xv):
        xq = xq + self.drop_path(self.attn(self.normq(xq), self.normk(xk), self.normv(xv)))
        if self.has_mlp:
            xq = xq + self.drop_path(self.mlp(self.norm2(xq)))

        return xq

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class FusionBlock(nn.Module):
    def __init__(self, num_joint, num_frame, num_view, num_img, joint_in_ratio, joint_embed_ratio, img_in_ratio,
                 img_embed_ratio,
                 svt_mode, joint_out_ratio=None, add_residual=True):
        super(FusionBlock, self).__init__()

        self.num_joint = num_joint
        self.num_frame = num_frame
        self.num_view = num_view
        self.num_img = num_img
        self.add_residual = add_residual

        joint_num_heads = 8
        img_num_heads = 8
        mlp_ratio = 4.
        drop = 0.
        attn_drop = 0.
        drop_path = 0.2
        qkv_bias = True
        qk_scale = None
        self.joint_out_ratio = joint_out_ratio or joint_in_ratio

        if svt_mode == 'S':
            n_j = self.num_joint
            n_i = num_img
            joint_dim_in = joint_in_ratio * num_frame
            joint_dim_embed = joint_embed_ratio * num_frame
            joint_dim_out = joint_out_ratio * num_frame
            img_dim_in = img_in_ratio * num_frame
            img_dim_embed = img_embed_ratio * num_frame
            self.joint_proj = nn.Linear(joint_dim_in, joint_dim_embed)
            self.img_proj = nn.Linear(img_dim_in, img_dim_embed)
        elif svt_mode == 'V':
            n_j = self.num_view
            n_i = self.num_view
            joint_dim_in = joint_in_ratio * num_joint
            joint_dim_embed = joint_embed_ratio * num_joint
            joint_dim_out = joint_out_ratio * num_joint
            img_dim_in = img_in_ratio * num_img
            img_dim_embed = img_embed_ratio * num_img
            self.joint_proj = nn.Linear(joint_dim_in, joint_dim_embed)
            self.img_proj = nn.Linear(img_dim_in, img_dim_embed)
        elif svt_mode == 'T':
            n_j = self.num_frame
            n_i = self.num_frame
            joint_dim_in = joint_in_ratio * num_joint
            joint_dim_embed = joint_embed_ratio * num_joint
            joint_dim_out = joint_out_ratio * num_joint
            img_dim_in = img_in_ratio * num_img
            img_dim_embed = img_embed_ratio * num_img
            self.joint_proj = nn.Linear(joint_dim_in, joint_dim_embed)
            self.img_proj = nn.Linear(img_dim_in, img_dim_embed)

        self.joint_pos_embed = nn.Parameter(torch.randn(1, n_j, joint_dim_embed))
        self.img_pos_embed = nn.Parameter(torch.randn(1, n_i, img_dim_embed))
        self.j_Q_embed = nn.Parameter(torch.randn(1, n_j, joint_dim_embed))
        self.i_Q_embed = nn.Parameter(torch.randn(1, n_i, img_dim_embed))

        self.proj_i2j_dim = nn.Linear(img_dim_embed, joint_dim_embed)
        self.proj_j2i_dim = nn.Linear(joint_dim_embed, img_dim_embed)
        self.i2j_K_embed = nn.Parameter(torch.randn(1, n_j, joint_dim_embed))
        self.j2i_K_embed = nn.Parameter(torch.randn(1, n_i, img_dim_embed))

        self.joint_SA_FFN = Block(dim=joint_dim_embed, num_heads=joint_num_heads, mlp_ratio=mlp_ratio,
                                  qkv_bias=qkv_bias,
                                  drop=drop, attn_drop=attn_drop, drop_path=drop_path)
        self.img_SA_FFN = Block(dim=img_dim_embed, num_heads=img_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                drop=drop, attn_drop=attn_drop, drop_path=drop_path)

        self.joint_CA_FFN = CrossAttentionBlock(q_dim=joint_dim_embed, k_dim=joint_dim_embed, v_dim=img_dim_embed,
                                                kv_num=n_j,
                                                num_heads=joint_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                                drop=drop, attn_drop=attn_drop, drop_path=drop_path, has_mlp=True)
        self.img_CA_FFN = CrossAttentionBlock(q_dim=img_dim_embed, k_dim=img_dim_embed, v_dim=joint_dim_embed,
                                              kv_num=n_i,
                                              num_heads=img_num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                                              drop=drop, attn_drop=attn_drop, drop_path=drop_path, has_mlp=True)

        self.proj_joint_feat2coor = nn.Linear(joint_dim_embed, joint_dim_out)
        self.proj_img_feat2orig = nn.Linear(img_dim_embed, img_dim_in)

    def forward(self, joint, img, mode):
        b, f, v, nj, cj = joint.shape
        b, f, v, ni, ci = img.shape

        if mode == 'ST':
            joint = rearrange(joint, 'b f v j c -> (b v) j (c f)')
            img = rearrange(img, 'b f v j c -> (b v) j (c f)')
        elif mode == 'VT':
            joint = rearrange(joint, 'b f v j c -> (b f) v (c j)')
            img = rearrange(img, 'b f v j c -> (b f) v (c j)')
        elif mode == 'TT':
            joint = rearrange(joint, 'b f v j c -> (b v) f (c j)')
            img = rearrange(img, 'b f v j c -> (b v) f (c j)')
        joint_feat, img_feat = self.joint_proj(joint), self.img_proj(
            img)

        joint_feat, img_feat = joint_feat + self.joint_pos_embed, img_feat + self.img_pos_embed

        joint_feat = self.joint_CA_FFN(joint_feat + self.j_Q_embed, self.proj_i2j_dim(img_feat) + self.i2j_K_embed,
                                       img_feat)
        img_feat = self.img_CA_FFN(img_feat + self.i_Q_embed, self.proj_j2i_dim(joint_feat) + self.j2i_K_embed,
                                   joint_feat)

        joint_feat, img_feat = self.joint_SA_FFN(joint_feat), self.img_SA_FFN(img_feat)

        if self.add_residual:
            joint, img = self.proj_joint_feat2coor(joint_feat) + joint, self.proj_img_feat2orig(img_feat) + img
        else:
            joint, img = self.proj_joint_feat2coor(joint_feat), self.proj_img_feat2orig(img_feat)

        if mode == 'ST':
            joint = rearrange(joint, '(b v) j (c f) -> b f v j c', b=b, v=v, j=nj, f=f)
            img = rearrange(img, '(b v) j (c f) -> b f v j c', b=b, v=v, j=ni, f=f)
        elif mode == 'VT':
            joint = rearrange(joint, '(b f) v (c j) -> b f v j c', b=b, f=f, v=v, j=nj)
            img = rearrange(img, '(b f) v (c j) -> b f v j c', b=b, f=f, v=v, j=ni)
        elif mode == 'TT':
            joint = rearrange(joint, '(b v) f (c j) -> b f v j c', b=b, v=v, f=f, j=nj)
            img = rearrange(img, '(b v) f (c j) -> b f v j c', b=b, v=v, f=f, j=ni)

        return joint, img

class Model(nn.Module):
    def __init__(self, num_frame=9, num_joints=17, in_chans=2, embed_dim_ratio=32, img_embed_dim_ratio=32, depth=4,
                 num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=None):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        view_num = 4

        out_dim = num_joints * 3

        self.pos_drop = nn.Dropout(p=0.)
        self.conv = nn.Sequential(
            nn.BatchNorm2d(4, momentum=0.1),
            nn.Conv2d(4, 1, kernel_size=opt.mvf_kernel, stride=1, padding=int(opt.mvf_kernel // 2), bias=False),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(view_num * 3 * num_joints),
            nn.Linear(view_num * 3 * num_joints, out_dim),
        )

        self.block_depth = depth

        joint_embed_ratio = embed_dim_ratio
        img_in_ratio = 256
        img_embed_ratio = img_embed_dim_ratio
        self.num_img = num_joints

        self.fusionblocks = nn.ModuleList()
        for i in range(depth):
            if i == 0:

                self.fusionblocks.append(nn.ModuleList([
                    FusionBlock(num_joints, num_frame, view_num, self.num_img,
                                joint_in_ratio=2, joint_embed_ratio=joint_embed_ratio,
                                img_in_ratio=img_in_ratio, img_embed_ratio=img_embed_ratio,
                                svt_mode='S', joint_out_ratio=3, add_residual=False),
                    FusionBlock(num_joints, num_frame, view_num, self.num_img,
                                joint_in_ratio=3, joint_embed_ratio=joint_embed_ratio,
                                img_in_ratio=img_in_ratio, img_embed_ratio=img_embed_ratio,
                                svt_mode='V', joint_out_ratio=3, add_residual=False),
                    FusionBlock(num_joints, num_frame, view_num, self.num_img,
                                joint_in_ratio=3, joint_embed_ratio=joint_embed_ratio,
                                img_in_ratio=img_in_ratio, img_embed_ratio=img_embed_ratio,
                                svt_mode='T', joint_out_ratio=3, add_residual=False),
                ]))
            else:
                self.fusionblocks.append(nn.ModuleList([
                    FusionBlock(num_joints, num_frame, view_num, self.num_img,
                                joint_in_ratio=3, joint_embed_ratio=joint_embed_ratio,
                                img_in_ratio=img_in_ratio, img_embed_ratio=img_embed_ratio,
                                svt_mode='S', joint_out_ratio=3, add_residual=True),
                    FusionBlock(num_joints, num_frame, view_num, self.num_img,
                                joint_in_ratio=3, joint_embed_ratio=joint_embed_ratio,
                                img_in_ratio=img_in_ratio, img_embed_ratio=img_embed_ratio,
                                svt_mode='V', joint_out_ratio=3, add_residual=True),
                    FusionBlock(num_joints, num_frame, view_num, self.num_img,
                                joint_in_ratio=3, joint_embed_ratio=joint_embed_ratio,
                                img_in_ratio=img_in_ratio, img_embed_ratio=img_embed_ratio,
                                svt_mode='T', joint_out_ratio=3, add_residual=True),
                ]))

    def forward(self, x, img):
        b, f, v, j, c = x.shape
        b2, f2, v2, c2 = img.shape
        assert (b, f, v) == (b2, f2, v2), "Mismatch between x and img dimensions"

        img = img.unsqueeze(-2).repeat(1, 1, 1, self.num_img, 1)
        for i in range(self.block_depth):
            fusion_s, fusion_v, fusion_t = self.fusionblocks[i]
            x, img = fusion_s(x, img, 'ST')
            x, img = fusion_v(x, img, 'VT')
            x, img = fusion_t(x, img, 'TT')

        x = rearrange(x, 'b f v j c -> b f (v j c)', b=b, j=j, v=v)
        x = self.head(x)

        x = rearrange(x, 'b f (j c) -> b f j c', j=j).contiguous()

        return x
