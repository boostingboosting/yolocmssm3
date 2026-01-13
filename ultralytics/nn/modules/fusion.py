import torch
import torch.nn as nn
from jinja2.utils import concat
from torch import Tensor

import torch.nn.functional as F

from timm.layers import trunc_normal_
import math

from einops import rearrange, repeat
# from model_others.RGB_T.CMX.models.net_utils import FeatureFusionModule as FFM
# from model_others.RGB_T.CMX.models.net_utils import FeatureRectifyModule as FRM
# from models.attention_module import FeatureFusionModule
# from model_others.RGB_T.MAINet import TSFA
# from model_others.RGB_T.MDNet.model import MultiSpectralAttentionLayer, SS_Conv_SSM
from ultralytics.nn.modules.backbone.MedMamba import SS2D
# from model_others.RGB_T.sigma.encoders.vmamba import CrossMambaFusionBlock, ConcatMambaFusionBlock
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass

from .conv import Conv
from .rs_mamba_cd import CrossScan, CrossMerge

"""
class Fusion_Module(nn.Module):
    def __init__(self, fusion_mode, channels, num_heads=[1, 2, 4, 8], norm_fuse=nn.BatchNorm2d):
        super(Fusion_Module, self).__init__()
        self.fusion_mode = fusion_mode
        self.norm_layer = nn.ModuleList()
        self.fusion = nn.ModuleList()

        if self.fusion_mode == 'cat':
            for i in channels:
                self.fusion.append(BasicConv(2 * i, i, 3, 1, 1))

        if self.fusion_mode == 'demo1':
            for i in channels:
                self.fusion.append(Demo1(i))

        # if self.fusion_mode == 'CMX':
        #     self.FRMs = nn.ModuleList([
        #         FRM(dim=channels[0], reduction=1),
        #         FRM(dim=channels[1], reduction=1),
        #         FRM(dim=channels[2], reduction=1),
        #         FRM(dim=channels[3], reduction=1)])

        #     self.FFMs = nn.ModuleList([
        #         FFM(dim=channels[0], reduction=1, num_heads=num_heads[0], norm_layer=norm_fuse),
        #         FFM(dim=channels[1], reduction=1, num_heads=num_heads[1], norm_layer=norm_fuse),
        #         FFM(dim=channels[2], reduction=1, num_heads=num_heads[2], norm_layer=norm_fuse),
        #         FFM(dim=channels[3], reduction=1, num_heads=num_heads[3], norm_layer=norm_fuse)])

        # if self.fusion_mode == 'CDA':
        #     self.CDAs = nn.ModuleList([
        #         FeatureFusionModule(channels[0]),
        #         FeatureFusionModule(channels[1]),
        #         FeatureFusionModule(channels[2]),
        #         FeatureFusionModule(channels[3]),
        #     ])
            # self.CDA = FeatureFusionModule(channels[3])

        # if self.fusion_mode == 'sigma':
        #     self.CroMB = nn.ModuleList()
        #     self.ConMB = nn.ModuleList()
        #     for i in channels:
        #         self.CroMB.append(CrossMambaFusionBlock(hidden_dim=i,
        #                               mlp_ratio=0.0,
        #                               d_state=4, ))
        #         self.ConMB.append(ConcatMambaFusionBlock(hidden_dim=i,
        #                                                         mlp_ratio=0.0,
        #                                                         d_state=4,))

        #     self.FFMs = nn.ModuleList([
        #         FFM(dim=channels[0], reduction=1, num_heads=num_heads[0], norm_layer=norm_fuse),
        #         FFM(dim=channels[1], reduction=1, num_heads=num_heads[1], norm_layer=norm_fuse),
        #         FFM(dim=channels[2], reduction=1, num_heads=num_heads[2], norm_layer=norm_fuse),
        #         FFM(dim=channels[3], reduction=1, num_heads=num_heads[3], norm_layer=norm_fuse)])

        # if self.fusion_mode == 'MDFusion':
        #     self.fusion = MDFusion(channels)

        # if self.fusion_mode == 'TSFA':
        #     for channel in channels:
        #         self.fusion.append(TSFA(channel))

        if self.fusion_mode == 'CM-SSM' or self.fusion_mode == 'CM-SSM-offset':
            for channel in channels[1:]:
                self.fusion.append(CM_SSM(channel))

        if self.fusion_mode.endswith("align"): ##先对齐，再融合
            for channel in channels[1:]:
                self.fusion.append(CM_SSM(channel))


        if self.fusion_mode == 'M-SSM':
            for channel in channels:
                self.fusion.append(M_SSM(channel))


        if self.fusion_mode == 'MoE':
            # print("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")
            for channel in channels:
                self.fusion.append(FusionMoE(channel))





    def forward(self, rgb, t):
        outs = []
        if self.fusion_mode == 'MDFusion':
            outs = self.fusion(rgb, t)

        for i in range(len(t)):
            if self.fusion_mode == 'add':
                outs.append(rgb[i] + t[i])

            if self.fusion_mode == 'max':
                out, _ = torch.max(torch.stack([rgb[i], t[i]], dim=1), dim=1)
                outs.append(out)

            if self.fusion_mode == 'demo1':
                outs.append(self.fusion[i](rgb[i], t[i]))

            # if self.fusion_mode == 'CMX':
            #     rgb_, t_ = self.FRMs[i](rgb[i], t[i])
            #     # rgb_, t_ = rgb[i], t[i]
            #     out = self.FFMs[i](rgb_, t_)
            #     outs.append(out)

            # if self.fusion_mode == 'TSFA':
            #     outs.append(self.fusion[i](rgb[i], t[i]))

            if self.fusion_mode == 'CDA':
                if i >= 1:
                    outs.append(self.CDAs[i](rgb[i], t[i]))
                else:
                    outs.append(rgb[i]+t[i])
                # else:
                #     outs.append(self.CDAs[i](rgb[i], t[i]))
                # outs.append(self.CDAs[i](rgb[i], t[i]))

            if self.fusion_mode == 'cat':
                out= self.fusion[i](torch.cat((rgb[i], t[i]), dim=1))
                outs.append(out)


            # Mamba4 is best
            if self.fusion_mode in ['CM-SSM', 'CM-SSM-offset', 'M-SSM', 'MoE']:
                outs.append(self.fusion[i](rgb[i], t[i]))

            if self.fusion_mode.endswith("align"): ##先对齐，再融合
                outs.append(self.fusion[i](rgb[i], t[i]))

            

            if self.fusion_mode == 'sigma':
                rgb_, t_ = self.CroMB[i](rgb[i].permute(0, 2, 3, 1), t[i].permute(0, 2, 3, 1))
                out = self.ConMB[i](rgb_, t_).permute(0, 3, 1, 2)
                outs.append(out)

        return outs

class CM_SSM(nn.Module):
    def __init__(self, in_c, rgb_residual=True):
        super(CM_SSM, self).__init__()
        self.SS2D = SS2D_rgbt(in_c)
        self.conv1 = nn.Sequential(nn.Conv2d(2*in_c, in_c, 3, 1, 1),
                                   nn.BatchNorm2d(in_c),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(3*in_c, in_c, 1, 1, 0),
                                   nn.BatchNorm2d(in_c),
                                   nn.ReLU())

        self.rgb_residual = rgb_residual

    def forward(self, rgb, t):
        B, C, H, W = rgb.shape
        left = self.conv1(torch.cat((rgb, t), dim=1))

        rgb_, t_ = self.SS2D(rgb.permute(0, 2, 3, 1), t.permute(0, 2, 3, 1))

        rgb_ = rgb_ + rgb
        t_ = t_ + t

        out = self.conv2(torch.cat((left, rgb_, t_), dim=1))
        return out

##全局仿射变换预测

class AffinePredictor(nn.Module):
    def __init__(self, in_ch=64):
        super().__init__()
        
        # 为每层创建仿射预测头（带物理约束）
        self.affine_heads=nn.Sequential(
                nn.Conv2d(2*in_ch, 2*in_ch, 3, padding=1),
                nn.BatchNorm2d(2*in_ch),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(2 * in_ch, 6)  # 2*in_ch: IR+加权RGB融合后通道数
            )
        
        # **物理先验约束参数（可学习）**
        self.scale_a = nn.Parameter(torch.tensor([0.2]))  # a/d范围 [0.8,1.2]
        self.scale_d = nn.Parameter(torch.tensor([0.2]))
        self.scale_b = nn.Parameter(torch.tensor([0.1]))  # b/c范围 [-0.05,0.05]
        self.scale_c = nn.Parameter(torch.tensor([0.1]))
        self.scale_tx = nn.Parameter(torch.tensor([0.2]))  # tx/ty范围 [-0.1,0.1]
        self.scale_ty = nn.Parameter(torch.tensor([0.2]))
        self.bias_a = nn.Parameter(torch.tensor([1.0]))
        self.bias_d = nn.Parameter(torch.tensor([1.0]))
        self.bias_b = nn.Parameter(torch.tensor([0.0]))
        self.bias_c = nn.Parameter(torch.tensor([0.0]))
        self.bias_tx = nn.Parameter(torch.tensor([0.0]))
        self.bias_ty = nn.Parameter(torch.tensor([0.0]))
    
    def forward(self, fused):      
        affine_params_raw = self.affine_heads(fused)  # [B, 6]
        
        # 5. 应用物理先验约束（确保参数合理）
        a = F.sigmoid(affine_params_raw[:, 0]) * self.scale_a + self.bias_a
        b = torch.tanh(affine_params_raw[:, 1]) * self.scale_b + self.bias_b
        c = torch.tanh(affine_params_raw[:, 2]) * self.scale_c + self.bias_c
        d = F.sigmoid(affine_params_raw[:, 3]) * self.scale_d + self.bias_d
        tx = torch.tanh(affine_params_raw[:, 4]) * self.scale_tx + self.bias_tx
        ty = torch.tanh(affine_params_raw[:, 5]) * self.scale_ty + self.bias_ty
        
        # 拼接成6参数
        affine_params = torch.stack([a, b, c, d, tx, ty], dim=1)
    
        return affine_params

##局部非线性像素级微调
class LocalOffsetPredictor(nn.Module):
    def __init__(self, in_ch):
        super(LocalOffsetPredictor, self).__init__()
        self.conv_offset = nn.Conv2d(in_ch, 2, kernel_size=3, padding=1)  # 输出两个通道，分别对应δx和δy
        
    def forward(self, x):
        offset = self.conv_offset(x)  # (N, 2, H, W)
        offset = torch.tanh(offset) * 0.01  # 将偏移量限制在 [-0.01, 0.01]
        return offset

##rgb图像矫正器
class DepthwiseConv(nn.Module):
    ##深度可分离卷积（轻量级核心）
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return x

class RGBAdjuster(nn.Module):
    def __init__(self, channels=[64, 128, 256]):
        super().__init__()
        self.weight_heads = nn.ModuleList()  # 每层的权重预测头（DWConv）
        self.affine_heads = nn.ModuleList()  # 每层的仿射预测头（带物理约束）
        self.offset_heads = nn.ModuleList()  # 每层的局部非线性像素级预测头
        
        # 为每层创建权重预测头（DWConv + Sigmoid）,每个像素上有无无人机
        for in_ch in channels:
            self.weight_heads.append(
                nn.Sequential(
                    DepthwiseConv(in_ch, 1),  # 输出1通道（权重图）
                    nn.Sigmoid()              # [0,1]归一化
                )
            )

        for in_ch in channels:
            self.affine_heads.append(AffinePredictor(in_ch))
            self.offset_heads.append(LocalOffsetPredictor(in_ch*2))

    def forward(self, rgb_feats, ir_feats):
        txtys = []
        wraped_rgb_feats = []
        # print("len(ir_feats):",len(ir_feats))
        for i in range(len(ir_feats)):
            # print("rgb_feats[i].shape:",rgb_feats[i].shape)
            # 1. 预测RGB权重图（无人机置信度）,每个像素上是否是无人机
            weight_map = self.weight_heads[i](rgb_feats[i])  # [B, 1, H, W]
            
            # 2. 加权RGB特征（空间感知）
            weighted_rgb = rgb_feats[i] * weight_map  # [B, C, H, W]
            
            # 3. 融合IR和加权RGB特征
            fused = torch.cat([ir_feats[i], weighted_rgb], dim=1)  # [B, 2*C, H, W] 

            # 4. 偏移预测
            affine_params = self.affine_heads[i](fused)
            # print("affine_params.shape:", affine_params.shape) #[B,6]
            txtys.append(affine_params[:, 4:])
            offsets = self.offset_heads[i](fused)
            # print("rgb_feats[i].shape:",rgb_feats[i].shape)
            wraped_rgb_feats.append(self.warp_features(ir_feats[i], rgb_feats[i], affine_params, offsets))

        # print("txtys:", txtys)
        return wraped_rgb_feats, ir_feats, txtys

    #应用偏移到图像
    def warp_features(self, ir_feats, rgb_feats, affine_params, offsets):
        
        #对RGB特征图进行矫正，保持尺寸不变
        #ir_feats: list of [B, C, H, W] (IR特征)
        #rgb_feats: list of [B, C, H, W] (RGB特征)
        #affine_params: list of [B, 6] (每层仿射参数)
        #offsets: list of [B, 2, H, W] (每层局部偏移)
        #Returns: 
        #    warped_rgb_feats: list of [B, C, H, W] (矫正后的RGB特征)
        
        B, C, H, W = rgb_feats.shape
        
        # 1. 应用仿射变换 (得到基础对齐)
        affine_params = affine_params.view(-1, 2, 3)  # [B, 2, 3]
        
        # 生成仿射网格 (归一化坐标 [-1,1])
        grid = F.affine_grid(affine_params, size=[B, C, H, W], align_corners=True)
        
        # 2. 应用局部偏移 (在仿射网格上叠加)
        offsets = offsets.permute(0, 2, 3, 1)  # [B, H, W, 2]
        
        # 转换为归一化坐标偏移 ([-1,1]范围)
        offset_norm = offsets * 2.0   # 归一化到[-1,1]
        final_grid = grid + offset_norm
        
        # 3. 采样矫正后的RGB特征
        warped = F.grid_sample(
            rgb_feats, 
            final_grid, 
            align_corners=True, 
            mode='bilinear'
        )
        
        
        return warped


# class CrossAttention(nn.Module):
#     def __init__(self, dim, heads=8):
#         super().__init__()
#         self.conv1 = nn.Sequential(nn.Conv2d(2*dim, dim, 3, 1, 1),
#                                    nn.BatchNorm2d(dim),
#                                    nn.ReLU())
#         self.conv2 = nn.Sequential(nn.Conv2d(3*dim, dim, 1, 1, 0),
#                                    nn.BatchNorm2d(dim),
#                                    nn.ReLU())
#         self.attn_rgb = nn.MultiheadAttention(dim, heads, batch_first=True)
#         self.attn_t = nn.MultiheadAttention(dim, heads, batch_first=True)
        
#     def forward(self, rgb, t):
#         B, C, H, W = rgb.shape
#         left = self.conv1(torch.cat((rgb, t), dim=1))

#         # 修正：使用 view 而不是 view_as
#         rgb_flat = rgb.view(B, C, H*W).permute(0, 2, 1).contiguous()
#         t_flat = t.view(B, C, H*W).permute(0, 2, 1).contiguous()
        
#         # 修正：使用正确的注意力模块名称
#         rgb_sup, _ = self.attn_rgb(rgb_flat, t_flat, t_flat)
#         t_sup, _ = self.attn_t(t_flat, rgb_flat, rgb_flat)
        
#         # 修正：使用 view 而不是 view_as
#         rgb_ = (rgb_flat + rgb_sup).permute(0, 2, 1).view(B, C, H, W)
#         t_ = (t_flat + t_sup).permute(0, 2, 1).view(B, C, H, W)

#         out = self.conv2(torch.cat((left, rgb_, t_), dim=1))
#         return out
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class LightCrossAttentionFusion(nn.Module):
    """
    轻量级交叉注意力融合模块
    适用于 两输入特征 shape 完全一致 (B,C,H,W) 的情况
    适合替代标准 Cross-Attention，更省显存、参数更少、可复用性高
    """
    def __init__(self, dim, reduction=4, share_proj=True):
        """
        Args:
            dim (int): 输入特征通道数 C
            reduction (int): 通道压缩比（越大越省显存）
            share_proj (bool): 是否共享 Q/K/V 投影权重（更省参数）
        """
        super().__init__()
        self.dim = dim
        self.dim_r = dim // reduction  # 压缩后的通道维度

        # ========== 1. 线性投影 (Q/K/V) ==========
        # 采用 1x1 Conv 作为 Channel 线性投影

        self.q_proj = nn.Conv2d(dim, self.dim_r, 1, bias=False)
        self.k_proj = nn.Conv2d(dim, self.dim_r, 1, bias=False)
        self.v_proj = nn.Conv2d(dim, self.dim_r, 1, bias=False)

        # ========== 2. 轻量空间建模 (DepthWise) ==========
        # 代替昂贵的 MHA，使模型更轻量
        self.dw_q = nn.Conv2d(self.dim_r, self.dim_r, 3, padding=1, groups=self.dim_r)
        self.dw_k = nn.Conv2d(self.dim_r, self.dim_r, 3, padding=1, groups=self.dim_r)
        self.dw_v = nn.Conv2d(self.dim_r, self.dim_r, 3, padding=1, groups=self.dim_r)

        # ========== 3. 输出投影恢复通道 ==========
        self.out_proj = nn.Sequential(
            nn.Conv2d(self.dim_r, dim, 1),
            nn.BatchNorm2d(dim)
        )

        self.scale = self.dim_r ** -0.5  # 注意力缩放系数

    def forward(self, x1, x2):
        # x1, x2 shape: (B, C, H, W)
        q, k, v= self.q_proj(x1), self.k_proj(x2), self.v_proj(x2)
        # ---------- DepthWise 轻量空间关联 ----------
        q, k, v = self.dw_q(q), self.dw_k(k), self.dw_v(v)


        # ---------- Reshape 以用于交叉注意力计算 ----------
        B, C, H, W = q.shape
        # flatten spatial
        q = q.flatten(2).transpose(1, 2)  # (B, HW, C)
        k = k.flatten(2).transpose(1, 2)
        v = v.flatten(2).transpose(1, 2)

        # ---------- 交叉注意力 ----------
        attn1 = (q @ k.transpose(-2, -1)) * self.scale  # x1 关注 x2
        # attn2 = (q2 @ k.transpose(-2, -1)) * self.scale  # x2 关注 x1

        attn1 = F.softmax(attn1, dim=-1)
        # attn2 = F.softmax(attn2, dim=-1)

        out1 = attn1 @ v
        # out2 = attn2 @ v

        # reshape 回原空间结构
        out1 = out1.transpose(1, 2).view(B, C, H, W)
        # out2 = out2.transpose(1, 2).view(B, C, H, W)

        # ---------- 线性输出投影 + 残差融合 ----------
        out1 = self.out_proj(out1) + x1


        return out1  # 返回双向融合后的特征

class CAF(nn.Module):
    def __init__(self, in_c,ratio=0.5):
        super(CAF, self).__init__()
        self.c = int(in_c * ratio)
        # 1×1 分裂主干 (a: 直连, b: 参与Cross-Attn)
        self.split_rgb = Conv(in_c, 2 * self.c, 1)
        self.split_t = Conv(in_c, 2 * self.c, 1)



        self.LCAF = LightCrossAttentionFusion(self.c)

        self.merge = nn.Sequential(nn.Conv2d(3*self.c, in_c, 1, 1, 0),
                                   nn.BatchNorm2d(in_c),
                                   nn.ReLU())
    def forward(self, rgb, t):
        # 1) 通道拆分
        a_rgb, b_rgb = self.split_rgb(rgb).split(self.c, dim=1)
        a_t, b_t = self.split_t(t).split(self.c, dim=1)

        # 2) 交叉模态信息注入
        fuse = self.LCAF(b_rgb, b_t)

        # 4) 与保留分支concat+融合输出
        out = self.merge(torch.cat([a_rgb, a_t, fuse], dim=1))

        return out
# ---------------------------------------------
# 多模态融合主模块（Wang-Zheng Multi-Modal Fusion）
# ---------------------------------------------
# class WangZhengMMF(nn.Module):
#     """
#     输入：
#         x  : 主特征 (B, C, H, W)  例如：图像
#         ctx: 辅助特征 (B, C, H, W)  例如：音频/文本投影后的2D结构
#     结构：
#         1) 低通道分支保留原始信息
#         2) 高语义分支执行 Cross-Attention 融合
#         3) concat 之后 1×1 conv 融合恢复
#     特性：
#         ✔ 轻量 ✔ 低显存 ✔ 多模态交叉增强 ✔ 即插即用
#     """
#     def __init__(self, channels, ratio=0.5, heads=4):
#         super().__init__()
#         self.c = int(channels * ratio)

#         # 1×1 分裂主干 (a: 直连, b: 参与Cross-Attn)
#         self.split = Conv(channels, 2 * self.c, 1)
#         self.merge = Conv(2 * self.c, channels, 1)

#         # 轻量 Cross Attention + FFN
#         self.cross_attn = CAF(self.c)
#         self.ffn = nn.Sequential(
#             Conv(self.c, self.c * 2, 1),
#             Conv(self.c * 2, self.c, 1, act=False)
#         )

#     def forward(self, x, ctx):
#         # 1) 通道拆分
#         a, b = self.split(x).split(self.c, dim=1)

#         # 2) 交叉模态信息注入
#         b = b + self.cross_attn(b, ctx)

#         # 3) 轻量FFN增强
#         b = b + self.ffn(b)

#         # 4) 与保留分支concat+融合输出
#         out = self.merge(torch.cat([a, b], dim=1))
#         return out


class FusionMoE(nn.Module):
    def __init__(self, inc, num_cross=2, num_cm=2, heads=8, 
                 num_experts_per_group=1):
        super().__init__()
        self.dim = inc
        self.num_experts_per_group = num_experts_per_group

        # 分层专家池
        self.cross_experts = nn.ModuleList([
            CAF(self.dim) for _ in range(num_cross)
        ])
        self.cm_experts = nn.ModuleList([
            CM_SSM(self.dim) for _ in range(num_cm)
        ])
        
        self.num_cross_experts = num_cross
        self.num_cm_experts = num_cm

        # 分层路由器网络
        self.router_reduce = nn.Linear(self.dim * 2, self.dim)
        
        # 组级路由器 - 决定使用哪个组
        self.group_router = nn.Linear(self.dim, 2)  # 2个组：cross和cm
        
        # 专家级路由器 - 每个组内选择专家
        self.cross_router = nn.Linear(self.dim, num_cross)
        self.cm_router = nn.Linear(self.dim, num_cm)

    def forward(self, rgb, t):
        B, C, H, W = rgb.shape
        
        # 1. 序列级路由输入
        rgb_flat = rgb.view(B, C, H*W).transpose(1, 2)
        t_flat = t.view(B, C, H*W).transpose(1, 2)
        
        v_mean = rgb_flat.mean(dim=1)  # [B, D]
        a_mean = t_flat.mean(dim=1)   # [B, D]
        
        routing_input = torch.cat([v_mean, a_mean], dim=-1)  # [B, 2*D]
        reduced_routing = self.router_reduce(routing_input)  # [B, D]

        # 2. 组级路由 - 使用top1选择组
        group_logits = self.group_router(reduced_routing)  # [B, 2]
        group_probs = F.softmax(group_logits, dim=-1)
        
        # 选择top1组
        _, selected_groups = torch.topk(group_probs, 1, dim=-1)  # [B, 1]
        selected_groups = selected_groups.squeeze(-1)  # [B]
        
        # 3. 专家级路由
        cross_logits = self.cross_router(reduced_routing)  # [B, num_cross]
        cm_logits = self.cm_router(reduced_routing)       # [B, num_cm]
        
        # 选择每个组内的专家
        cross_weights = F.softmax(cross_logits, dim=-1)
        cm_weights = F.softmax(cm_logits, dim=-1)
        
        # 选择top-k专家
        cross_expert_weights, cross_expert_indices = torch.topk(
            cross_weights, self.num_experts_per_group, dim=-1
        )
        cm_expert_weights, cm_expert_indices = torch.topk(
            cm_weights, self.num_experts_per_group, dim=-1
        )
        
        # 归一化专家权重
        cross_expert_weights = cross_expert_weights / cross_expert_weights.sum(dim=-1, keepdim=True)
        cm_expert_weights = cm_expert_weights / cm_expert_weights.sum(dim=-1, keepdim=True)
        
        # 4. 计算专家输出
        final_output = torch.zeros_like(rgb)
        aux_loss = 0.0
        
        # 为每个样本计算专家输出
        for i in range(B):
            selected_group = selected_groups[i].item()
            
            if selected_group == 0:  # 选择Cross Attention组
                output = torch.zeros_like(rgb[i:i+1])
                for j in range(self.num_experts_per_group):
                    expert_idx = cross_expert_indices[i, j]
                    expert = self.cross_experts[expert_idx]
                    expert_out = expert(t[i:i+1], rgb[i:i+1])
                    weight = cross_expert_weights[i, j]
                    output += weight * expert_out
                final_output[i:i+1] = output
                
            else:  # 选择CM-SSM组
                output = torch.zeros_like(rgb[i:i+1])
                for j in range(self.num_experts_per_group):
                    expert_idx = cm_expert_indices[i, j]
                    expert = self.cm_experts[expert_idx]
                    expert_out = expert(rgb[i:i+1], t[i:i+1])
                    weight = cm_expert_weights[i, j]
                    output += weight * expert_out
                final_output[i:i+1] = output
        
        # 5. 计算辅助损失（负载均衡损失）
        # if self.training:
        #     # Cross专家负载均衡损失
        #     cross_expert_usage = torch.zeros(self.num_cross_experts, device=rgb.device)
        #     for i in range(B):
        #         if selected_groups[i] == 0:  # 使用Cross组的样本
        #             for j in range(self.num_experts_per_group):
        #                 expert_idx = cross_expert_indices[i, j]
        #                 cross_expert_usage[expert_idx] += 1
        #     if (selected_groups == 0).sum() > 0:
        #         cross_expert_usage = cross_expert_usage / (selected_groups == 0).sum()
        #         cross_router_prob = cross_weights.mean(dim=0)
        #         cross_balance_loss = (cross_expert_usage * cross_router_prob).sum() * self.num_cross_experts
        #     else:
        #         cross_balance_loss = torch.tensor(0.0, device=rgb.device)
            
        #     # CM专家负载均衡损失
        #     cm_expert_usage = torch.zeros(self.num_cm_experts, device=rgb.device)
        #     for i in range(B):
        #         if selected_groups[i] == 1:  # 使用CM组的样本
        #             for j in range(self.num_experts_per_group):
        #                 expert_idx = cm_expert_indices[i, j]
        #                 cm_expert_usage[expert_idx] += 1
        #     if (selected_groups == 1).sum() > 0:
        #         cm_expert_usage = cm_expert_usage / (selected_groups == 1).sum()
        #         cm_router_prob = cm_weights.mean(dim=0)
        #         cm_balance_loss = (cm_expert_usage * cm_router_prob).sum() * self.num_cm_experts
        #     else:
        #         cm_balance_loss = torch.tensor(0.0, device=rgb.device)
            
        #     # # 组负载均衡损失 - 鼓励均匀使用两个组
            # group_balance_loss = F.cross_entropy(group_logits, 
            #                                    torch.ones(B, device=rgb.device, dtype=torch.long) * 0.5)
            
            # aux_loss = cross_balance_loss + cm_balance_loss + group_balance_loss * 0.1
            return final_output#, aux_loss
        # print("final_output.shape:", final_output.shape)
        return final_output#, None



class M_SSM(nn.Module):
    def __init__(self, in_c):
        super(M_SSM, self).__init__()
        self.SS2D_rgb = SS2D(in_c)
        self.SS2D_t = SS2D(in_c)
        self.conv1 = nn.Sequential(nn.Conv2d(2*in_c, in_c, 3, 1, 1),
                                   nn.BatchNorm2d(in_c),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(3*in_c, in_c, 1, 1, 0),
                                   nn.BatchNorm2d(in_c),
                                   nn.ReLU())
    def forward(self, rgb, t):
        B, C, H, W = rgb.shape
        left = self.conv1(torch.cat((rgb, t), dim=1))

        rgb_ = self.SS2D_rgb(rgb.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        t_ = self.SS2D_t(t.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        rgb_ = rgb_ + rgb
        t_ = t_ + t

        out = self.conv2(torch.cat((left, rgb, t), dim=1))
        return out

class Demo1(nn.Module):
    def __init__(self, in_c):
        super(Demo1, self).__init__()
        self.conv1 = BasicConv(in_c * 2, in_c, 3, 1, 1)
        self.conv2 = BasicConv(in_c* 2, in_c, 1, 1, 0)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb, t):
        B, C, H, W = rgb.shape
        path1 = self.conv1(torch.cat((rgb, t), dim=1))
        avg = self.avgpool(torch.cat((rgb, t), dim=1))
        max = self.maxpool(torch.cat((rgb, t), dim=1))
        path2 = torch.mul(torch.cat((rgb, t), dim=1), self.sigmoid(avg+max))
        path2 = self.conv2(path2)
        fusion = path1 + path2
        return fusion



#################################################################################
#                             Basic Layers                                      #
#################################################################################
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class BasicConv(nn.Module):
    def __init__(self, in_c, out_c, k, s, p):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p)
        self.norm = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout2d(0.2)
    def forward(self, x):
        out = self.relu(self.norm(self.conv(x)))
        # out = self.drop(out)
        return out

from torchvision.ops import DeformConv2d
class MultiGranularityAdaptiveAlignment(nn.Module):
    def __init__(self, channels, deform_groups=8, expansion_ratio=0.25):
        super().__init__()
        self.channels = channels
        self.deform_groups = deform_groups
        self.hidden_dim = int(channels * expansion_ratio)
        
        # 原有的网络
        self.offset_net = nn.Sequential(
            nn.Conv2d(channels * 2, self.hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, 2 * deform_groups * 3 * 3, 3, padding=1),
            nn.Tanh()
        )
        
        # 多粒度缩放因子预测
        self.global_scale_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, self.hidden_dim // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim // 2, 1, 1),
            nn.Softplus()
        )
        
        self.channel_scale_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, deform_groups, 1),
            nn.Softplus()
        )
        
        self.spatial_scale_predictor = nn.Sequential(
            nn.Conv2d(channels * 2, self.hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, 1, 3, padding=1),
            nn.Softplus()
        )
        
        self.modulation_net = nn.Sequential(
            nn.Conv2d(channels * 2, self.hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.hidden_dim, deform_groups * 3 * 3, 3, padding=1),
            nn.Sigmoid()
        )
        
        # 权重参数
        self.global_weight = nn.Parameter(torch.tensor(0.33))
        self.channel_weight = nn.Parameter(torch.tensor(0.33))
        self.spatial_weight = nn.Parameter(torch.tensor(0.34))
        
        self.deform_conv = DeformConv2d(
            channels, channels, 3, padding=1, 
            groups=deform_groups
        )
        
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, rgb_feat, thermal_feat):
        B, C, H, W = rgb_feat.shape
        
        offset_input = torch.cat([rgb_feat, thermal_feat], dim=1)
        
        # 1. 预测基础偏移量
        base_offsets = self.offset_net(offset_input)
        
        # 2. 多粒度缩放因子预测
        # 全局缩放因子
        global_scale = self.global_scale_predictor(offset_input).view(B, 1, 1, 1)
        
        # 通道组缩放因子
        channel_scale = self.channel_scale_predictor(offset_input)  # [B, deform_groups, 1, 1]
        channel_scale = channel_scale.view(B, self.deform_groups, 1, 1)
        # 扩展到每个通道组的采样点
        channel_scale_expanded = channel_scale.repeat(1, 1, 3*3, 1).view(B, -1, 1, 1)
        channel_scale_expanded = channel_scale_expanded.repeat(1, 2, H, W)
        
        # 空间缩放因子
        spatial_scale = self.spatial_scale_predictor(offset_input)  # [B, 1, H, W]
        spatial_scale_expanded = spatial_scale.repeat(1, 2 * self.deform_groups * 3 * 3, 1, 1)
        
        # 3. 多粒度融合
        weights = torch.softmax(torch.stack([
            self.global_weight, self.channel_weight, self.spatial_weight
        ]), dim=0)
        
        combined_scale = (
            weights[0] * global_scale + 
            weights[1] * channel_scale_expanded + 
            weights[2] * spatial_scale_expanded
        )
        
        # 4. 应用组合缩放
        offsets = base_offsets * combined_scale
        
        # 5. 预测调制标量
        modulation = self.modulation_net(offset_input)
        modulation = modulation.view(B, self.deform_groups * 3 * 3, H, W)
        
        # 6. 应用可变形卷积
        aligned_rgb = self.deform_conv(rgb_feat, offsets, modulation)
        
        # 7. 自适应融合
        # fusion_weight = torch.sigmoid(self.beta)
        # final_feat = fusion_weight * aligned_rgb + (1 - fusion_weight) * thermal_feat
        
        # return final_feat, combined_scale.mean()
        return aligned_rgb

class DirectOffsetAlignment(nn.Module):
    """
    直接预测偏移场来对齐RGB到红外
    简单高效，专门解决位置对齐问题
    """
    def __init__(self, channels, hidden_dim=64):
        super().__init__()
        self.channels = channels
        
        # 轻量级偏移量预测网络
        self.offset_net = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 2, 3, padding=1),  # 输出2个通道：x,y偏移
            nn.Tanh()  # 限制在[-1,1]范围
        )
        
        # 可学习的最大偏移幅度
        self.max_offset = nn.Parameter(torch.tensor(5.0))
        
    def forward(self, rgb_feat, thermal_feat):
        """
        直接预测每个像素的偏移量，将RGB对齐到红外
        
        参数:
            rgb_feat: RGB特征 [B, C, H, W]
            thermal_feat: 红外特征 [B, C, H, W] (参考)
        
        返回:
            aligned_rgb: 对齐后的RGB特征 [B, C, H, W]
        """
        B, C, H, W = rgb_feat.shape
        
        # 1. 预测归一化的偏移场 [-1, 1]
        offset_input = torch.cat([rgb_feat, thermal_feat], dim=1)
        normalized_offsets = self.offset_net(offset_input)  # [B, 2, H, W]
        
        # 2. 缩放到实际像素偏移
        # max_offset控制最大偏移像素数
        pixel_offsets = normalized_offsets * self.max_offset  # [B, 2, H, W]
        
        # 3. 应用网格采样进行对齐
        aligned_rgb = self.apply_offset(rgb_feat, pixel_offsets, H, W)
        
        return aligned_rgb
    
    def apply_offset(self, rgb_feat, offsets, H, W):
        """应用偏移场进行图像变形"""
        B, C, H, W = rgb_feat.shape
        
        # 创建标准网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=rgb_feat.device),
            torch.linspace(-1, 1, W, device=rgb_feat.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]
        grid = grid.repeat(B, 1, 1, 1)  # [B, H, W, 2]
        
        # 应用偏移
        # 将offsets从[B, 2, H, W]转换为[B, H, W, 2]并归一化到[-1,1]范围
        offsets_normalized = offsets.permute(0, 2, 3, 1) / torch.tensor([W/2, H/2], 
                                                                       device=offsets.device)
        deformed_grid = grid + offsets_normalized
        
        # 双线性采样
        aligned_rgb = F.grid_sample(
            rgb_feat, 
            deformed_grid, 
            mode='bilinear', 
            padding_mode='border',  # 使用边界填充避免边缘问题
            align_corners=True
        )
        
        return aligned_rgb


class FlowBasedAlignment(nn.Module):
    """
    基于光流思想的对齐方法
    预测从RGB到红外的密集光流场
    """
    def __init__(self, channels, flow_layers=3):
        super().__init__()
        self.channels = channels
        
        # 光流预测网络
        self.flow_predictor = nn.Sequential(
            nn.Conv2d(channels * 2, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 3, padding=1),  # 输出光流场 [dx, dy]
        )
        
        # 初始化最后一层为小值，鼓励从零偏移开始学习
        self.flow_predictor[-1].weight.data.zero_()
        self.flow_predictor[-1].bias.data.zero_()
        
    def forward(self, rgb_feat, thermal_feat):
        B, C, H, W = rgb_feat.shape
        
        # 预测光流场
        flow_input = torch.cat([rgb_feat, thermal_feat], dim=1)
        flow = self.flow_predictor(flow_input)  # [B, 2, H, W]
        
        # 应用光流变形
        aligned_rgb = self.warp_with_flow(rgb_feat, flow, H, W)
        
        return aligned_rgb
    
    def warp_with_flow(self, rgb_feat, flow, H, W):
        """使用光流场进行图像变形"""
        B, C, H, W = rgb_feat.shape
        
        # 创建标准网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=rgb_feat.device),
            torch.linspace(-1, 1, W, device=rgb_feat.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]
        grid = grid.repeat(B, 1, 1, 1)  # [B, H, W, 2]
        
        # 将光流归一化到网格坐标空间
        # flow是像素偏移，需要转换为[-1,1]范围的网格偏移
        flow_normalized = flow.permute(0, 2, 3, 1)  # [B, H, W, 2]
        flow_normalized = flow_normalized / torch.tensor([W/2, H/2], 
                                                       device=flow.device)
        
        # 应用光流变形
        deformed_grid = grid + flow_normalized
        
        # 双线性采样
        aligned_rgb = F.grid_sample(
            rgb_feat, 
            deformed_grid, 
            mode='bilinear', 
            padding_mode='zeros',  # 使用零填充
            align_corners=True
        )
        
        return aligned_rgb

class LightweightDeformAlign(nn.Module):
    """
    轻量级可变形卷积对齐
    专门为位置对齐优化的简化版本
    """
    def __init__(self, channels, deform_groups=4):
        super().__init__()
        self.channels = channels
        self.deform_groups = deform_groups
        
        # 极简偏移量预测
        self.offset_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 2 * deform_groups * 3 * 3, 3, padding=1),
        )
        
        # 可变形卷积
        self.deform_conv = DeformConv2d(
            channels, channels, 3, padding=1, 
            groups=deform_groups
        )
        
        # 自动学习偏移量幅度
        self.offset_scale = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, rgb_feat, thermal_feat):
        B, C, H, W = rgb_feat.shape
        
        # 预测偏移量
        offset_input = torch.cat([rgb_feat, thermal_feat], dim=1)
        offsets = self.offset_net(offset_input) * self.offset_scale
        offsets = offsets.view(B, 2 * self.deform_groups * 3 * 3, H, W)
        
        # 应用可变形卷积
        aligned_rgb = self.deform_conv(rgb_feat, offsets)
        
        return aligned_rgb


class SS2D_rgbt(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            deform_groups=8,  # 新增：可变形卷积分组数
            # align_method='flow',# 新增
            align_method='',
            K=4,# 新增
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj1 = nn.Linear(self.d_model, 2*self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj2 = nn.Linear(self.d_model, 2*self.d_inner, bias=bias, **factory_kwargs)
        self.conv2d1 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.conv2d2 = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        # ========== 主要改动1：添加可变形对齐模块 ==========
        # self.deform_align = MultiGranularityAdaptiveAlignment(
        #     channels=self.d_inner,
        #     deform_groups=deform_groups
        # )

        if align_method == 'direct':
            self.align_module = DirectOffsetAlignment(self.d_inner)
        elif align_method == 'flow':
            self.align_module = FlowBasedAlignment(self.d_inner)
        elif align_method == 'deform':
            self.align_module = LightweightDeformAlign(self.d_inner)
        elif align_method == '':
            self.align_module = None;
        else:
            raise ValueError(f"Unknown align method: {align_method}")

        # ========== 主要改动2：添加对角线方向 ==========
        self.K = K
        print("self.K:", self.K)
        self.x_proj = tuple(
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for i in range(self.K)
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = tuple(
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for i in range(self.K)
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.K, merge=True)  # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        if self.K == 4:
            self.forward_core = self.forward_corev0
        else:
            # self.forward_core = self.forward_corev0_mi
            self.forward_core = self.forward_corev0_8

        self.norm1 = nn.LayerNorm(self.d_inner)
        self.norm2 = nn.LayerNorm(self.d_inner)
        self.out_proj1 = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj2 = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D


    def forward_corev0_8(self, rgb: torch.Tensor, t: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = rgb.shape
        L = H * W
        K = 8  # 8个方向

        # 对两个模态分别进行8方向扫描
        rgb_8dir = CrossScan.apply(rgb)  # (B, 8, C, L)
        t_8dir = CrossScan.apply(t)     # (B, 8, C, L)
        
        # 将两个模态在序列维度上交错合并: [rgb1, t1, rgb2, t2, ...]
        x_combined = torch.stack([rgb_8dir, t_8dir], dim=-1)  # (B, 8, C, L, 2)
        x_combined = x_combined.view(B, 8, C, 2 * L)  # (B, 8, C, 2L)

        # 投影处理
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", x_combined, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        # 准备SSM输入
        xs = x_combined.float().view(B, -1, 2 * L)  # (b, 8 * C, 2L)
        dts = dts.contiguous().float().view(B, -1, 2 * L)  # (b, 8 * C, 2L)
        Bs = Bs.float().view(B, K, -1, 2 * L)  # (b, 8, d_state, 2L)
        Cs = Cs.float().view(B, K, -1, 2 * L)  # (b, 8, d_state, 2L)
        Ds = self.Ds.float().view(-1)  # (8 * C)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (8 * C, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (8 * C)

        # 选择性扫描
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, 2 * L)  # (B, 8, C, 2L)
        assert out_y.dtype == torch.float

        # 分离两个模态
        out_combined = out_y.view(B, K, C, 2, L)  # (B, 8, C, 2, L)
        out_rgb = out_combined[:, :, :, 0, :]  # (B, 8, C, L)
        out_t = out_combined[:, :, :, 1, :]    # (B, 8, C, L)

        # 8方向合并
        rgb_out = CrossMerge.apply(out_rgb.view(B, K, C, H, W))  # (B, C, H, W)
        t_out = CrossMerge.apply(out_t.view(B, K, C, H, W))      # (B, C, H, W)

        return rgb_out, t_out
        # return rgb_out+t_out


    

    def forward_corev0(self, rgb: torch.Tensor, t: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = rgb.shape
        L = H * W
        K = 4
        # 核心，进行横向检索与竖向检索
        rgb_hwwh = torch.stack([rgb.view(B, -1, L), torch.transpose(rgb, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        t_hwwh = torch.stack([t.view(B, -1, L), torch.transpose(t, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                               dim=1).view(B, 2, -1, L)
        x_hwwh = torch.stack((rgb_hwwh, t_hwwh), dim=-1).view(B, 2, -1, 2*L)

        # 进行正向检索与反向检索
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, 2*L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, 2*L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, 2*L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, 2*L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, 2*L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, 2*L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, 2*L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, 2*L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, 2*L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, 2*L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, rgb: torch.Tensor, t: torch.Tensor, **kwargs):
        # print("rgb.shape:",rgb.shape)
        # print("t.shape:",t.shape)
        B, H, W, C = rgb.shape
        rgb = self.in_proj1(rgb)
        rgb1, rgb2 = rgb.chunk(2, dim=-1)
        rgb1 = rgb1.permute(0, 3, 1, 2).contiguous()
        rgb1 = self.act(self.conv2d1(rgb1))  # (b, d, h, w)

        t = self.in_proj2(t)
        t1, t2 = t.chunk(2, dim=-1)
        t1 = t1.permute(0, 3, 1, 2).contiguous()
        t1 = self.act(self.conv2d1(t1))

        # ========== 唯一改动：添加位置对齐 ==========
        # aligned_rgb1 = self.deform_align(rgb1, t1)
        if self.align_module:
            aligned_rgb1 = self.align_module(rgb1, t1)  
        else:
            aligned_rgb1 = rgb1

        # self.K=4
        # print("hasattr(self, 'K'):", hasattr(self, 'K'))
        # print("self.K:", self.K)
        if not hasattr(self, 'K') or self.K==4:
            y1, y2, y3, y4 = self.forward_core(aligned_rgb1, t1)
            assert y1.dtype == torch.float32
            y = y1 + y2 + y3 + y4
            # print("y1.shape:",y1.shape,
            # "y2.shape:",y2.shape,
            # "y3.shape:",y3.shape,
            # "y4.shape:",y4.shape)

            # 3. 输出重建和融合
            rgb_out = self._process_output(y, rgb2, self.norm1, self.out_proj1, B, H, W, is_rgb=True)
            t_out = self._process_output(y, t2, self.norm2, self.out_proj2, B, H, W, is_rgb=False)
        else:
            # 8方向核心处理 - 直接返回合并后的结果
            rgb_out, t_out = self.forward_core(aligned_rgb1, t1)

            # 改进的门控融合
            # print("rgb_out.shape:",rgb_out.shape)
            # print("t_out.shape:",t_out.shape)

            rgb_out = rgb_out.view(B, -1, H, W).permute(0, 2, 3, 1).contiguous()
            rgb_out = self.norm1(rgb_out) * F.silu(rgb2)  # 保持门控
            rgb_out = self.out_proj1(rgb_out).permute(0, 3, 1, 2)

            t_out = t_out.view(B, -1, H, W).permute(0, 2, 3, 1).contiguous()
            t_out = self.norm2(t_out) * F.silu(t2)  # 保持门控
            t_out = self.out_proj2(t_out).permute(0, 3, 1, 2)

        

        return rgb_out, t_out

    def _process_output(self, y, feat2, norm, out_proj, B, H, W, is_rgb=True):
        """
        处理输出分支 - 重构为独立方法提高代码复用性
        """
        if is_rgb:
            # RGB分支：取偶数索引的特征
            feat1 = y[:, 0::2, :].permute(0, 2, 1).contiguous().view(B, H, W, -1)
        else:
            # 热成像分支：取奇数索引的特征
            feat1 = y[:, 1::2, :].permute(0, 2, 1).contiguous().view(B, H, W, -1)
        
        # 层归一化 + SiLU门控
        output = norm(feat1) * F.silu(feat2)
        # 输出投影并调整维度
        return out_proj(output).permute(0, 3, 1, 2)

    # def forward_corev0_mi(self, rgb: torch.Tensor, t: torch.Tensor):
    #     """
    #     改进的选择性扫描核心 - 支持米字型8方向扫描，保持交替融合方式
    #     """
    #     self.selective_scan = selective_scan_fn

    #     B, C, H, W = rgb.shape
    #     L = H * W
    #     K = self.K
        
    #     print(f"输入: rgb.shape: {rgb.shape}, t.shape: {t.shape}")
        
    #     # 8方向米字型扫描
    #     # 方向1-2: 水平方向
    #     rgb_h = rgb.view(B, -1, L)  # [B, C, L]
    #     t_h = t.view(B, -1, L)      # [B, C, L]
        
    #     # 方向3-4: 垂直方向  
    #     rgb_v = torch.transpose(rgb, dim0=2, dim1=3).contiguous().view(B, -1, L)  # [B, C, L]
    #     t_v = torch.transpose(t, dim0=2, dim1=3).contiguous().view(B, -1, L)      # [B, C, L]
        
    #     # 方向5-6: 主对角线方向
    #     rgb_diag1 = self.extract_diagonal_fixed(rgb, H, W, direction='main')  # [B, C, L]
    #     t_diag1 = self.extract_diagonal_fixed(t, H, W, direction='main')      # [B, C, L]
        
    #     # 方向7-8: 副对角线方向
    #     rgb_diag2 = self.extract_diagonal_fixed(rgb, H, W, direction='anti')  # [B, C, L]
    #     t_diag2 = self.extract_diagonal_fixed(t, H, W, direction='anti')      # [B, C, L]
        
    #     print(f"序列提取后:")
    #     print(f"rgb_h: {rgb_h.shape}, t_h: {t_h.shape}")
    #     print(f"rgb_v: {rgb_v.shape}, t_v: {t_v.shape}") 
    #     print(f"rgb_diag1: {rgb_diag1.shape}, t_diag1: {t_diag1.shape}")
    #     print(f"rgb_diag2: {rgb_diag2.shape}, t_diag2: {t_diag2.shape}")
        
    #     # 关键修改：按照交替方式合并两个模态的序列
    #     # 对于每个方向，将RGB和红外序列交替合并为 a1,b1,a2,b2,...,aL,bL
    #     def interleave_sequences(rgb_seq, t_seq):
    #         # rgb_seq: [B, C, L], t_seq: [B, C, L]
    #         # 在序列维度交替合并
    #         interleaved = torch.stack([rgb_seq, t_seq], dim=-1)  # [B, C, L, 2]
    #         interleaved = interleaved.view(B, C, 2*L)  # [B, C, 2*L] - 现在是 a1,b1,a2,b2,...
    #         return interleaved
        
    #     # 对每个方向应用交替合并
    #     h_interleaved = interleave_sequences(rgb_h, t_h)  # [B, C, 2*L]
    #     v_interleaved = interleave_sequences(rgb_v, t_v)  # [B, C, 2*L]
    #     diag1_interleaved = interleave_sequences(rgb_diag1, t_diag1)  # [B, C, 2*L]
    #     diag2_interleaved = interleave_sequences(rgb_diag2, t_diag2)  # [B, C, 2*L]
        
    #     # 合并所有方向
    #     x_sequences = torch.stack([h_interleaved, v_interleaved, diag1_interleaved, diag2_interleaved], dim=1)  # [B, 4, C, 2*L]
        
    #     print(f"交替合并后: x_sequences: {x_sequences.shape}")
        
    #     # 添加正向和反向扫描
    #     x_forward = x_sequences
    #     x_backward = torch.flip(x_sequences, dims=[-1])  # 反向扫描
    #     xs = torch.cat([x_forward, x_backward], dim=1)  # [B, 8, C, 2*L]
        
    #     print(f"最终输入SSM: xs: {xs.shape}")

    #     # 投影得到SSM参数
    #     x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, 2*L), self.x_proj_weight)
    #     dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
    #     dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, 2*L), self.dt_projs_weight)

    #     # 准备SSM参数
    #     xs = xs.float().view(B, -1, 2*L)  # [B, K*C, 2*L]
    #     dts = dts.contiguous().float().view(B, -1, 2*L)
    #     Bs = Bs.float().view(B, K, -1, 2*L)
    #     Cs = Cs.float().view(B, K, -1, 2*L)
    #     Ds = self.Ds.float().view(-1)
    #     As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
    #     dt_projs_bias = self.dt_projs_bias.float().view(-1)

    #     # 执行选择性扫描
    #     out_y = self.selective_scan(
    #         xs, dts, As, Bs, Cs, Ds, z=None,
    #         delta_bias=dt_projs_bias,
    #         delta_softplus=True,
    #         return_last_state=False,
    #     ).view(B, K, -1, 2*L)  # [B, K, C, 2*L]
        
    #     print(f"SSM输出: out_y: {out_y.shape}")

    #     # 8方向重建逻辑
    #     forward_y = out_y[:, :4]  # [B, 4, C, 2*L]
    #     backward_y = out_y[:, 4:] # [B, 4, C, 2*L]
        
    #     # 提取各个方向
    #     y_h = forward_y[:, 0]     # [B, C, 2*L]
    #     y_v = forward_y[:, 1]     # [B, C, 2*L]
    #     y_diag1 = forward_y[:, 2] # [B, C, 2*L]
    #     y_diag2 = forward_y[:, 3] # [B, C, 2*L]
        
    #     y_h_inv = backward_y[:, 0] # [B, C, 2*L]
    #     y_v_inv = backward_y[:, 1] # [B, C, 2*L]
    #     y_diag1_inv = backward_y[:, 2] # [B, C, 2*L]
    #     y_diag2_inv = backward_y[:, 3] # [B, C, 2*L]
        
    #     print(f"各方向输出:")
    #     print(f"y_h: {y_h.shape}, y_v: {y_v.shape}")
    #     print(f"y_diag1: {y_diag1.shape}, y_diag2: {y_diag2.shape}")
        
    #     # 重建空间特征 - 现在需要将交替序列分离并分别重建
    #     def deinterleave_and_reconstruct(interleaved_seq, H, W, is_diagonal=False, diag_direction=None):
    #         # interleaved_seq: [B, C, 2*L] - 交替序列 a1,b1,a2,b2,...
    #         B, C, seq_len = interleaved_seq.shape
            
    #         # 分离RGB和红外序列
    #         rgb_seq = interleaved_seq[:, :, 0::2]  # 取偶数索引: a1,a2,...,aL
    #         t_seq = interleaved_seq[:, :, 1::2]    # 取奇数索引: b1,b2,...,bL
            
    #         if is_diagonal:
    #             # 对角线方向需要特殊重建
    #             rgb_spatial = self.reconstruct_diagonal_fixed(rgb_seq, H, W, direction=diag_direction)
    #             t_spatial = self.reconstruct_diagonal_fixed(t_seq, H, W, direction=diag_direction)
    #         else:
    #             # 水平和垂直方向直接reshape
    #             rgb_spatial = rgb_seq.view(B, C, H, W)
    #             t_spatial = t_seq.view(B, C, H, W)
                
    #             # 垂直方向需要转置
    #             if diag_direction == 'vertical':
    #                 rgb_spatial = torch.transpose(rgb_spatial, dim0=2, dim1=3).contiguous()
    #                 t_spatial = torch.transpose(t_spatial, dim0=2, dim1=3).contiguous()
            
    #         return rgb_spatial, t_spatial
        
    #     # 重建各个方向的空间特征
    #     y_h_rgb, y_h_t = deinterleave_and_reconstruct(y_h, H, W, is_diagonal=False, diag_direction='horizontal')
    #     y_v_rgb, y_v_t = deinterleave_and_reconstruct(y_v, H, W, is_diagonal=False, diag_direction='vertical')
    #     y_diag1_rgb, y_diag1_t = deinterleave_and_reconstruct(y_diag1, H, W, is_diagonal=True, diag_direction='main')
    #     y_diag2_rgb, y_diag2_t = deinterleave_and_reconstruct(y_diag2, H, W, is_diagonal=True, diag_direction='anti')
        
    #     y_h_inv_rgb, y_h_inv_t = deinterleave_and_reconstruct(y_h_inv, H, W, is_diagonal=False, diag_direction='horizontal')
    #     y_v_inv_rgb, y_v_inv_t = deinterleave_and_reconstruct(y_v_inv, H, W, is_diagonal=False, diag_direction='vertical')
    #     y_diag1_inv_rgb, y_diag1_inv_t = deinterleave_and_reconstruct(y_diag1_inv, H, W, is_diagonal=True, diag_direction='main')
    #     y_diag2_inv_rgb, y_diag2_inv_t = deinterleave_and_reconstruct(y_diag2_inv, H, W, is_diagonal=True, diag_direction='anti')
        
    #     # 合并RGB和红外特征
    #     y_h_combined = torch.cat([y_h_rgb, y_h_t], dim=1)  # [B, 2*C, H, W]
    #     y_v_combined = torch.cat([y_v_rgb, y_v_t], dim=1)  # [B, 2*C, H, W]
    #     y_diag1_combined = torch.cat([y_diag1_rgb, y_diag1_t], dim=1)  # [B, 2*C, H, W]
    #     y_diag2_combined = torch.cat([y_diag2_rgb, y_diag2_t], dim=1)  # [B, 2*C, H, W]
        
    #     y_h_inv_combined = torch.cat([y_h_inv_rgb, y_h_inv_t], dim=1)  # [B, 2*C, H, W]
    #     y_v_inv_combined = torch.cat([y_v_inv_rgb, y_v_inv_t], dim=1)  # [B, 2*C, H, W]
    #     y_diag1_inv_combined = torch.cat([y_diag1_inv_rgb, y_diag1_inv_t], dim=1)  # [B, 2*C, H, W]
    #     y_diag2_inv_combined = torch.cat([y_diag2_inv_rgb, y_diag2_inv_t], dim=1)  # [B, 2*C, H, W]
        
    #     print(f"重建后:")
    #     print(f"y_h_combined: {y_h_combined.shape}, y_v_combined: {y_v_combined.shape}")
    #     print(f"y_diag1_combined: {y_diag1_combined.shape}, y_diag2_combined: {y_diag2_combined.shape}")
        
    #     # 展平为序列格式返回
    #     return (
    #         y_h_combined.view(B, -1, L),           # 水平正向
    #         y_v_combined.view(B, -1, L),           # 垂直正向  
    #         y_diag1_combined.view(B, -1, L),       # 主对角线正向
    #         y_diag2_combined.view(B, -1, L),       # 副对角线正向
    #         y_h_inv_combined.view(B, -1, L),       # 水平反向
    #         y_v_inv_combined.view(B, -1, L),       # 垂直反向
    #         y_diag1_inv_combined.view(B, -1, L),   # 主对角线反向
    #         y_diag2_inv_combined.view(B, -1, L)    # 副对角线反向
    #     )
    # def extract_diagonal_fixed(self, x: torch.Tensor, H: int, W: int, direction: str = 'main'):
    #     """
    #     修复的对角线序列提取 - 确保输出维度与水平/垂直方向一致
    #     """
    #     B, C, H, W = x.shape
        
    #     if direction == 'main':
    #         # 主对角线: 从左上到右下
    #         # 创建与输入相同维度的输出
    #         output_sequences = []
    #         for b in range(B):
    #             batch_seqs = []
    #             for c in range(C):
    #                 # 提取主对角线序列
    #                 diag_seq = []
    #                 for i in range(-H + 1, W):
    #                     diag = torch.diagonal(x[b, c], offset=i)
    #                     diag_seq.append(diag)
    #                 # 拼接并填充到长度L
    #                 full_seq = torch.cat(diag_seq)
    #                 # 如果序列长度不等于L，进行填充或截断
    #                 if full_seq.numel() < H * W:
    #                     # 填充到L
    #                     padding = torch.zeros(H * W - full_seq.numel(), device=x.device)
    #                     full_seq = torch.cat([full_seq, padding])
    #                 elif full_seq.numel() > H * W:
    #                     # 截断到L
    #                     full_seq = full_seq[:H * W]
    #                 batch_seqs.append(full_seq)
    #             output_sequences.append(torch.stack(batch_seqs))
    #         return torch.stack(output_sequences)  # [B, C, L]
        
    #     else:  # direction == 'anti'
    #         # 副对角线: 从右上到左下
    #         output_sequences = []
    #         for b in range(B):
    #             batch_seqs = []
    #             for c in range(C):
    #                 # 水平翻转后提取主对角线
    #                 flipped = torch.flip(x[b, c], dims=[1])
    #                 diag_seq = []
    #                 for i in range(-H + 1, W):
    #                     diag = torch.diagonal(flipped, offset=i)
    #                     diag_seq.append(diag)
    #                 full_seq = torch.cat(diag_seq)
    #                 if full_seq.numel() < H * W:
    #                     padding = torch.zeros(H * W - full_seq.numel(), device=x.device)
    #                     full_seq = torch.cat([full_seq, padding])
    #                 elif full_seq.numel() > H * W:
    #                     full_seq = full_seq[:H * W]
    #                 batch_seqs.append(full_seq)
    #             output_sequences.append(torch.stack(batch_seqs))
    #         return torch.stack(output_sequences)  # [B, C, L]

    # def reconstruct_diagonal_fixed(self, diag_seq: torch.Tensor, H: int, W: int, direction: str = 'main'):
    #     """
    #     修复的对角线序列重建 - 确保输出通道数与输入一致
    #     """
    #     B, C, L = diag_seq.shape
        
    #     # 创建空的输出特征图
    #     output = torch.zeros(B, C, H, W, device=diag_seq.device, dtype=diag_seq.dtype)
        
    #     # 重建逻辑保持不变，但确保处理所有C个通道
    #     if direction == 'main':
    #         start_idx = 0
    #         for i in range(-H + 1, W):
    #             diag_len = min(H, W, H + i, W - i)
    #             if start_idx + diag_len > L:
    #                 break
                    
    #             diag_vals = diag_seq[:, :, start_idx:start_idx + diag_len]  # [B, C, diag_len]
    #             start_idx += diag_len
                
    #             for b in range(B):
    #                 for c in range(C):
    #                     for j in range(diag_len):
    #                         row = max(0, -i) + j if i < 0 else j
    #                         col = max(0, i) + j if i >= 0 else j
    #                         if row < H and col < W:
    #                             output[b, c, row, col] = diag_vals[b, c, j]
        
    #     else:  # direction == 'anti'
    #         start_idx = 0
    #         temp_output = torch.zeros(B, C, H, W, device=diag_seq.device, dtype=diag_seq.dtype)
            
    #         for i in range(-H + 1, W):
    #             diag_len = min(H, W, H + i, W - i)
    #             if start_idx + diag_len > L:
    #                 break
                    
    #             diag_vals = diag_seq[:, :, start_idx:start_idx + diag_len]
    #             start_idx += diag_len
                
    #             for b in range(B):
    #                 for c in range(C):
    #                     for j in range(diag_len):
    #                         row = max(0, -i) + j if i < 0 else j
    #                         col = max(0, i) + j if i >= 0 else j
    #                         if row < H and col < W:
    #                             temp_output[b, c, row, col] = diag_vals[b, c, j]
            
    #         # 水平翻转得到副对角线
    #         output = torch.flip(temp_output, dims=[3])
        
    #     return output

#################################################################################
#                             Basic Functions                                   #
#################################################################################

def channel_shuffle(x: Tensor, groups: int) -> Tensor:

    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x

#################################################################################
#                             Comparison Methods                                #
#################################################################################

class MDFusion(nn.Module):
    def __init__(self, channels):
        super(MDFusion, self).__init__()

        # self.mca1 = SS_Conv_SSM(channels[0] // 4)
        # self.mca2 = SS_Conv_SSM(channels[1] // 4)
        # self.mca3 = SS_Conv_SSM(channels[2] // 4)
        # self.mca4 = SS_Conv_SSM(channels[3] // 4)

        # self.dct1 = MultiSpectralAttentionLayer(channels[0] // 4, shape[0] // 4, shape[1] // 4)
        # self.dct2 = MultiSpectralAttentionLayer(channels[1] // 4, shape[0] // 8, shape[0] // 8)
        # self.dct3 = MultiSpectralAttentionLayer(channels[2] // 4, shape[0] // 16, shape[0] // 16)
        # self.dct4 = MultiSpectralAttentionLayer(channels[3] // 4, shape[0] // 32, shape[0] // 32)

        # self.mca1t = SS_Conv_SSM(channels[0] // 4)
        # self.mca2t = SS_Conv_SSM(channels[1] // 4)
        # self.mca3t = SS_Conv_SSM(channels[2] // 4)
        # self.mca4t = SS_Conv_SSM(channels[3] // 4)

        self.lpr1 = nn.Conv2d(channels[0], channels[0]//4, 1, 1, 0)
        self.lpr2 = nn.Conv2d(channels[1], channels[1]//4, 1, 1, 0)
        self.lpr3 = nn.Conv2d(channels[2], channels[2]//4, 1, 1, 0)
        self.lpr4 = nn.Conv2d(channels[3], channels[3]//4, 1, 1, 0)

        self.lpt1 = nn.Conv2d(channels[0], channels[0]//4, 1, 1, 0)
        self.lpt2 = nn.Conv2d(channels[1], channels[1]//4, 1, 1, 0)
        self.lpt3 = nn.Conv2d(channels[2], channels[2]//4, 1, 1, 0)
        self.lpt4 = nn.Conv2d(channels[3], channels[3]//4, 1, 1, 0)

        self.lp1 = nn.Conv2d(channels[0] // 2, channels[0]//4, 1, 1, 0)
        self.lp2 = nn.Conv2d(channels[1] // 2, channels[1]//4, 1, 1, 0)
        self.lp3 = nn.Conv2d(channels[2] // 2, channels[2]//4, 1, 1, 0)
        self.lp4 = nn.Conv2d(channels[3] // 2, channels[3]//4, 1, 1, 0)

    def forward(self, x, thermal):

        x1_1a, x1_1b = self.mca1(self.lpr1(x[0]).permute(0,2,3,1)).permute(0,3,1,2), self.mca1t(self.lpt1(thermal[0]).permute(0,2,3,1)).permute(0,3,1,2)
        x2_1a, x2_1b = self.mca2(self.lpr2(x[1]).permute(0,2,3,1)).permute(0,3,1,2), self.mca2t(self.lpt2(thermal[1]).permute(0,2,3,1)).permute(0,3,1,2)
        x3_1a, x3_1b = self.mca3(self.lpr3(x[2]).permute(0,2,3,1)).permute(0,3,1,2), self.mca3t(self.lpt3(thermal[2]).permute(0,2,3,1)).permute(0,3,1,2)
        x4_1a, x4_1b = self.mca4(self.lpr4(x[3]).permute(0,2,3,1)).permute(0,3,1,2), self.mca4t(self.lpt4(thermal[3]).permute(0,2,3,1)).permute(0,3,1,2)

        x1_1lp = torch.nn.Sigmoid()(self.lp1(torch.cat([x1_1a, x1_1b], dim=1)))
        x1_1 = x1_1a*x1_1lp+x1_1b*(1-x1_1lp)


        x2_1lp = torch.nn.Sigmoid()(self.lp2(torch.cat([x2_1a, x2_1b], dim=1)))
        x2_1 = x2_1a * x2_1lp + x2_1b * (1 - x2_1lp)


        x3_1lp = torch.nn.Sigmoid()(self.lp3(torch.cat([x3_1a, x3_1b], dim=1)))
        x3_1 = x3_1a * x3_1lp + x3_1b * (1 - x3_1lp)


        x4_1lp = torch.nn.Sigmoid()(self.lp4(torch.cat([x4_1a, x4_1b], dim=1)))
        x4_1 = x4_1a * x4_1lp + x4_1b * (1 - x4_1lp)

        return x1_1, x2_1, x3_1, x4_1