import torch
import torch.nn as nn
from .Efficientvit import Encoder_Efficientvit
from .Efficientvit import Encoder_RGBT_Efficientvit
# from models.decoder.MLP import Decoder_MLP
# from models.decoder.MLP_plus import Decoder_MLP_plus
# from models.decoder.MLP_antiUAV import Detector_AntiUAV
import torch.nn.functional as F
# from proposed.fuison_strategy.base_fusion import Fusion_Module
from .fusion import SS2D_rgbt #Fusion_Module, RGBAdjuster
# from models.decoder.DeepLabV3 import DeepLabHeadV3Plus

"""
class CMSSM(nn.Module):
    def __init__(self, mode='b1', inputs='rgbt', n_class=1, fusion_mode='CM-SSM', norm_fuse=nn.BatchNorm2d):
        super(CMSSM, self).__init__()
        self.fusion_mode = fusion_mode
        if mode == 'b0':
            channels = [16, 32, 64, 128]
            emb_c = 128
        elif mode == 'b1':
            channels = [32, 64, 128, 256]
            emb_c = 256
        elif mode == 'b2':
            channels = [48, 96, 192, 384]
            emb_c = 256
        elif mode in ['b3', 'l1', 'l2']:
            channels = [64, 128, 256, 512]
            emb_c = 768
        self.inputs = inputs
        if inputs == 'unimodal':
            self.encoder = Encoder_Efficientvit(mode=mode)
        if inputs == 'rgbt':
            self.encoder = Encoder_RGBT_Efficientvit(mode=mode)
            if 'rgbCenter' in fusion_mode:
                self.center_predictor = CenterPredictor(channels=channels[1:])
            if 'offset' in fusion_mode:
                self.adjuster = RGBAdjuster(channels=channels[1:])
            self.fusion_module = Fusion_Module(fusion_mode=fusion_mode, channels=channels)
        # self.decoder = Detector_AntiUAV(in_channels=channels, embed_dim=emb_c, num_classes=n_class)
        # self.decoder = Decoder_MLP(in_channels=channels, embed_dim=emb_c, num_classes=n_class)
        # self.decoder = DeepLabHeadV3Plus(in_channels=channels[-1], low_level_channels=channels[0], num_classes=12)

    def forward(self, x):
        rgb = x[0]
        t = x[1]
        # print("rgb.shape",rgb.shape)#([1, 3, 256, 256])
        # print("t.shape",t.shape)#([1, 3, 256, 256])
        if t == None:
            t = rgb
        if self.inputs == 'unimodal':
            fusions = self.encoder(t)
            # print("fusion.shape",fusions[0].shape)#([1, 32, 64, 64])
            # print("fusion.shape",fusions[1].shape)#([1, 64, 32, 32])
            # print("fusion.shape",fusions[2].shape)#([1, 128, 16, 16])
            # print("fusion.shape",fusions[3].shape)#([1, 256, 8, 8])
            return fusions[1:],None,None
        else:
            f_rgb, f_t = self.encoder(rgb, t)
            if 'rgbCenter' in self.fusion_mode:
                center_points = self.center_predictor(f_rgb[1:])
            if 'offset' in self.fusion_mode:
                f_rgb, f_t, txtys = self.adjuster(f_rgb[1:], f_t[1:])
                # print("f_rgb.shape",f_rgb[0].shape)#([1, 32, 64, 64])
                # print("f_rgb.shape",f_rgb[1].shape)#([1, 64, 32, 32])
                # print("f_rgb.shape",f_rgb[2].shape)#([1, 128, 16, 16])
                # print("f_rgb.shape",f_rgb[3].shape)#([1, 256, 8, 8])
                fusions = self.fusion_module(f_rgb, f_t)
                # if not self.training:
                #     print("txtys111111111:",txtys)
                return fusions,txtys,center_points
            else:
                fusions = self.fusion_module(f_rgb[1:], f_t[1:])
                return fusions,None,None
        # print("fusion.shape",fusions[0].shape)#([1, 32, 64, 64])
        # print("fusion.shape",fusions[1].shape)#([1, 64, 32, 32])
        # print("fusion.shape",fusions[2].shape)#([1, 128, 16, 16])
        # print("fusion.shape",fusions[3].shape)#([1, 256, 8, 8])

        # print("len(txtys):", len(txtys))
        # print("txtys:", txtys)
        # print("txtys[0].shape:", txtys[0].shape) #[B,2]
        # print("txtys[1].shape:", txtys[1].shape)
        # print("txtys[2].shape:", txtys[2].shape)
        
class CenterPredictor(nn.Module):
    def __init__(self, channels=[64, 128, 256]):
        super().__init__()
        self.predict_heads = nn.ModuleList()  # 每层的局部非线性像素级预测头
        
        for in_ch in channels:
            self.predict_heads.append(
                    nn.Sequential(
                    nn.Conv2d(in_ch, in_ch, 3, padding=1),
                    nn.BatchNorm2d(in_ch),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(in_ch, 2),
                    nn.sigmoid()
                )
            )
   

    def forward(self, rgb):
        center_points = []
        for i in range(len(rgb)):
            xy = self.predict_heads[i](rgb[i])
            center_points.append(xy)
        
        return center_points
"""

"""
class CRF(nn.Module):
    def __init__(self, channel, use_centerPoint=True, use_RGBAdjuster=True, n_class=1, norm_fuse=nn.BatchNorm2d):
        super(CRF, self).__init__()

        self.use_centerPoint = use_centerPoint
        self.use_RGBAdjuster = use_RGBAdjuster

        if self.use_centerPoint:
            self.center_predictor = CenterPredictor(channels=channel)
        if self.use_RGBAdjuster:
            self.adjuster = RGBAdjuster(channels=channel)
        self.fusion_module = Fusion_Module(channel=channel)

    def forward(self, x):
        rgb = x[0]
        t = x[1]
        print("rgb.shape",rgb.shape)#([1, 3, 256, 256])
        print("t.shape",t.shape)#([1, 3, 256, 256])

        center_point = None
        txty = None
        if self.use_centerPoint:
            center_point = self.center_predictor(rgb)
        if self.use_RGBAdjuster:
            f_rgb, f_t, txty = self.adjuster(f_rgb, f_t)
            # print("f_rgb.shape",f_rgb[0].shape)#([1, 32, 64, 64])
            # print("f_rgb.shape",f_rgb[1].shape)#([1, 64, 32, 32])
            # print("f_rgb.shape",f_rgb[2].shape)#([1, 128, 16, 16])
            # print("f_rgb.shape",f_rgb[3].shape)#([1, 256, 8, 8])
        fusion = self.fusion_module(f_rgb, f_t)
            # if not self.training:
            #     print("txtys111111111:",txtys)
        return fusion,txty,center_point

        # print("fusion.shape",fusions[0].shape)#([1, 32, 64, 64])
        # print("fusion.shape",fusions[1].shape)#([1, 64, 32, 32])
        # print("fusion.shape",fusions[2].shape)#([1, 128, 16, 16])
        # print("fusion.shape",fusions[3].shape)#([1, 256, 8, 8])

        # print("len(txtys):", len(txtys))
        # print("txtys:", txtys)
        # print("txtys[0].shape:", txtys[0].shape) #[B,2]
        # print("txtys[1].shape:", txtys[1].shape)
        # print("txtys[2].shape:", txtys[2].shape)
"""


class CenterPredictor(nn.Module):
    def __init__(self, channel=64):
        super().__init__()

        self.predict_heads= nn.Sequential(
                nn.Conv2d(channel, channel, 3, padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channel, 2),
                nn.Sigmoid()
            ) 

    def forward(self, rgb):
        return self.predict_heads(rgb)

class ExistPredictor(nn.Module):
    def __init__(self, in_ch=64):
        super().__init__()
        
        # 为每层创建仿射预测头（带物理约束）
        self.exist_heads=nn.Sequential(
                nn.Conv2d(in_ch, in_ch, 3, padding=1),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_ch, 1),
                nn.Sigmoid()  
            )
    def forward(self, rgb):   
        exist = self.exist_heads(rgb)  # [B,1]
         
        return exist  # 
   

class Fusion_Module(nn.Module):
    def __init__(self, channel, norm_fuse=nn.BatchNorm2d):
        super(Fusion_Module, self).__init__()
        self.norm_layer = nn.ModuleList()
        self.fusion = CM_SSM(channel)
        self.channel = channel

        # if self.fusion_mode == 'MoE':
        #     # print("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")
        #     for channel in channels:
        #         self.fusion.append(FusionMoE(channel))

    def forward(self, x):
        rgb, t = x[0], x[1]
        # print("x[0].shape:", x[0].shape)#([1, 128, 32, 32])([1, 128, 16, 16])([1, 256, 8, 8]) ，Hyper-yolo[1, 64, 32, 32]，[1, 128, 16, 16]，[1, 256, 8, 8]
        # print("x[1].shape:", x[1].shape) #rtdetr ([1, 512, 80, 80])[1, 1024, 40, 40][1, 2048, 20, 20]  picodet([1, 72, 64, 64]),([1, 144, 32, 32]),([1, 288, 16, 16])
        # print("self.channel:", self.channel)
        # return rgb + t
        return torch.cat(x, 1)

        return self.fusion(rgb, t)

class CM_SSM(nn.Module):
    def __init__(self, in_c):
        super(CM_SSM, self).__init__()
        self.SS2D = SS2D_rgbt(in_c)
        self.conv1 = nn.Sequential(nn.Conv2d(2*in_c, in_c, 3, 1, 1),
                                   nn.BatchNorm2d(in_c),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(3*in_c, 2*in_c, 1, 1, 0),
                                   nn.BatchNorm2d(2*in_c),
                                   nn.ReLU())

    def forward(self, rgb, t):
        B, C, H, W = rgb.shape
        left = self.conv1(torch.cat((rgb, t), dim=1))

        rgb_, t_ = self.SS2D(rgb.permute(0, 2, 3, 1), t.permute(0, 2, 3, 1))

        rgb_ = rgb_ + rgb
        t_ = t_ + t

        out = self.conv2(torch.cat((left, rgb_, t_), dim=1))
        # print("out.shape:", out.shape)
        return out

##全局仿射变换预测

import torch
import torch.nn as nn
import torch.nn.functional as F

class AffinePredictor(nn.Module):
    def __init__(self, in_ch=64, kernel_size=3, padding=1):
        super().__init__()
        
        # 仿射预测头：保持原有高效特征提取结构，新增Dropout提升泛化性
        self.affine_heads = nn.Sequential(
            nn.Conv2d(2 * in_ch, 2 * in_ch, kernel_size, padding=padding),
            nn.BatchNorm2d(2 * in_ch),
            nn.ReLU(inplace=True),  # inplace=True 节省内存
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.1),  # 新增：轻微Dropout防止过拟合，不影响特征表达
            nn.Linear(2 * in_ch, 6)  # 2*in_ch: IR+加权RGB融合后通道数
        )
        
        # **物理先验约束参数（可学习，优化后参数范围更合理）**
        # a/d：缩放/旋转参数，约束在 [0.8, 1.2]（1.0上下对称波动，支持放大/缩小）
        self.scale_a = nn.Parameter(torch.tensor([0.2]))
        self.scale_d = nn.Parameter(torch.tensor([0.2]))
        # b/c：剪切参数，约束在 [-0.05, 0.05]（微小剪切，避免特征失真）
        self.scale_b = nn.Parameter(torch.tensor([0.05]))
        self.scale_c = nn.Parameter(torch.tensor([0.05]))
        # tx/ty：平移参数，约束在 [-1, 1]（适配F.affine_grid的尺度要求，可按需调整scale）
        # 若仅需小幅平移，可将scale_tx/ty改为0.2，约束在 [-0.2, 0.2]
        self.scale_tx = nn.Parameter(torch.tensor([-0.2]))
        self.scale_ty = nn.Parameter(torch.tensor([-0.2]))
        # 偏置参数：初始为恒等变换参数，保证训练初期稳定
        self.bias_a = nn.Parameter(torch.tensor([1.0]))
        self.bias_d = nn.Parameter(torch.tensor([1.0]))
        self.bias_b = nn.Parameter(torch.tensor([0.0]))
        self.bias_c = nn.Parameter(torch.tensor([0.0]))
        self.bias_tx = nn.Parameter(torch.tensor([0.0]))
        self.bias_ty = nn.Parameter(torch.tensor([0.0]))

        # 恒等变换初始化：保留原有合理设计，避免训练初期几何变形
        last_linear = None
        for m in self.affine_heads.modules():
            if isinstance(m, nn.Linear):
                last_linear = m

        if last_linear is not None:
            # 权重初始化为微小扰动，避免初始参数波动过大
            nn.init.normal_(last_linear.weight, mean=0.0, std=1e-4)
            # bias初始化为恒等变换：[a=1, b=0, c=0, d=1, tx=0, ty=0]
            last_linear.bias.data = torch.tensor(
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                dtype=last_linear.bias.dtype,
                device=last_linear.bias.device  # 新增：指定设备，避免多GPU/CPU不兼容
            )
    
    def forward(self, fused):      
        # 提取原始6维仿射参数
        affine_params_raw = self.affine_heads(fused)  # [B, 6]
        
        # 应用物理先验约束（优化后：参数范围更合理，计算更严谨）
        # 1. a/d：sigmoid约束为正，且通过计算实现1.0上下对称波动（[0.8, 1.2]）
        a = torch.sigmoid(affine_params_raw[:, 0]) * 2 * self.scale_a + (self.bias_a - self.scale_a)
        d = torch.sigmoid(affine_params_raw[:, 3]) * 2 * self.scale_d + (self.bias_d - self.scale_d)
        
        # 2. b/c：tanh约束为对称小范围，避免严重剪切变形
        b = torch.tanh(affine_params_raw[:, 1]) * self.scale_b + self.bias_b
        c = torch.tanh(affine_params_raw[:, 2]) * self.scale_c + self.bias_c
        
        # 3. tx/ty：tanh约束在[-1, 1]（适配F.affine_grid），支持正负平移
        tx = torch.tanh(affine_params_raw[:, 4]) * self.scale_tx + self.bias_tx
        ty = torch.tanh(affine_params_raw[:, 5]) * self.scale_ty + self.bias_ty
        
        # 拼接成最终6维仿射参数 [B, 6]
        affine_params = torch.stack([a, b, c, d, tx, ty], dim=1)
    
        return affine_params

        # return affine_params_raw

# ##局部非线性像素级微调（“取值相对宽/高比例”）
# class LocalOffsetPredictor(nn.Module):
#     def __init__(self, in_ch):
#         super(LocalOffsetPredictor, self).__init__()
#         self.conv_offset = nn.Conv2d(in_ch*2, 2, kernel_size=3, padding=1)  # 输出两个通道，分别对应δx和δy
        
#     def forward(self, x):
#         offset = self.conv_offset(x)  # (N, 2, H, W)
#         offset = torch.tanh(offset) * 0.01  # 将偏移量限制在 [-0.01, 0.01]
#         return offset

from torchvision.ops import DeformConv2d
"""
class LocalOffsetPredictor(nn.Module):
    def __init__(self, in_ch, kernel_size=3, offset_scale=0.01):
        super().__init__()
        self.kernel_size = kernel_size
        self.offset_scale = offset_scale
        k2 = kernel_size * kernel_size

        # ### 修改行：DCNv2 offset+mask 预测头（输出 2k² + k²）
        self.offset_mask_conv = nn.Conv2d(in_ch * 2, 3 * k2, kernel_size=3, padding=1)

        # ### 修改行：真正执行几何变形的 DCN
        self.deform_conv = DeformConv2d(in_ch, in_ch, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.norm = nn.BatchNorm2d(in_ch)
        self.act = nn.ReLU(inplace=True)

        # ### 修改行：初始化为 0，使初始等价“无偏移”
        nn.init.constant_(self.offset_mask_conv.weight, 0.)
        nn.init.constant_(self.offset_mask_conv.bias, 0.)

    def predict(self, x_fused):

        out = self.offset_mask_conv(x_fused)
        k2 = self.kernel_size * self.kernel_size

        # ### 修改行：拆分 offset 和 mask
        offset = out[:, :2 * k2, :, :]
        mask   = out[:, 2 * k2:, :, :]

        # ### 修改行：数值约束（与你原 tanh*0.01 一致，但扩展到 2k² 通道）
        offset = torch.tanh(offset) * self.offset_scale
        mask   = torch.sigmoid(mask)
        return offset, mask

    @staticmethod
    def _resize_offset_mask(offset, mask, target_hw):

        B, C_off, H0, W0 = offset.shape
        H1, W1 = target_hw

        # ### 修改行：插值到目标尺寸
        offset_t = F.interpolate(offset, size=(H1, W1), mode="bilinear", align_corners=False)
        mask_t   = F.interpolate(mask,   size=(H1, W1), mode="bilinear", align_corners=False)

        # ### 修改行：dx/dy 缩放（非常关键，否则不同层位移物理含义不一致）
        sx = W1 / float(W0)
        sy = H1 / float(H0)
        offset_t[:, 0::2, :, :] = offset_t[:, 0::2, :, :] * sx  # dx
        offset_t[:, 1::2, :, :] = offset_t[:, 1::2, :, :] * sy  # dy
        return offset_t, mask_t

    def apply(self, feat_to_warp, offset, mask):
  
        # ### 修改行：若尺寸不同，内部自适配 resize + scale
        if offset.shape[-2:] != feat_to_warp.shape[-2:]:
            offset, mask = self._resize_offset_mask(offset, mask, feat_to_warp.shape[-2:])

        # ### 修改行：执行 DCN 变形
        y = self.deform_conv(feat_to_warp, offset, mask)
        y = self.act(self.norm(y))
        return y
"""
"""
class LocalOffsetPredictor(nn.Module):
    def __init__(self, in_ch, kernel_size=3):
        super().__init__()

        self.kernel_size = kernel_size
        k2 = kernel_size * kernel_size

        # ============================================================
        # ###【修改 1】offset + mask 预测头（DCNv2 规范）
        # 输出通道数 = 2*k*k (offset) + k*k (mask)
        # ============================================================
        self.offset_mask_conv = nn.Conv2d(
            in_ch * 2,
            3 * k2,
            kernel_size=3,
            padding=1
        )

        # ============================================================
        # ###【修改 2】可变形卷积，用于真正执行局部几何变形
        # ============================================================
        self.deform_conv = DeformConv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )

        self.norm = nn.BatchNorm2d(in_ch)
        self.act = nn.ReLU(inplace=True)

        self._init_offset()

    def _init_offset(self):

        nn.init.constant_(self.offset_mask_conv.weight, 0.)
        nn.init.constant_(self.offset_mask_conv.bias, 0.)

    def forward(self, x, feat_to_warp):

        # ============================================================
        # ###【修改 4】预测 offset 与 mask
        # ============================================================
        # print("888888888888")
        out = self.offset_mask_conv(x)
        k2 = self.kernel_size * self.kernel_size

        offset = out[:, :2 * k2, :, :]          # (B, 2k², H, W)
        mask   = out[:, 2 * k2:, :, :]           # (B, k², H, W)
        mask   = torch.sigmoid(mask)

        # ============================================================
        # ###【修改 5】offset 数值约束（稳定训练）
        # 相对位移，[-0.01, 0.01]，与你原设定一致
        # ============================================================
        offset = torch.tanh(offset) * 0.05
        # offset = torch.tanh(offset)

        # ============================================================
        # ###【修改 6】执行可变形卷积（核心）
        # ============================================================
        # print("99999999999999")
        # print("feat_to_warp.shape：", torch.max(feat_to_warp), torch.min(feat_to_warp))
        # print("offset:",torch.max(offset), torch.min(offset))
        # print("mask:", torch.max(mask), torch.min(mask))
        warped_feat = self.deform_conv(
            feat_to_warp,
            offset,
            mask
        )

        # print("aaaaaaaaaaaaaaaaa")
        warped_feat = self.norm(warped_feat)
        warped_feat = self.act(warped_feat)

        return warped_feat
"""

##局部非线性像素级微调
class LocalOffsetPredictor(nn.Module):
    def __init__(self, in_ch):
        super(LocalOffsetPredictor, self).__init__()
        self.conv_offset = nn.Conv2d(in_ch, 2, kernel_size=3, padding=1)  # 输出两个通道，分别对应δx和δy
        
    def forward(self, x):
        offset = self.conv_offset(x)  # (N, 2, H, W)
        offset = torch.tanh(offset) * 0.05  # 将偏移量限制在 [-0.01, 0.01]
        return offset


##rgb图像矫正器
class DepthwiseConv(nn.Module):
    """深度可分离卷积（轻量级核心）"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return x

import numpy as np 
class RGBAdjuster(nn.Module):
    def __init__(self, channel=496, kernel_size=3, padding=1,
                 use_Affine=False, use_centerPoint=False, use_exist=False, use_weight_map=False, use_localOffset=False, num_features=5):
        super().__init__()
        
        self.use_Affine = use_Affine
        self.use_centerPoint = use_centerPoint
        self.use_exist = use_exist
        self.use_weight_map = use_weight_map
        self.use_localOffset = use_localOffset

        if num_features==5:
            channel_list=[16,64,128,128,256]  #592
        else:
            channel_list=[128,128,256] #后三层,512

        if self.use_Affine:
            self.affine_heads = AffinePredictor(channel//2, kernel_size=kernel_size, padding=padding)

        if use_centerPoint:
            self.centerPoint_heads = CenterPredictor(channel//2)
        if use_exist:
            self.uav_exist_heads = ExistPredictor(channel//2)
        if use_weight_map:
            self.weight_heads = nn.ModuleList()
            for i in range(num_features):
                self.weight_heads.append(nn.Sequential(
                    DepthwiseConv(channel_list[i], 1),
                    nn.Sigmoid()
                ))

        if use_localOffset:
            self.offset_heads = nn.ModuleList()
            for i in range(num_features):
                self.offset_heads.append(LocalOffsetPredictor(channel_list[i]*2))  # ### 修改行：使用支持 predict/apply 的 DCN 版本

        # ### 修改行：5层×2模态 => 10C，按 rgb5,ir5,rgb4,ir4,... 交错拼接后降到 2C（尺寸保持不变）
        self.cross_level_reduce = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=1, padding=0),  # 10C -> 2C
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),  # 保持 H,W 不变
            nn.ReLU(inplace=True),
        )
        self.align_level_idx = 2

        self.rgb_level_reduce = nn.Sequential(
            nn.Conv2d(channel, channel//2, kernel_size=1, padding=0),  # 10C -> 2C
            nn.ReLU(inplace=True),
        )

        self.ir_level_reduce = nn.Sequential(
            nn.Conv2d(channel, channel//2, kernel_size=1, padding=0),  # 10C -> 2C
            nn.ReLU(inplace=True),
        )


    def forward(self, x):
        """
        x[0]=rgb_feats: list/tuple(5) of [B,C,Hi,Wi]
        x[1]=ir_feats : list/tuple(5) of [B,C,Hi,Wi]

        return:
        warped_rgb_feats: list(5) of [B,C,Hi,Wi]
        center_points   : optional
        uav_exist       : optional
        txtys           : [B,2]
        """
        # print("x[0].shape:", x[0].shape)  #([1, 16, 128, 128])
        # print("x[1].shape:", x[1].shape)  #([1, 64, 64, 64])
        # print("x[2].shape:", x[2].shape)  #([1, 128, 32, 32])
        # print("x[3].shape:", x[3].shape)  #([1, 128, 16, 16])
        # print("x[4].shape:", x[4].shape)  #[1, 256, 8, 8])
        num_features = len(x)//2
        rgb_feats, ir_feats = x[0:num_features], x[num_features:num_features*2]

        # ============================================================
        # 0) 以第3层作为对齐目标尺度（H3,W3）
        # ============================================================
        l3 = self.align_level_idx  # 通常=2
        rgb_l3 = rgb_feats[l3]
        B, C, Ht, Wt = rgb_l3.shape

        # ============================================================
        # 1) 5层 RGB/IR 统一对齐到第3层尺寸（pool2d / upsample）
        # ============================================================
        rgb_aligned, ir_aligned = [], []
        for i in range(num_features):
            rgb_aligned.append(self._align_to_hw(rgb_feats[i], (Ht, Wt)))  # ### 修改：全都对齐到 H3,W3
            ir_aligned.append(self._align_to_hw(ir_feats[i],  (Ht, Wt)))   # ### 修改：全都对齐到 H3,W3

        rgb_stack = torch.cat([rgb_aligned_i for rgb_aligned_i in rgb_aligned], dim=1)  # ### 修改：只拼 RGB
        rgb_fused = self.rgb_level_reduce(rgb_stack) 

        center_points, uav_exist = None, None

        if self.use_centerPoint:
            center_points = self.centerPoint_heads(rgb_fused)  # ### 修改：输入 rgb_fused

        if self.use_exist:
            # 约定：ExistPredictor 返回 (weighted_feat, exist_score)
            uav_exist = self.uav_exist_heads(rgb_fused)  # ### 修改：输入 rgb_fused
            # uav_exist_t = uav_exist.view(uav_exist.size(0), 1, 1, 1)
            # rgb_fused = rgb_fused * uav_exist_t

        if self.use_Affine:
            ir_stack = torch.cat([ir_aligned_i for ir_aligned_i in ir_aligned], dim=1)   # ### 修改：只拼 IR
            ir_fused = self.ir_level_reduce(ir_stack)  # [B,C,Ht,Wt]  # ### 修改：IR 融合降维
            fused_for_affine = torch.cat([ir_fused, rgb_fused], dim=1)  # [B,2C,Ht,Wt]  # ### 修改：符合你的要求
            # fused_for_affine, _ = self.interleave_by_level(rgb_aligned, ir_aligned, order=(4,3,2,1,0))
            affine_params = self.affine_heads(fused_for_affine)            # [B,6]
            txtys = affine_params[:, 4:]                                   # [B,2]
            theta = affine_params.view(-1, 2, 3)  # [B,2,3]

            # B,C,H,W = weighted_rgb.shape
            # grid = F.affine_grid(theta, size=[B, C, H, W], align_corners=True)
            # warped_rgb = F.grid_sample(weighted_rgb, grid, align_corners=True, mode='bilinear')

            # if self.use_localOffset:
            #     # refine_fused = torch.cat([ir_fused, warped_rgb], dim=1)  
            #     refine_fused, _ = self.interleave_by_level(rgb_aligned, ir_aligned, order=(4,3,2,1,0))
            #     warped_rgb = self.offset_heads(refine_fused, warped_rgb)

            # ch_list = [256,128,64,32,16]
            # warped_rgb_feats = self.split_by_channels(warped_rgb, ch_list, order=(4,3,2,1,0))
            warped_rgb_feats = []
            for i in range(num_features):
                feat_i = rgb_feats[i]  # 保持原尺度,没经过weight_map
                Bi, Ci, Hi, Wi = feat_i.shape

                # ### 关键：θ 相同，但每层分辨率不同，必须按各自 size 生成 grid
                grid_i = F.affine_grid(theta, size=[Bi, Ci, Hi, Wi], align_corners=True)
                warped_i = F.grid_sample(feat_i, grid_i, align_corners=True, mode='bilinear')

                if self.use_localOffset: #整图漂移
                    refine_fused = torch.cat([ir_feats[i], warped_i], dim=1)  
                    # warped_i = self.offset_heads[i](refine_fused, warped_i)
                    offsets = self.offset_heads[i](refine_fused)
                    offset_norm = offsets.permute(0, 2, 3, 1)* 2.0   # 归一化到[-1,1]  # [B, H, W, 2]
                    final_grid = grid_i + offset_norm
                    warped_i = F.grid_sample(feat_i, final_grid, align_corners=True, mode='bilinear')
                if self.use_exist:
                    uav_exist_i = uav_exist.view(uav_exist.size(0), 1, 1, 1)
                    warped_i = warped_i * uav_exist_i
                if self.use_weight_map:
                    weight_map = self.weight_heads[i](warped_i)          #预测权重图，是否存在无人机
                    warped_i = warped_i * weight_map           # ### 修改：对融合RGB进行像素抑制  
                warped_rgb_feats.append(warped_i)
        else:
            warped_rgb_feats = []
            affine_params = None
            for i in range(num_features):
                feat_i = rgb_feats[i]  # 保持原尺度,没经过weight_map
                Bi, Ci, Hi, Wi = feat_i.shape

                if self.use_localOffset: #整图漂移
                    refine_fused = torch.cat([ir_feats[i], feat_i], dim=1)  
                    warped_i = self.offset_heads[i](refine_fused, feat_i)
                if self.use_exist:
                    uav_exist_i = uav_exist.view(uav_exist.size(0), 1, 1, 1)
                    warped_i = feat_i * uav_exist_i
                if self.use_weight_map:
                    weight_map = self.weight_heads[i](feat_i)          #预测权重图，是否存在无人机
                    warped_i = feat_i * weight_map           # ### 修改：对融合RGB进行像素抑制  
                warped_rgb_feats.append(feat_i)
        # # ============================================================
        # # 6) Stage B：局部偏移（仅预测一次 offset/mask），然后对 5 层 RGB 复用
        # #    用 (ir_fused + 第3层仿射后RGB) 进行 refine（同样满足“与5层IR融合cat后续操作”的精神）
        # # ============================================================
        # if self.use_localOffset:
        #     warped_rgb_l3 = warped_rgb_feats[l3]  # 第3层尺度
        #     # ### 修改：refine 输入使用 5层IR融合特征 ir_fused（在第3层尺度）+ 第3层 warped_rgb
        #     refine_fused = torch.cat([ir_fused, self._align_to_hw(warped_rgb_l3, (Ht, Wt))], dim=1)  # [B,2C,Ht,Wt]

        #     # ### 修改：仅预测一次 offset/mask（base=第3层尺度）
        #     offset_base, mask_base = self.offset_heads.predict(refine_fused)

        #     # ### 修改：对 5 层统一 apply（内部自动 resize+scale dx/dy）
        #     for i in range(5):
        #         warped_rgb_feats[i] = self.offset_heads.apply(warped_rgb_feats[i], offset_base, mask_base)

        # return warped_rgb_feats, center_points, uav_exist, txtys
        return warped_rgb_feats, center_points, uav_exist, affine_params

    
    def _align_to_hw(self, feat, target_hw):
        """
        feat: [B,C,H,W]
        target_hw: (Ht, Wt)  # 第3层空间尺寸
        策略：
          - 若 feat 更大：用 adaptive_avg_pool2d 下采样（等价“pool2d”语义，且不依赖倍率整除）
          - 若 feat 更小：用 bilinear 上采样
          - 相等：直接返回
        """
        Ht, Wt = target_hw
        H, W = feat.shape[-2:]

        # ### 修改行：下采样到 target_hw（pool2d 语义）
        if (H > Ht) or (W > Wt):
            return F.avg_pool2d(feat, kernel_size=H//Ht, stride=H//Ht)

        # ### 修改行：上采样到 target_hw
        if (H < Ht) or (W < Wt):
            return F.interpolate(feat, size=(Ht, Wt), mode="bilinear", align_corners=False)

        return feat

    def interleave_by_level(self, rgb_aligned, ir_aligned, order=(4, 3, 2, 1, 0)):
        """
        rgb_aligned/ir_aligned: list(5)，每个元素 [B, Ci, H, W]，要求同一层 RGB/IR 的 H,W 一致
        order: 层的拼接顺序，默认从高层到低层（5->1）
        return:
        fused: [B, sum_i(2*Ci), H, W]，通道顺序为 rgb_i, ir_i, rgb_j, ir_j, ...
        rgb_splits: list(5)  # 用于后续把 fused 中的 RGB 部分拆出来（按拼接顺序记录每段通道数）
        """
        feats = []
        rgb_splits = []
        for i in order:
            # ### 修改行：按层整块交错，而非逐通道交错
            feats.append(rgb_aligned[i])                 # [B, Ci, H, W]
            feats.append(ir_aligned[i])                  # [B, Ci, H, W]
            rgb_splits.append(rgb_aligned[i].shape[1])   # 记录该层 RGB 的通道数 Ci
        fused = torch.cat(feats, dim=1)
        return fused, rgb_splits

    def split_by_channels(self, packed: torch.Tensor, ch_list, order=(4, 3, 2, 1, 0)):
        """
        packed: [B, sum(ch_list), H, W]
        ch_list: 长度=5，例如 [16,32,64,128,256]，表示每层 RGB 原始通道数
        order: packed 的拼接层顺序（必须与当时拼接一致）
        return: list(5)，按原索引 0..4 放回
        """
        out = [None] * 5
        idx = 0
        for i in order:
            ci = ch_list[i]
            out[i] = packed[:, idx: idx + ci, :, :]
            idx += ci
        return out


if __name__ == '__main__':
    rgb = torch.rand(1, 3, 480, 640).to('cuda:0')
    t = torch.rand(1, 3, 480, 640).to('cuda:0')
    model = Model(mode='b1', inputs='rgbt', fusion_mode='CM-SSM', n_class=12,).eval().to('cuda:0')
    out = model(rgb, t)
    print(out.shape)

    # from ptflops import get_model_complexity_info

    # flops, params = get_model_complexity_info(model, (3, 480, 640), as_strings=True, print_per_layer_stat=False)
    # print('Flops ' + flops)
    # print('Params ' + params)

    # from fvcore.nn import flop_count_table, FlopCountAnalysis
    #
    # print(flop_count_table(FlopCountAnalysis(model, rgb)))
    # from thop import profile
    # flops, params = profile(model, inputs=(rgb, t))
    # print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")  # 转换为 GFLOPs
    # print(f"Parameters: {params / 1e6:.2f} M")
