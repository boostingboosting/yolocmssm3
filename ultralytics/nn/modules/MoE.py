import torch
import torch.nn as nn
import torch.nn.functional as F

class CM_SSM(nn.Module):
    def __init__(self, in_c):
        super(CM_SSM, self).__init__()
        self.SS2D = SS2D_rgbt(in_c)  # 注意：SS2D_rgbt 需要定义
        self.conv1 = nn.Sequential(nn.Conv2d(2*in_c, in_c, 3, 1, 1),
                                   nn.BatchNorm2d(in_c),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(3*in_c, in_c, 1, 1, 0),
                                   nn.BatchNorm2d(in_c),
                                   nn.ReLU())
    def forward(self, rgb, t):
        B, C, H, W = rgb.shape
        left = self.conv1(torch.cat((rgb, t), dim=1))

        rgb_, t_ = self.SS2D(rgb.permute(0, 2, 3, 1), t.permute(0, 2, 3, 1))
        rgb_ = rgb_ + rgb
        t_ = t_ + t

        out = self.conv2(torch.cat((left, rgb_, t_), dim=1))
        return out

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(2*dim, dim, 3, 1, 1),
                                   nn.BatchNorm2d(dim),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(3*dim, dim, 1, 1, 0),
                                   nn.BatchNorm2d(dim),
                                   nn.ReLU())
        self.attn_rgb = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.attn_t = nn.MultiheadAttention(dim, heads, batch_first=True)
        
    def forward(self, rgb, t):
        B, C, H, W = rgb.shape
        left = self.conv1(torch.cat((rgb, t), dim=1))

        # 修正：使用 view 而不是 view_as
        rgb_flat = rgb.view(B, C, H*W).permute(0, 2, 1).contiguous()
        t_flat = t.view(B, C, H*W).permute(0, 2, 1).contiguous()
        
        # 修正：使用正确的注意力模块名称
        rgb_sup, _ = self.attn_rgb(rgb_flat, t_flat, t_flat)
        t_sup, _ = self.attn_t(t_flat, rgb_flat, rgb_flat)
        
        # 修正：使用 view 而不是 view_as
        rgb_ = (rgb_flat + rgb_sup).permute(0, 2, 1).view(B, C, H, W)
        t_ = (t_flat + t_sup).permute(0, 2, 1).view(B, C, H, W)

        out = self.conv2(torch.cat((left, rgb_, t_), dim=1))
        return out

class FusionMoE(nn.Module):
    def __init__(self, inc, num_cross=2, num_cm=2, heads=8, 
                 tau=1.0, group_balance=0.01, expert_balance=0.01):
        super().__init__()
        self.dim = inc
        self.tau = tau
        self.group_balance = group_balance
        self.expert_balance = expert_balance

        # 专家池
        self.cross_experts = nn.ModuleList([
            CrossAttention(self.dim, heads) for _ in range(num_cross)
        ])
        self.cm_experts = nn.ModuleList([
            CM_SSM(self.dim) for _ in range(num_cm)
        ])

        # 路由器网络 - 修正：只定义实际存在的路由器
        self.router_reduce = nn.Linear(self.dim * 2, self.dim)
        self.group_router = nn.Linear(self.dim, 2)  # 修正：只有2个组（cross和cm）
        self.cross_router = nn.Linear(self.dim, num_cross)
        self.cm_router = nn.Linear(self.dim, num_cm)  # 修正：使用cm_router

    def forward(self, rgb, t):
        B, C, H, W = rgb.shape
        
        # 1. 序列级路由输入
        # 修正：使用view而不是view_as
        rgb_flat = rgb.view(B, C, H*W).transpose(1, 2)
        t_flat = t.view(B, C, H*W).transpose(1, 2)
        
        v_mean = rgb_flat.mean(dim=1)  # [B, D]
        a_mean = t_flat.mean(dim=1)   # [B, D]
        
        routing_input = torch.cat([v_mean, a_mean], dim=-1)  # [B, 2*D]
        reduced_routing = self.router_reduce(routing_input)  # [B, D]

        # 2. 组路由决策
        group_logits = self.group_router(reduced_routing)  # [B, 2]
        
        # 3. 专家路由决策
        cross_logits = self.cross_router(reduced_routing)  # [B, num_cross]
        cm_logits = self.cm_router(reduced_routing)       # [B, num_cm]
        
        # 初始化输出
        out = torch.zeros_like(rgb)
        reg_loss = 0.0
        
        if self.training:
            # 训练模式
            group_gumbel = -torch.log(-torch.log(torch.rand_like(group_logits)) + 1e-8)
            group_gates = torch.sigmoid((group_logits + group_gumbel) / self.tau)  # [B, 2]
            
            cross_gumbel = -torch.log(-torch.log(torch.rand_like(cross_logits)) + 1e-8)
            cross_gates = torch.sigmoid((cross_logits + cross_gumbel) / self.tau)  # [B, num_cross]
            
            cm_gumbel = -torch.log(-torch.log(torch.rand_like(cm_logits)) + 1e-8)
            cm_gates = torch.sigmoid((cm_logits + cm_gumbel) / self.tau)  # [B, num_cm]
            
            # 计算所有专家输出
            cross_outputs = [expert(rgb, t) for expert in self.cross_experts]  # 修正：参数数量
            cm_outputs = [expert(rgb, t) for expert in self.cm_experts]        # 修正：参数数量
            
            # 加权求和
            for i in range(len(self.cross_experts)):
                w = group_gates[:, 0].unsqueeze(1) * cross_gates[:, i].unsqueeze(1)
                out += w.unsqueeze(1).unsqueeze(1) * cross_outputs[i]  # 修正：增加维度匹配
            
            for i in range(len(self.cm_experts)):
                w = group_gates[:, 1].unsqueeze(1) * cm_gates[:, i].unsqueeze(1)
                out += w.unsqueeze(1).unsqueeze(1) * cm_outputs[i]  # 修正：增加维度匹配
            
            # 稀疏性正则化
            reg_loss += self.group_balance * torch.mean(torch.sum(group_gates, dim=-1))
            reg_loss += self.expert_balance * (
                torch.mean(torch.sum(cross_gates, dim=-1)) +
                torch.mean(torch.sum(cm_gates, dim=-1))
            )

        else:
            # 推理模式
            group_mask = (group_logits > 0).float()  # [B, 2]
            cross_mask = (cross_logits > 0).float()  # [B, num_cross]
            cm_mask = (cm_logits > 0).float()  # [B, num_cm]
            
            for i, expert in enumerate(self.cross_experts):
                active_batches = (group_mask[:, 0] * cross_mask[:, i]).sum() > 0
                if active_batches:
                    expert_out = expert(rgb, t)  # 修正：参数数量
                    w = (group_mask[:, 0] * cross_mask[:, i]).unsqueeze(1)
                    out += w.unsqueeze(1).unsqueeze(1) * expert_out
            
            for i, expert in enumerate(self.cm_experts):
                active_batches = (group_mask[:, 1] * cm_mask[:, i]).sum() > 0
                if active_batches:
                    expert_out = expert(rgb, t)  # 修正：参数数量
                    w = (group_mask[:, 1] * cm_mask[:, i]).unsqueeze(1)
                    out += w.unsqueeze(1).unsqueeze(1) * expert_out
        
        # 训练时返回正则化损失
        if self.training:
            return out, reg_loss
        return out, None