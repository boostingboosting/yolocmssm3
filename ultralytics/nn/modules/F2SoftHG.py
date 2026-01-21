import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .conv import Conv
from .block import C3k, Bottleneck  

class SoftHyperedgeGeneration(nn.Module):
    def __init__(self, node_dim, num_hyperedges, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.num_hyperedges = num_hyperedges
        self.head_dim = node_dim // num_heads

        self.prototype_base = nn.Parameter(torch.Tensor(num_hyperedges, node_dim))
        nn.init.xavier_uniform_(self.prototype_base)
        
        self.context_net = nn.Linear(2 * node_dim, num_hyperedges * node_dim)
        self.pre_head_proj = nn.Linear(node_dim, node_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scaling = math.sqrt(self.head_dim)

    def forward(self, X):
        B, N, D = X.shape
        avg_context = X.mean(dim=1)         
        max_context, _ = X.max(dim=1)        
        context_cat = torch.cat([avg_context, max_context], dim=-1) 
        
        prototype_offsets = self.context_net(context_cat).view(B, self.num_hyperedges, D) 
        prototypes = self.prototype_base.unsqueeze(0) + prototype_offsets               
        
        X_proj = self.pre_head_proj(X)  
        X_heads = X_proj.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        proto_heads = prototypes.view(B, self.num_hyperedges, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        X_heads_flat = X_heads.reshape(B * self.num_heads, N, self.head_dim)
        proto_heads_flat = proto_heads.reshape(B * self.num_heads, self.num_hyperedges, self.head_dim).transpose(1, 2)
        
        logits = torch.bmm(X_heads_flat, proto_heads_flat) / self.scaling  
        logits = logits.view(B, self.num_heads, N, self.num_hyperedges).mean(dim=1) 
        logits = self.dropout(logits)   
        
        return F.softmax(logits, dim=1)

class SoftHGNN(nn.Module):
    def __init__(self, embed_dim, num_hyperedges_intra=8, num_hyperedges_inter=4, num_heads=4, dropout=0.1):
        super().__init__()
        self.edge_generator_rgb = SoftHyperedgeGeneration(embed_dim, num_hyperedges_intra, num_heads, dropout)
        self.edge_proj_rgb = nn.Sequential(
            nn.Linear(embed_dim, embed_dim ),
            nn.GELU()
        )
        self.node_proj_rgb = nn.Sequential(
            nn.Linear(embed_dim, embed_dim ),
            nn.GELU()
        )


        self.edge_generator_ir = SoftHyperedgeGeneration(embed_dim, num_hyperedges_intra, num_heads, dropout)
        self.edge_proj_ir = nn.Sequential(
            nn.Linear(embed_dim, embed_dim ),
            nn.GELU()
        )
        self.node_proj_ir = nn.Sequential(
            nn.Linear(embed_dim, embed_dim ),
            nn.GELU()
        )

        self.edge_generator_crossmodal = SoftHyperedgeGeneration(embed_dim, num_hyperedges_inter, num_heads, dropout)
        self.edge_proj_crossmodal = nn.Sequential(
            nn.Linear(embed_dim, embed_dim ),
            nn.GELU()
        )
        self.node_proj_crossmodal = nn.Sequential(
            nn.Linear(embed_dim, embed_dim ),
            nn.GELU()
        )

        
    def forward(self, x_rgb, x_ir):
        # 获取单模态超边
        A_rgb = self.edge_generator_rgb(x_rgb) #每个节点对超边的参与度
        He_rgb = torch.bmm(A_rgb.transpose(1, 2), x_rgb) 
        He_rgb = self.edge_proj_rgb(He_rgb)

        A_ir = self.edge_generator_ir(x_ir)
        He_ir = torch.bmm(A_ir.transpose(1, 2), x_ir)
        He_ir = self.edge_proj_ir(He_ir)

        # 超图的超图，更新单模态超边
        He_fuse = torch.cat([He_rgb,He_ir], dim=1) #B,16,C。问题：没有显示建模一个模态与另一个模态的关系，很可能只学单模态内部的关系。
        A_crossmodal = self.edge_generator_crossmodal(He_fuse)
        He_crossmodal = torch.bmm(A_crossmodal.transpose(1, 2), He_fuse)#跨模态超边
        He_crossmodal = self.edge_proj_crossmodal(He_crossmodal)

        He_fuse_new = torch.bmm(A_crossmodal, He_crossmodal) 
        He_fuse_new = self.node_proj_crossmodal(He_fuse_new)
        He_rgb_new, He_ir_new = torch.chunk(He_fuse_new, chunks=2, dim=1) #更新后的rgb、ir超边

        # 更新节点
        He_rgb = He_rgb+He_rgb_new
        He_ir = He_ir+He_ir_new
        x_rgb_new = torch.bmm(A_rgb, He_rgb) 
        x_rgb_new = self.node_proj_rgb(x_rgb_new)

        He_ir = He_ir+He_ir_new
        He_ir = He_ir+He_ir_new
        x_ir_new = torch.bmm(A_ir, He_ir) 
        x_ir_new = self.node_proj_ir(x_ir_new)

        x_rgb_new = x_rgb_new + x_rgb
        x_ir_new = x_ir_new + x_ir
        X_new = torch.cat([x_rgb_new,x_ir_new], dim=-1)
        return X_new

class SoftHGBlock(nn.Module):
    def __init__(self, embed_dim, num_hyperedges_intra=8, num_hyperedges_inter=4, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.cv = Conv(2*embed_dim, embed_dim)
        self.softhgnn = SoftHGNN(
            embed_dim=embed_dim,
            num_hyperedges_intra=num_hyperedges_intra,
            num_hyperedges_inter=num_hyperedges_inter,
            num_heads=num_heads,
            dropout=dropout
        )
        
    def forward(self, x):
        x_rgb, x_ir = x[0],x[1]
        B, C, H, W = x_rgb.shape
        tokens_rgb = x_rgb.flatten(2).transpose(1, 2) #B,HW,C
        tokens_ir = x_ir.flatten(2).transpose(1, 2)
        tokens = self.softhgnn(tokens_rgb, tokens_ir)  
        x_out = tokens.transpose(1, 2).view(B, 2*C, H, W)
        x_out = self.cv(x_out)
        return x_out 
       
class FusionModule(nn.Module):
    def __init__(self, C, adjust_channels):
        super().__init__()
        
        self.downsample = nn.AvgPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        if adjust_channels:
            self.conv_out = Conv(4 * C, C, 1)
        else:
            self.conv_out = Conv(3 * C, C, 1)
        
    def forward(self, x):
        x0_ds = self.downsample(x[0])
        x2_up = self.upsample(x[2])
        x_cat = torch.cat([x0_ds, x[1], x2_up], dim=1)
        out = self.conv_out(x_cat)
        return out

class F2SoftHG(nn.Module):
    def __init__(self, c1, c2, n=1, c3k=False, shortcut=False, g=1, e=0.5, adjust_channels=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 3 * self.c, 1, 1)
        self.cv2 = Conv((4 + n) * self.c, c2, 1) 
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )
        self.fuse = FusionModule(c1, adjust_channels)
        self.softhgbranch1 = SoftHGBlock(embed_dim=self.c, 
                                   num_hyperedges_intra=8, 
                                   num_heads=8,
                                   dropout=0.1)
        self.softhgbranch2 = SoftHGBlock(embed_dim=self.c, 
                                   num_hyperedges_intra=8, 
                                   num_heads=8,
                                   dropout=0.1)
                    
    def forward(self, X):
        x = self.fuse(X)
        y = list(self.cv1(x).chunk(3, 1))
        softhg_out1 = self.softhgbranch1(y[1])
        softhg_out2 = self.softhgbranch2(y[1])
        y.extend(m(y[-1]) for m in self.m)
        y[1] = softhg_out1
        y.append(softhg_out2)
        return self.cv2(torch.cat(y, 1))

class ShapeAlignConv(nn.Module):
    def __init__(self, in_channels, adjust_channels=True):
        super().__init__()
        self.adjust_channels = adjust_channels
        self.downsample = nn.AvgPool2d(kernel_size=2)
        if adjust_channels:
            self.conv = Conv(in_channels, in_channels * 2, 1)
    
    def forward(self, x):
        x = self.downsample(x)
        if self.adjust_channels:
            x = self.conv(x)
        return x

class MergeConv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = Conv(in_channels * 2, in_channels, 1)
    def forward(self, x):
        x_cat = torch.cat(x, dim=1)
        return self.conv(x_cat)