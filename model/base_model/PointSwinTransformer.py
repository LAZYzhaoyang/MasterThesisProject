from matplotlib.cbook import flatten
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from torch import einsum

import numpy as np
import os

#===================================Swin Transformer Block===================================#

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        # x: [b, n, dim]
        return torch.roll(x, shifts=self.displacement, dims=1)

class WindowAttention1D(nn.Module):
    def __init__(self,
                 dim:int,
                 heads:int, 
                 head_dim:int, 
                 shifted:bool, 
                 window_size:int):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.shifted = shifted
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        # change the window local by shifted the image
        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
        
        self.pos_embedding = nn.Parameter(torch.randn(window_size, window_size))   

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)
        # x : [b, n, dim]
        b, n, _, h = *x.shape, self.heads 
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # qkv : [b, n, inner_dim, 3]
        n_w = n // self.window_size
        q, k, v = map(
            lambda t: rearrange(t, 'b (n_w ws) (h d) -> b h n_w ws d',
                                h=h, ws=self.window_size, n_w=n_w), qkv)
        # q, k, v : [b, head, windows_num(n_w), windows_size, head_dim]
        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale
        # dots : [b, head, windows_num, windows_size, windows_size]
        dots += self.pos_embedding
        attn = dots.softmax(dim=-1)
        # attn.shape = dots.shape
        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        # out: [b, head, windows_num, windows_size, head_dim]
        # out = rearrange(out, 'b h n_w ws d -> b (n_w ws) (h d)',
        #                 h=h, ws=self.window_size, n_w=n_w)
        out = rearrange(out, 'b h n_w ws d -> b (n_w ws) (h d)')
        # out: [b, n, inner_dim]
        out = self.to_out(out)
        # out: [b, n, out_dim]
        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out

class PointSwinBlock(nn.Module):
    def __init__(self, dim:int,heads:int,head_dim:int,shifted:bool,window_size:int, mlp_dim:int):
        super().__init__()
        self.AttentionBlock = Residual(PreNorm(dim, WindowAttention1D(dim=dim,
                                                                      heads=heads,
                                                                      head_dim=head_dim,
                                                                      shifted=shifted,
                                                                      window_size=window_size)))
        self.MlpBlock = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))
        
    def forward(self, x):
        # x: [b, n, dim]
        x = self.AttentionBlock(x)
        # x: [b, n, dim]
        x = self.MlpBlock(x)
        # x: [b, n, dim]
        return x
#===================================Scale Block===================================#

class PointPatchMerging(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 downscaling_factor:int):
        super().__init__()
        # down sample the point cloud
        self.downscaling_factor = downscaling_factor
        self.linear = nn.Linear(in_channels * downscaling_factor, out_channels)
        #print(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        #print(x.shape)
        b, dim, n = x.shape
        #x:[b, dim, n]
        new_n = n//self.downscaling_factor
        x = rearrange(x, 'b d (wn w) -> b wn (w d)', wn=new_n, w=self.downscaling_factor, d= dim)
        # x [N, new_n, down scaling factor*dim]
        # looks like channel attention 
        x = self.linear(x)
        # x : [N, new_n, out_channels]
        x = x.permute(0, 2, 1)
        # output x : [N, out_channels, new_n]
        return x

class PointPixelShuffle(nn.Module):
    def __init__(self, 
                 in_channel:int,
                 out_channel:int,
                 up_scale_factor:int):
        super().__init__()
        # up sample the point cloud
        self.inC = in_channel
        self.outC = out_channel
        self.upscale = up_scale_factor
        self.add_channel = nn.Linear(in_channel, out_channel*up_scale_factor)
        
    def forward(self, x):
         # x: [b, in_C, n]
        x = x.permute(0, 2, 1)
        # x: [b, n, in_C]
        x = self.add_channel(x)
        # x: [N, n, out_C*upscale]
        x = x.permute(0, 2, 1)
        # x: [N, out_C*upscale, n]
        x = rearrange(x, 'b (outc upc) n -> b outc (n upc)', outc=self.outC, upc=self.upscale)
        # x: [N, out_C, n*upscale]
        return x

#===================================Stage Block===================================#

class PointSwinAttentionBlock(nn.Module):
    def __init__(self, dim:int, heads:int, head_dim:int, window_size:int, mlp_dim=None):
        super().__init__()
        if mlp_dim is None:
            mlp_dim = dim*4
        self.wmha = PointSwinBlock(dim=dim, heads=heads, head_dim=head_dim, window_size=window_size, mlp_dim=mlp_dim, shifted=False)
        
        self.swmha = PointSwinBlock(dim=dim, heads=heads, head_dim=head_dim, window_size=window_size, mlp_dim=mlp_dim, shifted=True)
        
    def forward(self, x):
        # x: [b, n, dim]
        x = self.wmha(x)
        # x: [b, n, dim]
        x = self.swmha(x)
        # x: [b, n, dim]
        return x


class PointSwinEncoderLayer(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, downscaling_factor:int, layers:int, 
                 heads:int, head_dim:int, window_size:int, mlp_dim=None):
        super().__init__()
        self.attentions = nn.ModuleList([])
        for i in range(layers):
            self.attentions.append(PointSwinAttentionBlock(dim=in_channels, heads=heads, head_dim=head_dim,
                                                           window_size=window_size, mlp_dim=mlp_dim))
        self.downsample = PointPatchMerging(in_channels=in_channels, out_channels=out_channels, downscaling_factor=downscaling_factor)
        
    def forward(self,x):
        #x:[b, inC, n]
        x = x.permute(0,2,1)
        #x:[b, n, inC]
        for attn in self.attentions:
            x = attn(x)
        #x:[b, n, inC]
        x = x.permute(0,2,1)
        #x:[b, inC, n]
        x = self.downsample(x)
        #x:[b, outC, n]
        return x
    

class PointSwinDecoderLayer(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, up_scale_factor:int, layers:int, 
                 heads:int, head_dim:int, window_size:int, mlp_dim=None):
        super().__init__()
        self.upsample = PointPixelShuffle(in_channel=in_channels, out_channel=out_channels, up_scale_factor=up_scale_factor)
        
        self.attentions = nn.ModuleList([])
        for i in range(layers):
            self.attentions.append(PointSwinAttentionBlock(dim=out_channels, heads=heads, head_dim=head_dim,
                                                           window_size=window_size, mlp_dim=mlp_dim))
        
    def forward(self,x):
        #x:[b, inC, n]
        x = self.upsample(x)
        #x:[b, outC, n]
        x = x.permute(0,2,1)
        #x:[b, n, outC]
        for attn in self.attentions:
            x = attn(x)
        #x:[b, n, outC]
        x = x.permute(0,2,1)
        #x:[b, outC, n]
        return x
    
#===================================Point Swin Block===================================#
class PointSwinEncoder(nn.Module):
    def __init__(self, in_channels:int=32, stage_num:int=3, downscaling_factor=4, layernums=1, heads=8, head_dims=32, window_sizes=4, mlp_dims=None, attn_layers=4):
        super().__init__()
        self.inC = getlist(in_channels, stage_num, scale=2)
        self.outC = getlist(in_channels*2, stage_num, scale=2)
        self.downf = getlist(downscaling_factor, stage_num)
        self.layers = getlist(layernums, stage_num)
        self.heads = getlist(heads, stage_num)
        self.head_dims = getlist(head_dims, stage_num)
        self.window_sizes = getlist(window_sizes, stage_num)
        if mlp_dims==None:
            mlp_dims = window_sizes*4
        self.mlp_dims = getlist(mlp_dims, stage_num)
        
        self.encoders = nn.ModuleList([])
        
        for i in range(stage_num):
            self.encoders.append(PointSwinEncoderLayer(in_channels=self.inC[i], out_channels=self.outC[i],
                                                      downscaling_factor=self.downf[i], layers=self.layers[i],
                                                      heads=self.heads[i], head_dim=self.head_dims[i],
                                                      window_size=self.window_sizes[i], mlp_dim=self.mlp_dims[i])) 
        #self.attn = nn.Sequential()
        self.attns = nn.ModuleList([])
        
        for i in range(attn_layers):
            self.attns.append(PointSwinAttentionBlock(dim=self.outC[-1],  heads=self.heads[-1], 
                                                     window_size=self.window_sizes[-1], head_dim=head_dims,
                                                     mlp_dim=self.mlp_dims[-1]))
        
    def forward(self,x):
        features=[]
        for stage in self.encoders:
            x = stage(x)
            features.insert(0, x)
        out = features[0]
        out = out.permute(0,2,1)
        for attn in self.attns:
            out = attn(out)
        out = out.permute(0,2,1)
        features = features[1:]
        return out, features

class PointSwinDecoder(nn.Module):
    def __init__(self, out_channels:int=32, stage_num:int=3, upscaling_factor=4, layernums=1, heads=8, head_dims=32, window_sizes=4, mlp_dims=None):
        super().__init__()
        self.outC = getlist(out_channels, stage_num, scale=2)[::-1]
        self.inC = getlist(out_channels*2, stage_num, scale=2)[::-1]
        self.upf = getlist(upscaling_factor, stage_num)[::-1]
        self.layers = getlist(layernums, stage_num)[::-1]
        self.heads = getlist(heads, stage_num)[::-1]
        self.head_dims = getlist(head_dims, stage_num)[::-1]
        self.window_sizes = getlist(window_sizes, stage_num)[::-1]
        if mlp_dims==None:
            mlp_dims = window_sizes*4
        self.mlp_dims = getlist(mlp_dims, stage_num)[::-1]
        
        self.decoders = nn.ModuleList([])
        for i in range(stage_num):
            self.decoders.append(PointSwinDecoderLayer(in_channels=self.inC[i], out_channels=self.outC[i],
                                                      up_scale_factor=self.upf[i], layers=self.layers[i],
                                                      heads=self.heads[i], head_dim=self.head_dims[i], 
                                                      window_size=self.window_sizes[i],
                                                      mlp_dim=self.mlp_dims[i]))
            
    def forward(self, init_feature, features):
        out = self.decoders[0](init_feature)
        for i in range(len(self.decoders)-1):
            out = out+features[i]
            out = self.decoders[i+1](out)
        return out
    
        
#===================================Response Proxy Model===================================#

class ResponsePointSwinEmbedding(nn.Module):
    def __init__(self, channels:int=3, embedding_dim:int=32, param_dim:int=8, patch_size:int=4):
        super().__init__()
        self.channels = channels
        self.embedding_dim = embedding_dim
        
        self.point_embed = PointPatchMerging(in_channels=channels, out_channels=embedding_dim, downscaling_factor=patch_size)
        self.param_embed = nn.Linear(in_features=param_dim, out_features=embedding_dim)
        
    def forward(self,x, param):
        # x: [b, c, n]
        # param: [b, pd]
        point_embedding = self.point_embed(x)
        param_embedding = self.param_embed(param)
        
        # point_embed: [b, embeddim, n//patch_size]
        # param_embed: [b, embeddim]
        b, h = param_embedding.size()
        param_embedding = param_embedding.view(b,h,1)
        
        feature = point_embedding + param_embedding
        
        return feature
        
class ResponsePointSwinHead(nn.Module):
    def __init__(self, embedding_dim:int=32, pred_dim:int=15, patch_size:int=4, response_dim:int=2, mlp_dim:int=64):
        super().__init__()
        self.pointlayer = PointSwinDecoderLayer(in_channels=embedding_dim, out_channels=pred_dim, up_scale_factor=patch_size,
                                               layers=1, heads=8, head_dim=32, window_size=16, mlp_dim=64)
        
        #self.pooling = nn.Conv1d(in_channels=embedding_dim, out_channels=mlp_dim, kernel_size=)
        self.pooling1 = nn.AdaptiveAvgPool1d(256)
        self.pooling2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=embedding_dim, stride=1, padding=0)
        self.res_head = nn.Sequential(nn.Flatten(),
                                      nn.Linear(in_features=256, out_features=mlp_dim),
                                      nn.GELU(),
                                      nn.Linear(in_features=mlp_dim, out_features=response_dim))
        
    def forward(self, fea, patch_emb):
        x = fea + patch_emb
        
        pred_point = self.pointlayer(x)
        
        res_fea = self.pooling1(x)
        res_fea = res_fea.permute(0,2,1)
        res_fea = self.pooling2(res_fea)
        pred_res = self.res_head(res_fea)
        
        return pred_point, pred_res

class ResponsePointSwinTransformerProxyModel(nn.Module):
    def __init__(self, in_channels:int=3, out_channels:int=15, param_dim:int=8, res_dim:int=2, 
                 embed_dim:int=32, scale_factor:int=4,stage_num:int=3, layers_num:int=1, heads:int=8,
                 head_dims:int=32, window_size:int=4, attn_layers:int=4, mlp_dim=None):
        super().__init__()
        self.embedding = ResponsePointSwinEmbedding(channels=in_channels, embedding_dim=embed_dim, param_dim=param_dim, patch_size=scale_factor)
        self.encoder = PointSwinEncoder(in_channels=embed_dim, stage_num=stage_num, downscaling_factor=scale_factor, layernums=layers_num,
                                       heads=heads, head_dims=head_dims, window_sizes=window_size, mlp_dims=mlp_dim, attn_layers=attn_layers)
        self.decoder = PointSwinDecoder(out_channels=embed_dim, stage_num=stage_num, upscaling_factor=scale_factor, layernums=layers_num,
                                       heads=heads, head_dims=head_dims, window_sizes=window_size, mlp_dims=mlp_dim)
        self.head = ResponsePointSwinHead(embedding_dim=embed_dim, pred_dim=out_channels, patch_size=scale_factor, response_dim=res_dim)
    
    def forward(self, point, param, return_feature=False):
        embed_fea = self.embedding(point, param)
        init_fea, features = self.encoder(embed_fea)
        fea = self.decoder(init_fea, features)
        pred_point, pred_res = self.head(fea, embed_fea)
                        
        all_feas = [embed_fea, features, init_fea, fea]
        if return_feature:
            return pred_point, pred_res, all_feas
        else:
            return pred_point, pred_res

#===================================Point Swin Transformer===================================#

class PointSwinFeatureExtractor(nn.Module):
    def __init__(self, in_channels=15, feature_dim=128, hiddim=64,
                 heads=8, headdim=32, embeddim=4, stage_num=3, 
                 downscale=4, layer_num=2, window_size=4, attnlayer=1):
        super().__init__()
        self.embedding = PointPatchMerging(in_channels=in_channels, out_channels=hiddim, downscaling_factor=embeddim)
        self.backbone = PointSwinEncoder(in_channels=hiddim, 
                                         stage_num=stage_num, 
                                         heads=heads,
                                         head_dims=headdim,
                                         downscaling_factor=downscale, 
                                         layernums=layer_num,
                                         window_sizes=window_size,
                                         attn_layers=attnlayer)
        flatten_dim = self.backbone.outC[-1]
        self.fc = nn.Sequential(
            #nn.AdaptiveAvgPool1d(output_size=1),
            #nn.Flatten(),
            nn.Linear(in_features=flatten_dim, out_features=feature_dim*2),
            #nn.Dropout(p=0.5),
            nn.BatchNorm1d(feature_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=feature_dim*2, out_features=feature_dim),
            nn.BatchNorm1d(feature_dim, affine=False),
            )
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.backbone(x)
        #print(x.size())
        x = self.fc(x.mean(2))
        return x
    
class PointSwin(nn.Module):
    def __init__(self, in_channels=15, num_class=4, hiddim=64,
                 heads=8, headdim=32, embeddim=4, stage_num=3, 
                 downscale=4, layer_num=2, window_size=5, feature_dim=128,
                 attnlayer=1):
        super().__init__()
        self.backbone = PointSwinFeatureExtractor(in_channels=in_channels, feature_dim=feature_dim,
                                                           hiddim=hiddim, heads=heads, headdim=headdim, embeddim=embeddim,
                                                           stage_num=stage_num, downscale=downscale, layer_num=layer_num,
                                                           window_size=window_size, attnlayer=attnlayer)
        self.feature_dim = feature_dim
        self.clshead = nn.Sequential(
            #nn.GELU(),
            #nn.Dropout(p=0.5),
            nn.Linear(in_features=feature_dim, out_features=num_class),
            #nn.Softmax(dim=-1)
        )
    
    def forward(self, x, forward_pass='default'):
        if forward_pass == 'default':
            features = self.backbone(x)
            res = [self.clshead(features)]
        elif forward_pass == 'backbone':
            res = self.backbone(x)

        elif forward_pass == 'head':
            out = [self.clshead(x)]

        elif forward_pass == 'return_all':
            features = self.backbone(x)
            res = {'features': features, 'output': [self.clshead(features)]}
        
        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))        

        return res
        
        
#==================================utils==================================#

def int2list(num, l:int=3, scale=1):
    if isinstance(num, int):
        numlist = [int(num*scale**i) for i in range(l)]
        return numlist
    else:
        return num

def getlist(data, l, scale=1):
    if isinstance(data, list) or isinstance(data, tuple):
        assert len(data)==l
    elif isinstance(data, int):
        data = int2list(data, l, scale)
    else:
        TypeError('data must be one of (tuple, list, int)')
    return data







