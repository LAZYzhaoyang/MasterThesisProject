"""
# author: Zhaoyang Li
# 2023 03 14
# Central South University
"""
from matplotlib.cbook import flatten
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from torch import einsum

import numpy as np
import os

from .PointSwinTransformer import Residual, PreNorm, FeedForward, PointPatchMerging, \
    PointPixelShuffle, PointSwinAttentionBlock, getlist, int2list

#===================================Empty Block===================================#
class emptyWSA(nn.Module):
    def __init__(self,
                 dim:int):
        super().__init__()      
        self.attn = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1)
    
    def forward(self, x):
        # x : [b, n, dim]
        x = x.permute(0,2,1)
        # x : [b, dim, n]
        x = self.attn(x)
        x = x.permute(0,2,1)
        # x : [b, n, dim]

        return x 

class emptyPointSwinBlock(nn.Module):
    def __init__(self, dim:int, mlp_dim:int):
        super().__init__()
        self.AttentionBlock = Residual(PreNorm(dim, emptyWSA(dim=dim)))
        self.MlpBlock = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))
        
    def forward(self, x):
        # x: [b, n, dim]
        # print('in emp attn',x.size())
        x = self.AttentionBlock(x)
        # x: [b, n, dim]
        # print('mid emp attn',x.size())
        x = self.MlpBlock(x)
        # print('out emp attn',x.size())
        # x: [b, n, dim]
        return x

class emptyPointShuffle(nn.Module):
    def __init__(self,
                 in_channel:int,
                 out_channel:int,
                 up_scale_factor:int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=up_scale_factor)
        self.channel = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=1)
        
    def forward(self, x):
        # x: [b, in_C, n]
        # print('in emp shuffle',x.size())
        x = self.up(x)
        # print('up emp shuffle',x.size())
        x = self.channel(x)
        # print('out emp shuffle',x.size())
        return x

class emptyPointMerging(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 downscaling_factor:int):
        super().__init__()
        self.down = nn.MaxPool1d(kernel_size=downscaling_factor)
        self.channel = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
    
    def forward(self, x):
        # x: [b, in_C, n]
        # print('in emp merging',x.size())
        x = self.down(x)
        # print('down emp merging',x.size())
        x = self.channel(x)
        # print('out emp merging',x.size())
        return x

#===================================Stage Block===================================#

class emptyPointSwinAttentionBlock(nn.Module):
    def __init__(self, dim:int, mlp_dim=None):
        super().__init__()
        if mlp_dim is None:
            mlp_dim = dim*4
        self.wmha = emptyPointSwinBlock(dim=dim, mlp_dim=mlp_dim)
        
        self.swmha = emptyPointSwinBlock(dim=dim, mlp_dim=mlp_dim)
        
    def forward(self, x):
        # x: [b, n, dim]
        x = self.wmha(x)
        # x: [b, n, dim]
        x = self.swmha(x)
        # x: [b, n, dim]
        return x


class emptyPointSwinEncoderLayer(nn.Module):
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int, 
                 downscaling_factor:int, 
                 layers:int, 
                 heads:int, 
                 head_dim:int, 
                 window_size:int, 
                 empty_attn:bool=True,
                 empty_downsample:bool=True,
                 mlp_dim=None):
        super().__init__()
        self.attentions = nn.ModuleList([])
        for i in range(layers):
            if empty_attn:
                self.attentions.append(emptyPointSwinAttentionBlock(dim=in_channels, mlp_dim=mlp_dim))
            else:
                self.attentions.append(PointSwinAttentionBlock(dim=in_channels, 
                                                               heads=heads, 
                                                               head_dim=head_dim,
                                                               window_size=window_size, 
                                                               mlp_dim=mlp_dim))
        if empty_downsample:
            self.downsample = emptyPointMerging(in_channels=in_channels, out_channels=out_channels, downscaling_factor=downscaling_factor)
        else:
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
    

class emptyPointSwinDecoderLayer(nn.Module):
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int, 
                 up_scale_factor:int, 
                 layers:int, 
                 heads:int, 
                 head_dim:int, 
                 window_size:int, 
                 empty_attn:bool=True,
                 empty_upsample:bool=True,
                 mlp_dim=None):
        super().__init__()
        if empty_upsample:
            self.upsample = emptyPointShuffle(in_channel=in_channels, out_channel=out_channels, up_scale_factor=up_scale_factor)
        else:
            self.upsample = PointPixelShuffle(in_channel=in_channels, out_channel=out_channels, up_scale_factor=up_scale_factor)
        
        self.attentions = nn.ModuleList([])
        for i in range(layers):
            if empty_attn:
                self.attentions.append(emptyPointSwinAttentionBlock(dim=out_channels, mlp_dim=mlp_dim))
            else:
                self.attentions.append(PointSwinAttentionBlock(dim=out_channels, 
                                                               heads=heads, 
                                                               head_dim=head_dim,
                                                               window_size=window_size, 
                                                               mlp_dim=mlp_dim))
        
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


#===================================Empty Point Swin Block===================================#

class emptyPointSwinEncoder(nn.Module):
    def __init__(self, 
                 in_channels:int=32, 
                 stage_num:int=3, 
                 downscaling_factor=4, 
                 layernums=1, 
                 heads=8, 
                 head_dims=32, 
                 window_sizes=4, 
                 mlp_dims=None, 
                 empty_attn:bool=True,
                 empty_downsample:bool=True,
                 attn_layers=4):
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
            self.encoders.append(emptyPointSwinEncoderLayer(in_channels=self.inC[i], 
                                                            out_channels=self.outC[i],
                                                            downscaling_factor=self.downf[i], 
                                                            layers=self.layers[i],
                                                            heads=self.heads[i], 
                                                            head_dim=self.head_dims[i],
                                                            window_size=self.window_sizes[i], 
                                                            empty_attn=empty_attn,
                                                            empty_downsample=empty_downsample,
                                                            mlp_dim=self.mlp_dims[i])) 
        #self.attn = nn.Sequential()
        self.attns = nn.ModuleList([])
        
        for i in range(attn_layers):
            if empty_attn:
                self.attns.append(emptyPointSwinAttentionBlock(dim=self.outC[-1], mlp_dim=self.mlp_dims[-1]))
            else:
                self.attns.append(PointSwinAttentionBlock(dim=self.outC[-1],  
                                                          heads=self.heads[-1], 
                                                          window_size=self.window_sizes[-1], 
                                                          head_dim=head_dims,
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

class emptyPointSwinDecoder(nn.Module):
    def __init__(self, 
                 out_channels:int=32, 
                 stage_num:int=3, 
                 upscaling_factor=4, 
                 layernums=1, 
                 heads=8, 
                 head_dims=32, 
                 window_sizes=4, 
                 empty_attn:bool=True,
                 empty_upsample:bool=True,
                 mlp_dims=None):
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
            self.decoders.append(emptyPointSwinDecoderLayer(in_channels=self.inC[i], 
                                                            out_channels=self.outC[i],
                                                            up_scale_factor=self.upf[i], 
                                                            layers=self.layers[i],
                                                            heads=self.heads[i], 
                                                            head_dim=self.head_dims[i], 
                                                            window_size=self.window_sizes[i],
                                                            empty_attn=empty_attn,
                                                            empty_upsample=empty_upsample,
                                                            mlp_dim=self.mlp_dims[i]))
            
    def forward(self, init_feature, features):
        out = self.decoders[0](init_feature)
        for i in range(len(self.decoders)-1):
            # print(features[i].size())
            out = out+features[i]
            out = self.decoders[i+1](out)
        return out

#===================================Response Proxy Model===================================#

class emptyResponsePointSwinEmbedding(nn.Module):
    def __init__(self, 
                 channels:int=3, 
                 embedding_dim:int=32, 
                 param_dim:int=8, 
                 patch_size:int=4, 
                 empty_downsample:bool=True):
        super().__init__()
        self.channels = channels
        self.embedding_dim = embedding_dim
        if empty_downsample:
            self.point_embed = emptyPointMerging(in_channels=channels, out_channels=embedding_dim,downscaling_factor=patch_size)
        else:
            self.point_embed = PointPatchMerging(in_channels=channels, out_channels=embedding_dim, downscaling_factor=patch_size)
        self.param_embed = nn.Linear(in_features=param_dim, out_features=embedding_dim)
        
    def forward(self,x, param):
        # x: [b, c, n]
        # param: [b, pd]
        point_embedding = self.point_embed(x)
        param_embedding = self.param_embed(param)
        # print('point emb:',point_embedding.size())
        # print('param emb:', param_embedding.size())
        
        # point_embed: [b, embeddim, n//patch_size]
        # param_embed: [b, embeddim]
        b, h = param_embedding.size()
        param_embedding = param_embedding.view(b,h,1)
        
        feature = point_embedding + param_embedding
        # print('feature:', feature.size())
        
        return feature, param_embedding
        
class emptyResponsePointSwinHead(nn.Module):
    def __init__(self, 
                 embedding_dim:int=32, 
                 pred_dim:int=15, 
                 patch_size:int=4, 
                 response_dim:int=2, 
                 mlp_dim:int=64, 
                 empty_attn:bool=True,
                 empty_upsample:bool=True):
        super().__init__()
        self.pointlayer = emptyPointSwinDecoderLayer(in_channels=embedding_dim, 
                                                     out_channels=pred_dim, 
                                                     up_scale_factor=patch_size,
                                                     layers=1, 
                                                     heads=8, 
                                                     head_dim=32, 
                                                     window_size=16, 
                                                     empty_attn=empty_attn,
                                                     empty_upsample=empty_upsample,
                                                     mlp_dim=64)
        
        #self.pooling = nn.Conv1d(in_channels=embedding_dim, out_channels=mlp_dim, kernel_size=)
        self.pooling1 = nn.AdaptiveAvgPool1d(256)
        
        self.pooling2 = nn.Sequential(nn.Conv1d(in_channels=256, out_channels=256, kernel_size=embedding_dim, stride=1, padding=0),
                                      nn.Flatten(),)
        
        self.res_emb = nn.Sequential(nn.Flatten(),
                                     nn.Linear(in_features=embedding_dim, out_features=256),
                                     nn.GELU(),
                                     nn.Linear(in_features=256, out_features=256),
                                     nn.GELU())
        
        self.res_head = nn.Sequential(nn.Linear(in_features=256, out_features=mlp_dim),
                                      nn.GELU(),
                                      #nn.ReLU(),
                                      nn.Linear(in_features=mlp_dim, out_features=response_dim),
                                      nn.Sigmoid())
        
    def forward(self, fea, patch_emb, param_emb):
        # fea : [b, embedding_dim, point_size/up_scale_factor]
        # patch_emb: same size as fea
        # param_emb: [b, embedding_dim, 1]

        x = fea + patch_emb
        
        pred_point = self.pointlayer(x)
        
        res_emb = self.res_emb(param_emb)
        
        res_fea = self.pooling1(x)
        res_fea = res_fea.permute(0,2,1)
        res_fea = self.pooling2(res_fea)
        
        res_fea = res_fea + res_emb
        pred_res = self.res_head(res_fea)
        
        return pred_point, pred_res

class emptyResponsePointSwinTransformerProxyModel(nn.Module):
    def __init__(self, 
                 in_channels:int=3, 
                 out_channels:int=15, 
                 param_dim:int=8, 
                 res_dim:int=2, 
                 embed_dim:int=32, 
                 scale_factor:int=4,
                 stage_num:int=3, 
                 layers_num:int=1, 
                 heads:int=8,
                 head_dims:int=32, 
                 window_size:int=4, 
                 attn_layers:int=4, 
                 empty_attn:bool=True,
                 empty_upsample:bool=True,
                 empty_downsample:bool=True,
                 mlp_dim=None):
        super().__init__()
        self.embedding = emptyResponsePointSwinEmbedding(channels=in_channels, 
                                                         embedding_dim=embed_dim, 
                                                         param_dim=param_dim, 
                                                         patch_size=scale_factor,
                                                         empty_downsample=empty_downsample)
        self.encoder = emptyPointSwinEncoder(in_channels=embed_dim, 
                                             stage_num=stage_num, 
                                             downscaling_factor=scale_factor, 
                                             layernums=layers_num,
                                             heads=heads, 
                                             head_dims=head_dims, 
                                             window_sizes=window_size, 
                                             mlp_dims=mlp_dim, 
                                             attn_layers=attn_layers,
                                             empty_downsample=empty_downsample,
                                             empty_attn=empty_attn)
        self.decoder = emptyPointSwinDecoder(out_channels=embed_dim, 
                                             stage_num=stage_num, 
                                             upscaling_factor=scale_factor, 
                                             layernums=layers_num,
                                             heads=heads, 
                                             head_dims=head_dims, 
                                             window_sizes=window_size, 
                                             empty_attn=empty_attn,
                                             empty_upsample=empty_upsample,
                                             mlp_dims=mlp_dim)
        self.head = emptyResponsePointSwinHead(embedding_dim=embed_dim, 
                                               pred_dim=out_channels, 
                                               patch_size=scale_factor, 
                                               response_dim=res_dim, 
                                               empty_upsample=empty_upsample, 
                                               empty_attn=empty_attn)
    
    def forward(self, point, param, return_feature=False):
        embed_fea, param_emb = self.embedding(point, param)
        init_fea, features = self.encoder(embed_fea)
        fea = self.decoder(init_fea, features)
        pred_point, pred_res = self.head(fea, embed_fea, param_emb)
                        
        all_feas = [embed_fea, features, init_fea, fea]
        if return_feature:
            return pred_point, pred_res, all_feas
        else:
            return pred_point, pred_res

