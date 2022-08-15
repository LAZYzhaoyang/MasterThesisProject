'''
# Author: Zhaoyang Li
# Date: 2022-04-03
# Central South University
'''
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
import os

#============================utils============================#
# reference https://github.com/yanx27/Pointnet_Pointnet2_pytorch, modified by Zhaoyang Li


def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False, knn=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint]
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()
    if knn:
        dists = square_distance(new_xyz, xyz)  # B x npoint x N
        idx = dists.argsort()[:, :, :nsample]  # B x npoint x K
    else:
        idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) 
        # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, knn=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.knn = knn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, C]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points_concat: sample points feature data, [B, S, D']
        """
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, knn=self.knn)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0].transpose(1, 2)
        return new_xyz, new_points

class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list, knn=False):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.knn = knn
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points, seed_idx=None):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S) if seed_idx is None else seed_idx)
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            if self.knn:
                dists = square_distance(new_xyz, xyz)  # B x npoint x N
                group_idx = dists.argsort()[:, :, :K]  # B x npoint x K
            else:
                group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_points_concat = torch.cat(new_points_list, dim=1).transpose(1, 2)
        return new_xyz, new_points_concat
    
# NoteL this function swaps N and C
class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points



#============================PointTransformer Block============================#
class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k
        
        # xyz: b x n x 3, features: b x n x f
    
    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)
        
        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f
        
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
        
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre
        return res, attn
    
    
class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)
        
    def forward(self, xyz, points):
        return self.sa(xyz, points)


class TransitionUp(nn.Module):
    def __init__(self, dim1, dim2, dim_out):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)

        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(dim1, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(dim2, dim_out),
            SwapAxes(),
            nn.BatchNorm1d(dim_out),  # TODO
            SwapAxes(),
            nn.ReLU(),
        )
        self.fp = PointNetFeaturePropagation(-1, [])
    
    def forward(self, xyz1, points1, xyz2, points2):
        feats1 = self.fc1(points1)
        feats2 = self.fc2(points2)
        feats1 = self.fp(xyz2.transpose(1, 2), xyz1.transpose(1, 2), None, feats1.transpose(1, 2)).transpose(1, 2)
        return feats1 + feats2
        
#============================Original PointTransformer============================#
# class Backbone(nn.Module):
#     def __init__(self, in_channels=12, npoints=1024, nblocks=4, nneighbor=16, transformer_dim=512):
#         super().__init__()
#         #npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
#         self.fc1 = nn.Sequential(
#             nn.Linear(in_channels, 32),
#             nn.ReLU(),
#             nn.Linear(32, 32)
#         )
#         self.transformer1 = TransformerBlock(32, transformer_dim, nneighbor)
#         self.transition_downs = nn.ModuleList()
#         self.transformers = nn.ModuleList()
#         for i in range(nblocks):
#             channel = 32 * 2 ** (i + 1)
#             self.transition_downs.append(TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
#             self.transformers.append(TransformerBlock(channel, transformer_dim, nneighbor))
#         self.nblocks = nblocks
    
#     def forward(self, x):
#         xyz = x[..., :3]
#         points = self.transformer1(xyz, self.fc1(x))[0]

#         xyz_and_feats = [(xyz, points)]
#         for i in range(self.nblocks):
#             xyz, points = self.transition_downs[i](xyz, points)
#             points = self.transformers[i](xyz, points)[0]
#             xyz_and_feats.append((xyz, points))
#         return points, xyz_and_feats



# class PointTransformerCls(nn.Module):
#     def __init__(self, in_channels=3, npoints=1024, nblocks=4, nneighbor=16, n_c=50, transformer_dim=512):
#         super().__init__()
#         self.backbone = Backbone(in_channels=in_channels, 
#                                  npoints=npoints, 
#                                  nblocks=nblocks, 
#                                  nneighbor=nneighbor, 
#                                  transformer_dim=transformer_dim)
#         #npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
#         self.fc2 = nn.Sequential(
#             nn.Linear(32 * 2 ** nblocks, 256),
#             nn.ReLU(),
#             nn.Linear(256, 64),
#             nn.ReLU(),
#             nn.Linear(64, n_c)
#         )
#         self.nblocks = nblocks
    
#     def forward(self, x):
#         points, _ = self.backbone(x)
#         print('points: {}'.format(points.size()))
#         print('points.mean(1): {}'.format(points.mean(1).size()))
#         res = self.fc2(points.mean(1))
#         return res


# class PointTransformerSeg(nn.Module):
#     def __init__(self, in_channels=3, npoints=1024, nblocks=4, nneighbor=16, n_c=50, transformer_dim=512):
#         super().__init__()
#         self.backbone = Backbone(in_channels=in_channels, 
#                                  npoints=npoints, 
#                                  nblocks=nblocks, 
#                                  nneighbor=nneighbor, 
#                                  transformer_dim=transformer_dim)
#         #npoints, nblocks, nneighbor, n_c, d_points = cfg.num_point, cfg.model.nblocks, cfg.model.nneighbor, cfg.num_class, cfg.input_dim
#         self.fc2 = nn.Sequential(
#             nn.Linear(32 * 2 ** nblocks, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 32 * 2 ** nblocks)
#         )
#         self.transformer2 = TransformerBlock(32 * 2 ** nblocks, transformer_dim, nneighbor)
#         self.nblocks = nblocks
#         self.transition_ups = nn.ModuleList()
#         self.transformers = nn.ModuleList()
#         for i in reversed(range(nblocks)):
#             channel = 32 * 2 ** i
#             self.transition_ups.append(TransitionUp(channel * 2, channel, channel))
#             self.transformers.append(TransformerBlock(channel, transformer_dim, nneighbor))

#         self.fc3 = nn.Sequential(
#             nn.Linear(32, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, n_c)
#         )
    
#     def forward(self, x):
#         print('x.size: {}'.format(x.size()))
        
#         points, xyz_and_feats = self.backbone(x)
#         xyz = xyz_and_feats[-1][0]
        
#         print('points, xyz: ')
#         print(points.size(), xyz.size())
        
#         if type(xyz_and_feats)==list:
#             print('len xyz_and_feats: {}'.format(len(xyz_and_feats)))
#             for i in range(len(xyz_and_feats)):
#                 print('len xyz_and_feats[{}]: {}'.format(i, len(xyz_and_feats[i])))
#                 print('xyz_and_feats[{}][0]: {}, xyz_and_feats[{}][1]: {}'.format(i, xyz_and_feats[i][0].size(), i, xyz_and_feats[i][1].size()))
        
#         points = self.transformer2(xyz, self.fc2(points))[0]
        
#         print('point after transformer2: {}'.format(points.size()))
#         print('for each blocks: ')

#         for i in range(self.nblocks):
#             print('block {}'.format(i))
            
#             points = self.transition_ups[i](xyz, points, xyz_and_feats[- i - 2][0], xyz_and_feats[- i - 2][1])
            
#             print('points: {}'.format(points.size()))
            
#             xyz = xyz_and_feats[- i - 2][0]
            
#             print('xyz: {}'.format(xyz.size()))
            
#             points = self.transformers[i](xyz, points)[0]
            
#             print('points: {}'.format(points.size()))
        
#         res = self.fc3(points)
#         return res
    
#============================My PointTransformer Block============================#
class ResponsePointTransEmbedding(nn.Module):
    def __init__(self, in_channels=3, param_dim=8, embedding_dim=32, npoint=1024):
        super().__init__()
        assert in_channels>=3
        self.fc_x = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim//2)
        )
        self.fc_p = nn.Sequential(
            nn.Linear(param_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim//2)
        )
        self.fc_p_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=256, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv1d(in_channels=256, out_channels=npoint, kernel_size=5, padding=2)
        )
        self.npoint = npoint
        
    def forward(self, x, params):
        # x: [b, n, in_channels]
        # params: [b, param_dim]
        xyz = x[..., :3]
        point_embed = self.fc_x(x) # [b, n, embedding_dim]
        _,n,_ =point_embed.size()
        param_embed = self.fc_p(params) # [b, embedding_dim]
        #param_embed=torch.unsqueeze(param_embed, dim=1).repeat(1,n,1)
        param_embed=torch.unsqueeze(param_embed, dim=1)
        param_embed = self.fc_p_conv(param_embed)
        
        #embedding_fea = point_embed+param_embed
        #print(point_embed.size())
        #print(param_embed.size())
        embedding_fea = torch.cat((point_embed, param_embed), dim=-1)
        
        return xyz, embedding_fea
        

class ResponsePointTransHead(nn.Module):
    def __init__(self, embedding_dim=32, out_channels=15, res_dim=8):
        super().__init__()
        self.head_p = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.Tanh(),
            nn.Linear(64, res_dim)
        )
        
        self.head_x = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.Tanh(),
            nn.Linear(64, out_channels)
        )
        
    def forward(self, point):
        # points: [b, n, embedding_dim]
        pointcloud = self.head_x(point)
        res = self.head_p(point.mean(1))
        #out = {'PointCloud':pointcloud, 'Params':param}
        
        return pointcloud, res

class ResponsePointTransConvHead(nn.Module):
    def __init__(self, embedding_dim=32, out_channels=15, npoint=4096, res_dim=8):
        super().__init__()
        self.head_x = nn.Sequential(
            nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=17, padding=8),
            nn.Tanh(),
            nn.Conv1d(in_channels=64, out_channels=out_channels, kernel_size=17, padding=8)
        )
        
        self.head_p = nn.Sequential(
            nn.Linear(npoint, 64),
            nn.Tanh(),
            nn.Linear(64, res_dim)
        )
        
    def forward(self, point):
        # points: [b, n, embedding_dim]
        cpoint = point.permute(0,2,1)
        pointcloud = self.head_x(cpoint)
        #print(point.size())
        #print(point.mean(-1).size())
        res = self.head_p(point.mean(-1))
        #out = {'PointCloud':pointcloud, 'Params':param}
        pointcloud = pointcloud.permute(0,2,1)
        
        return pointcloud, res

class PointTransformerEmbedding(nn.Module):
    def __init__(self, in_channels=15, embedding_dim=32):
        super().__init__()
        assert in_channels>=3
        # self.fc = nn.Sequential(
        #     nn.Linear(in_channels, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, embedding_dim)
        # )
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=33, padding=16),
            nn.Tanh(),
            nn.Conv1d(in_channels=64, out_channels=embedding_dim, kernel_size=33, padding=16)
        )
    def forward(self, x):
        xyz = x[..., :3]
        x =  x.permute(0,2,1) # [b,c,n]
        embedding_fea = self.fc(x)
        embedding_fea = embedding_fea.permute(0,2,1) #[b,n,c]
        return xyz, embedding_fea
    

class PointTransformerClsHead(nn.Module):
    def __init__(self, embedding_dim=32, nblocks=4, n_c=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2 ** nblocks, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )
    def forward(self, points):
        # points: [b, n, fea_dim *2 **nblocks]
        res = self.fc(points.mean(1))
        return res

class PointTransformerSegHead(nn.Module):
    def __init__(self, embedding_dim=32, n_c=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_c)
        )
    def forward(self, points):
        # points: [b, n, fea_dim]
        res = self.fc(points)
        return res

class PointTransformerEncoder(nn.Module):
    def __init__(self, embedding_dim=32, npoints=1024, nblocks=4, nneighbor=16, transformer_dim=256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.transformer1 = TransformerBlock(embedding_dim, transformer_dim, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = embedding_dim * 2 ** (i + 1)
            self.transition_downs.append(TransitionDown(npoints // 4 ** (i + 1), nneighbor, [channel // 2 + 3, channel, channel]))
            self.transformers.append(TransformerBlock(channel, transformer_dim, nneighbor))
        self.nblocks = nblocks
    
    def forward(self, embedding_feature, xyz):
        # embedding_feature: [b, n, fea_dim]
        # xyz: [b, n, 3]
        # return:
        #       points: [b, n//(4**nblocks),  32*2**nblocks]
        #       xyz_and_feats: [xyz[nblocks+1], feats[nblocks+1]]
        #       for i in range(nblocks+1): 0,1,...,nblocks
        #       xyz[i]: [b, n//(4**i), 3]
        #       feats[i]: [b, n//(4**i), 32*2**i]
        points = self.transformer1(xyz, embedding_feature)[0]

        xyz_and_feats = [(xyz, points)]
        for i in range(self.nblocks):
            xyz, points = self.transition_downs[i](xyz, points)
            points = self.transformers[i](xyz, points)[0]
            xyz_and_feats.append((xyz, points))
        return points, xyz_and_feats

class PointTransformerDecoder(nn.Module):
    def __init__(self, embedding_dim=32,  nblocks=4, nneighbor=16, transformer_dim=256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2 ** nblocks, transformer_dim),
            nn.ReLU(),
            nn.Linear(transformer_dim, transformer_dim),
            nn.ReLU(),
            nn.Linear(transformer_dim, embedding_dim * 2 ** nblocks)
        )
        self.transformer = TransformerBlock(embedding_dim * 2 ** nblocks, transformer_dim, nneighbor)
        self.nblocks = nblocks
        self.transition_ups = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in reversed(range(nblocks)):
            channel = embedding_dim * 2 ** i
            self.transition_ups.append(TransitionUp(channel * 2, channel, channel))
            self.transformers.append(TransformerBlock(channel, transformer_dim, nneighbor))

        
    def forward(self, points, xyz_and_feats):
        # points: [b, n//(4**nblocks),  32*2**nblocks]
        # xyz_and_feats: [xyz[nblocks+1], feats[nblocks+1]]
        # for i in range(nblocks+1): 0,1,...,nblocks
        # xyz[i]: [b, n//(4**i), 3]
        # feats[i]: [b, n//(4**i), 32*2**i]
        
        # return:
        # fea: [b, n, embedding_dim]
        
        xyz = xyz_and_feats[-1][0]
        points = self.transformer(xyz, self.fc(points))[0]
        for i in range(self.nblocks):
            points = self.transition_ups[i](xyz, points, xyz_and_feats[- i - 2][0], xyz_and_feats[- i - 2][1])
            xyz = xyz_and_feats[- i - 2][0]
            points = self.transformers[i](xyz, points)[0]
        
        fea = points
        return fea

#============================My PointTransformer============================#

# class PointTransformerCls(nn.Module):
#     def __init__(self, in_channels=15, num_class=4, embedding_dim=32, 
#                  npoints=1024, nblocks=4, nneighbor=16, transformer_dim=128):
#         super().__init__()
#         self.embedding = PointTransformerEmbedding(in_channels=in_channels, embedding_dim=embedding_dim)
#         self.encoder = PointTransformerEncoder(embedding_dim=embedding_dim,
#                                                npoints=npoints,
#                                                nblocks=nblocks,
#                                                nneighbor=nneighbor,
#                                                transformer_dim=transformer_dim)
#         self.clshead = PointTransformerClsHead(embedding_dim=embedding_dim,
#                                                nblocks=nblocks,
#                                                n_c=num_class)
        
#     def forward(self, x, forward_pass='default'):
#         xyz, embedding_fea = self.embedding(x)
#         features, _ = self.encoder(embedding_fea, xyz)
#         res = self.clshead(features)
#         return res
    
class PointTransformerBackbone(nn.Module):
    def __init__(self, in_channels=15, feature_dim=128, embedding_dim=32, 
                 npoints=1024, nblocks=4, nneighbor=16, transformer_dim=128):
        super().__init__()
        self.embedding = PointTransformerEmbedding(in_channels=in_channels, embedding_dim=embedding_dim)
        self.encoder = PointTransformerEncoder(embedding_dim=embedding_dim,
                                               npoints=npoints,
                                               nblocks=nblocks,
                                               nneighbor=nneighbor,
                                               transformer_dim=transformer_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2 ** nblocks, feature_dim*2),
            #nn.Dropout(p=0.5),
            nn.BatchNorm1d(feature_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim*2, feature_dim),
            nn.BatchNorm1d(feature_dim, affine=False),
        )
    def forward(self, x):
        # x:[b dim n]
        #print(x.shape)
        b, dim, n = x.shape
        
        x = x.permute(0, 2, 1)
        xyz, embedding_fea = self.embedding(x)
        #print(xyz.shape, embedding_fea.shape)
        points, _ = self.encoder(embedding_fea, xyz)
        #print(points.shape)
        #points = points.permute(0, 2, 1)
        res = self.fc(points.mean(1))
        return res
    
class PointTransformerCls(nn.Module):
    def __init__(self, in_channels=15, num_class=4, feature_dim=128, embedding_dim=32, 
                 npoints=1024, nblocks=4, nneighbor=16, transformer_dim=128):
        super().__init__()
        self.backbone = PointTransformerBackbone(in_channels=in_channels,
                                                 feature_dim=feature_dim,
                                                 embedding_dim=embedding_dim,
                                                 npoints=npoints, nblocks=nblocks,
                                                 nneighbor=nneighbor, 
                                                 transformer_dim=transformer_dim)
        self.feature_dim = feature_dim
        self.clshead = nn.Sequential(
            #nn.Dropout(p=0.5),
            nn.Linear(feature_dim, num_class),
            #nn.Softmax(dim=-1)
        )        
    def forward(self, x, forward_pass='default'):
        if forward_pass == 'default':
            features = self.backbone(x)
            #print(features.shape)
            res = [self.clshead(features)]
        elif forward_pass == 'backbone':
            res = self.backbone(x)

        elif forward_pass == 'head':
            res = [self.clshead(x)]

        elif forward_pass == 'return_all':
            features = self.backbone(x)
            res = {'features': features, 'output': [self.clshead(features)]}
        
        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))        

        return res
    
# class PointTransformerSeg(nn.Module):
#     def __init__(self,  in_channels=3, num_class=4, embedding_dim=32, npoints=1024, nblocks=4, nneighbor=16, transformer_dim=128):
#         super().__init__()
#         self.embedding = PointTransformerEmbedding(in_channels=in_channels, embedding_dim=embedding_dim)
#         self.encoder = PointTransformerEncoder(embedding_dim=embedding_dim,
#                                                npoints=npoints,
#                                                nblocks=nblocks,
#                                                nneighbor=nneighbor,
#                                                transformer_dim=transformer_dim)
#         self.decoder = PointTransformerDecoder(embedding_dim=embedding_dim,
#                                                nblocks=nblocks,
#                                                nneighbor=nneighbor,
#                                                transformer_dim=transformer_dim)
#         self.seghead = PointTransformerSegHead(embedding_dim=embedding_dim,
#                                                n_c=num_class)
        
#     def forward(self, x):
#         xyz, embedding_fea =  self.embedding(x)
#         points, xyz_and_feats = self.encoder(embedding_fea, xyz)
#         fea = self.decoder(points, xyz_and_feats)
#         res = self.seghead(fea)
        
#         return res


class ResponsePointTransformerProxyModel(nn.Module):
    def __init__(self, in_channels=3, param_dim=8, res_dim=2, embedding_dim=32, out_channels=15, npoints=4096, nblocks=4, nneighbor=16, transformer_dim=128):
        super().__init__()
        self.embedding = ResponsePointTransEmbedding(in_channels=in_channels, param_dim=param_dim, embedding_dim=embedding_dim, npoint=npoints)
        self.encoder =  PointTransformerEncoder(embedding_dim=embedding_dim,
                                               npoints=npoints,
                                               nblocks=nblocks,
                                               nneighbor=nneighbor,
                                               transformer_dim=transformer_dim)
        self.decoder = PointTransformerDecoder(embedding_dim=embedding_dim,
                                               nblocks=nblocks,
                                               nneighbor=nneighbor,
                                               transformer_dim=transformer_dim)
        #self.response_head = ResponsePorxyHead(embedding_dim=embedding_dim, out_channels=out_channels, res_dim=res_dim)
        self.head = ResponsePointTransConvHead(embedding_dim=embedding_dim, out_channels=out_channels, npoint=npoints, res_dim=res_dim)
        self.frame = out_channels//in_channels
        
    def forward(self, x, params):
        # x: [b,c,n]
        # params: [b,8]
        x =  x.permute(0,2,1) # [b,n,c]
        xyz, embedding_fea = self.embedding(x, params)
        points, xyz_and_feats = self.encoder(embedding_fea, xyz)
        fea = self.decoder(points, xyz_and_feats)
        pc, res = self.head(fea)
        
        pc = pc+x.repeat(1,1,self.frame)
        
        pc = pc.permute(0,2,1) # [b,c,n]
        return pc, res



