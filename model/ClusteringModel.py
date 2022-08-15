"""
# author: Zhaoyang Li
# 2022 08 15
# Central South University
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import getBackbone

import sklearn
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import numpy as np
from ..utils.toollib import ToNumpy,ToTensor

class ContrastiveModel(nn.Module):
    def __init__(self, backbone, head='mlp', features_dim=128):
        super(ContrastiveModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.head = head
 
        if head == 'linear':
            self.contrastive_head = nn.Linear(self.backbone_dim, features_dim)

        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                    nn.Linear(self.backbone_dim, self.backbone_dim//2),
                    nn.BatchNorm1d(self.backbone_dim//2),
                    nn.ReLU(inplace=True), 
                    nn.Linear(self.backbone_dim//2, features_dim))
        
        else:
            raise ValueError('Invalid head {}'.format(head))

    def forward(self, x, forward_pass='default'):
        if forward_pass == 'default':
            features = self.contrastive_head(self.backbone(x))
            features = F.normalize(features, dim = 1)
        elif forward_pass == 'backbone':
            features = self.backbone(x)
            features = F.normalize(features, dim = 1)
        elif forward_pass == 'head':
            features = self.contrastive_head(x)
            features = F.normalize(features, dim = 1)
        elif forward_pass == 'return_all':
            fea = self.backbone(x)
            out = F.normalize(self.contrastive_head(fea),dim=1)
            features = {'features':fea, 'head_feature':out}
        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))   
        return features


class ClusteringModel(nn.Module):
    def __init__(self, backbone, nclusters, nheads=1):
        super(ClusteringModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.nheads = nheads
        assert(isinstance(self.nheads, int))
        assert(self.nheads > 0)
        self.cluster_head = nn.ModuleList([nn.Linear(self.backbone_dim, nclusters) for _ in range(self.nheads)])

    def forward(self, x, forward_pass='default'):
        if forward_pass == 'default':
            features = self.backbone(x)
            #features = F.normalize(features, dim = 1)
            out = [cluster_head(features) for cluster_head in self.cluster_head]

        elif forward_pass == 'backbone':
            out = self.backbone(x)
            #out = F.normalize(out, dim = 1)

        elif forward_pass == 'head':
            out = [cluster_head(x) for cluster_head in self.cluster_head]

        elif forward_pass == 'return_all':
            features = self.backbone(x)
            #features = F.normalize(features, dim = 1)
            out = {'features': features, 'output': [cluster_head(features) for cluster_head in self.cluster_head]}
        
        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))        

        return out



        
class DeepClusterCenter:
    def __init__(self, nclusters:int, loader:torch.utils.data.DataLoader, 
                 backbone_dim=128, beta=0.99, min_dis=5):
        super().__init__()
        self.nclusters = nclusters
        self.backbone_dim = backbone_dim
        self.min_dis = min_dis
        
        self.beta = beta
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        b = len(loader.sampler)
        self.fea_num = b
        self.cluster_centers = torch.randn((nclusters, backbone_dim), requires_grad=False)
        self.features = torch.randn((b, backbone_dim), requires_grad=False)
        self.prob = torch.rand((b,nclusters), requires_grad=False)
        self.pseudo_labs = torch.argmax(self.prob,dim=-1)
        self.ptr = 0
        self._to_device()
        
    def selflabel(self, fea:torch.Tensor, aug_fea:torch.Tensor):
        # fea, aug_fea: [batch, backbone_dim]
        # return:
        #   pseudo_lab: [batch, nclusters]
        b = fea.size(0)
        pseudo_lab = self.getPseudoLabel(fea)
        self.features[self.ptr:self.ptr+b] = aug_fea
        self.ptr+=b
        if self.ptr>=self.fea_num-1:
            self.ptr=0
        
        return pseudo_lab        
        
    def init_kmeans_center(self, net, loader:torch.utils.data.DataLoader):
        b = len(loader.sampler)
        bts = loader.batch_size
        ptr = 0
        if self.features is None:
            self.features = torch.zeros((b, self.backbone_dim))
            self.prob = torch.zeros((b, self.nclusters))
        for i,batch in enumerate(loader):
            nodes = ToTensor(batch['node']).to(self.device)
            #aug_node = ToTensor(batch['node_augmented']).to(self.device)
            net.eval()
            
            with torch.no_grad():
                fea1 = net.backbone(nodes)
                #fea2 = net.backbone(aug_node)
                fea1 = fea1.detach_()
                #fea2 = fea2.detach_()
            self.features[ptr:ptr+fea1.size(0)].copy_(fea1)
            ptr+=fea1.size(0)
        feas = ToNumpy(self.features)
        kmeansmodel = KMeans(n_clusters=self.nclusters, max_iter=10000)
        kmeansmodel.fit(feas)
        
        centers = kmeansmodel.cluster_centers_
        self.cluster_centers = ToTensor(centers)
        self.prob = self.getPseudoLabel(self.features)
        self.pseudo_labs = torch.argmax(self.prob, dim=-1)
        self._to_device()
            
    def update_center(self):
        for i in range(self.nclusters):
            new_center = torch.zeros_like(self.cluster_centers)
            if len(self.features[self.pseudo_labs==i])==0:
                center = torch.unsqueeze(self.cluster_centers, dim=0)
                unseqx = torch.unsqueeze(self.features, dim=1)
                dis = torch.sqrt(torch.sum((unseqx-center)**2,dim=-1))
                _, min_indices = torch.min(dis,dim=0)[i]
                new_center[i] = self.features[min_indices]
            else:
                c = torch.mean(self.features[self.pseudo_labs==i], dim=0)
                new_center[i]=c
        self.cluster_centers = self.beta*self.cluster_centers+(1-self.beta)*new_center
            
    def getPseudoLabel(self, x:torch.Tensor):
        with torch.no_grad():
            center = torch.unsqueeze(self.cluster_centers, dim=0).to(device=self.device)
            unseqx = torch.unsqueeze(x, dim=1).to(device=self.device)
            dis = torch.sqrt(torch.sum((unseqx-center)**2,dim=-1))
            dis = torch.clamp(dis-self.min_dis, min=1e-6)
            
            pseudo_label = F.softmax(1/dis, dim=-1)
        return pseudo_label
    
    def to(self, device):
        self.device = device
        self._to_device()
        
    def _to_device(self):
        self.cluster_centers = self.cluster_centers.to(device=self.device)
        self.features = self.features.to(device=self.device)
        self.pseudo_labs = self.pseudo_labs.to(device=self.device)
        self.prob = self.prob.to(self.device)
        
         
        
        
#========================================utils========================================#
def getClusterModel(config):
    backbone = getBackbone(config=config['BackboneConfig'], ModelType=config['BackboneType'])
    if config['task'] in ['simclr', 'simsiam', 'byol', 'moco']:
         model = ContrastiveModel(backbone, head=config['head'], features_dim=config['contrastive_feadim'])
    elif config['task'] in ['scan', 'selflabel', 'deepcluster', 'spice']:
        if config['task'] in ['selflabel', 'spice']:
            assert(config['nheads'] == 1)
        model = ClusteringModel(backbone, config['num_cluster'], config['nheads'])
    else:
        raise ValueError('Invalid task {}'.format(config['task']))
    return model

def saveClusterModel(clustermodel, save_path:str, epoch:int, filename:str, optimizer=None):
    state = getState(clustermodel)
    state['epoch'] = epoch
    if optimizer is not None:
        state['optimizer']=optimizer.state_dict()
    else:
        state['optimizer']=None
    
    filename = os.path.join(save_path, filename)
    
    torch.save(state, filename)

def loadClusterModel(net, save_path:str, file_class:str, model_name:str='PointSwin', epoch:int=0, load_backbone_only=False):
    class_list=['best', 'last', 'epochs']
    assert file_class in class_list, 'file class must be one of [best, last, epoch]'
    if file_class==class_list[0]:
        filename = os.path.join(save_path, '{}_best.pth'.format(model_name))
    elif file_class==class_list[1]:
        filename = os.path.join(save_path, '{}_last_epoch.pth'.format(model_name))
    else:
        filename = os.path.join(save_path, '{}_checkpoint-epoch{}.pth'.format(model_name, epoch))
        
    ckpt = torch.load(filename)
    
    epoch_start = ckpt['epoch']
    net = loadParameters(net, ckpt, load_backbone_only=load_backbone_only)
    
    return net, epoch_start

def getState(model):
    if isinstance(model, ContrastiveModel):
        state = {'backbone': model.backbone.state_dict(),
                 'backbone_dim': model.backbone_dim,
                 'contrastive_head': model.contrastive_head.state_dict(),
                 'head': model.head}
    elif isinstance(model, ClusteringModel):
        state = {'backbone': model.backbone.state_dict(),
                 'backbone_dim':model.backbone_dim,
                 'cluster_head': [cluster_head.state_dict() for cluster_head in model.cluster_head],
                 'nheads': model.nheads}
    else:
        raise ValueError('Invalid model type {}'.format(type(model)))
    
    return state

def loadParameters(model, ckpt, load_backbone_only=False):
    model.backbone_dim = ckpt['backbone_dim']
    model.backbone.load_state_dict(ckpt['backbone'])
    if not load_backbone_only:
        if isinstance(model, ContrastiveModel):
            model.head = ckpt['head']
            model.contrastive_head.load_state_dict(ckpt['contrastive_head'])
        elif isinstance(model, ClusteringModel):
            model.nheads = ckpt['nheads']
            for i in range(model.nheads):
                model.cluster_head[i].load_state_dict(ckpt['cluster_head'][i])
        else:
            raise ValueError('Invalid model type {}'.format(type(model)))
    
    return model

