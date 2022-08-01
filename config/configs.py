# coding=utf-8
from asyncio import tasks
import os

from pytz import BaseTzInfo
import torch
import numpy as np
from utils.toollib import check_dirs

#=================================Tools=================================#
def build_dirs(root):
    dirs_name = ['GT', 'Pred', 'Input', 'Loss', 'Log', 'Model_Library', 'contrast', 'indices']
    dirs = []
    for dir in dirs_name:
        dir_path = os.path.join(root, dir)
        check_dirs(dir_path)
        dirs.append(dir_path)
    dir_config ={}
    for i in range(len(dirs)):
        keyword = dirs_name[i]
        dir = dirs[i]
        dir_config[keyword]=dir
    
    path = dir_config['indices']
    keys =['train_index', 'val_index']
    addpath = ['train_indexes.npy', 'val_indexes.npy']
    for i in range(len(keys)):
        dir_config[keys[i]] = os.path.join(path, addpath[i])
        
    return dir_config

def get_config(task='ResponseProxy', model_type='PointSwin', opti='adamw', point2img=False, **kwargs):
    if task=='ResponseProxy':
        config = ResponseProxyConfig(ModelType=model_type, opti=opti)
        config.data_config['point2img'] = point2img
        config.data_config['task'] = task
    elif task in ['simclr', 'byol', 'simsiam', 'deepcluster', 'moco', 'scan', 'selflabel', 'spice', 'supervised']:
        config = ClusterConfig(task=task, BackboneType=model_type, opti=opti, point2img=point2img, **kwargs)
    else:
        raise ValueError('Invalid Task name {}.'.format(task))
    return config

#=================================Response Proxy Model Config=================================#

class ResponseProxyConfig:
    def __init__(self, ModelType='PointSwin', opti='adamw'):
        self.info_config = {'dataset_path':'./data',
                            'result_root':'./result/ResponseProxy{}'.format(ModelType),
                            'optimizer':opti}
        
        self.path_config = build_dirs(root=self.info_config['result_root'])
        
        if ModelType=='PointSwin':
            self.model_config = ResponseProxyPointSwinTransformerConfig
            self.train_config = PointSwinTransformer_ProxyTrainConfig
            self.data_config = PointSwinTransformer_ResponseDataConfig
        elif ModelType=='PointTrans':
            self.model_config = ResponseProxyPointTransformerConfig
            self.train_config = PointTransformer_ProxyTrainConfig
            self.data_config = PointTransformer_ResponseDataConfig
        else:
            raise ValueError('Invalid Model Type ({})'.format(ModelType))
        
        self.optimizer_config = getOptimizerConfig(opti=opti)
        self.optimizer_config['optimizer_kwargs']['lr']=self.train_config['lr']
        self.optimizer_config['optimizer_kwargs']['weight_decay']=self.train_config['weight_decay']
        self.model_name = 'Response{}'.format(ModelType)
        self.model_type = ModelType
        
        self.path_config['data_path'] = self.info_config['dataset_path']
        self.path_config['result_path'] = self.info_config['result_root']
        
        self.data_config['data_path'] = self.info_config['dataset_path']
        self.data_config['task'] = 'ResponseProxy'
        
        self.model_config['task'] = 'ResponseProxy'
        
### Point Swin Transformer
ResponseProxyPointSwinTransformerConfig = {
    'in_channel':3,
    'out_channel':15,
    'param_dim':8,
    'res_dim':2,
    'embed_dim':32,
    'npoints':1024,
    'scale_factor':4,
    'stage_num':3,
    'layers_num':1,
    'heads':8,
    'head_dims':32,
    'window_size':4,
    'attn_layers':4,
    'mlp_dim':None
}

PointSwinTransformer_ProxyTrainConfig = {
    'val_rate':0.05,
    'epochs':800,
    'lr':0.0001,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'weight_decay':5e-4,
    'momentum':0.8,
    'shuffle_data':True,
    'use_fp16':False,
    'save_per_epoch':100,
    'save_model_epoch':50,
    'show_iter':100,
    'train_loader':{'NumWorker':4,'BatchSize':4},
    'val_loader':{'NumWorker':4,'BatchSize':1}
}

PointSwinTransformer_ResponseDataConfig = {
    'point2img':False,
    'one_hot':True,
    'npoint':1024
}

### Point Transformer
ResponseProxyPointTransformerConfig = {
    'in_channel':3,
    'out_channel':15,
    'param_dim':8,
    'res_dim':2,
    'embedding_dim':32,
    'npoints':1024,
    'nblocks':4,
    'nneighbor':16,
    'transformer_dim':128
}

PointTransformer_ProxyTrainConfig = {
    'val_rate':0.05,
    'epochs':400,
    'batch_size':4,
    'lr':0.001,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'weight_decay':5e-4,
    'momentum':0.8,
    'shuffle_data':True,
    'use_fp16':False,
    'save_per_epoch':100,
    'save_model_epoch':50,
    'show_iter':100,
    'train_loader':{'NumWorker':4,'BatchSize':4},
    'val_loader':{'NumWorker':4,'BatchSize':1}
}

PointTransformer_ResponseDataConfig = {
    'point2img':False,
    'one_hot':True,
    'npoint':1024
}


#=================================Scan Config=================================#

BaseInfoConfig = {
    'dataset_path':'./data',
    'result_root':'./result/Clustering'
}

BaseTrainConfig = {
    'device':torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'epochs':200,
    'val_rate':0.2,
    'shuffle_data':True,
    'save_model_epoch':50,
    'train_loader':{'NumWorker':8,'BatchSize':8},
    'val_loader':{'NumWorker':8,'BatchSize':8},
}

BaseDataConfig = {
    'n_class':4,
    'one_hot':False, # simclr: False; supervised: True
    'weak_augmentation':True,
    'num_neighbors':16,
    'class_name':['圆环', '钻石', '混合', '其它'],
    'val_rate':0.2,
    'shuffle_data':True
}

BasePointSwinConfig = {
    'in_channels':15,
    'feature_dim':128,
    'hiddim':64,
    'heads':8,
    'headdim':32,
    'embeddim':4,
    'stage_num':3,
    'downscale':4,
    'layer_num':1,
    'window_size':4,
    'attnlayer':1
}

BasePointTransConfig = {
    'in_channels':15,
    'feature_dim':128,
    'embedding_dim':32,
    'npoints':1024,
    'nblocks':4,
    'nneighbor':32,
    'transformer_dim':128
}

class ClusterConfig:
    def __init__(self, task='simclr', BackboneType='PointSwin', 
                 DatsetName='tube', opti='adawm', scheduler='cosine', 
                 point2img=False, pretext:str='simclr', is_train:bool=True):
        self.task = task
        assert task in ['simclr', 'byol', 'simsiam', 'moco', 'scan', 'selflabel', 'spice', 'deepcluster', 'supervised']
        self.info_config = getInfoConfig(task=task, backbone=BackboneType, 
                                         datasetname=DatsetName)
        
        self.path_config = getPathConfig(task=task, root=self.info_config['result_root'], 
                                         db_name=self.info_config['train_db_name'], 
                                         BackboneType=BackboneType, pretext=pretext)
        self.train_config = getTrainConfig(task=task)
        self.data_config = getDataConfig(task=task, point2img=point2img, 
                                         data_path=self.info_config['dataset_path'])
        self.optimizer_config = getOptimizerConfig(opti=opti)
        self.scheduler_config = getSchedulerConfig(SchedulerType=scheduler)
        
        self.criterion_config = getCriterionConfig(task=task)
        
        if task in ['simclr', 'byol', 'simsiam', 'moco', 'scan', 'deepcluster', 'selflabel', 'spice']:
            self.model_config = getClusterModelConfig(ModelType=BackboneType, task=task)
            self.data_config['one_hot'] = False
        elif task == 'supervised':
            self.model_config = getSupervisedModelConfig(ModelType=BackboneType)
            self.model_config['nheads'] = 1
            self.data_config['one_hot'] = True
        else:
            raise ValueError('Invalid task {}'.format(task))
        
        
        if task == 'scan':
            self.data_config['paths'] = self.path_config

def getInfoConfig(task:str, backbone:str, datasetname:str):
    config = BaseInfoConfig
    config['task'] = task
    config['train_db_name'] = datasetname
    config['val_db_name'] = datasetname
    config['BackboneType'] = backbone
    
    return config

def getFeatureModelConfig(ModelType:str='PointSwin',feature_dim:int=128):
    if ModelType=='PointSwin':
        config = BasePointSwinConfig
    elif ModelType=='PointTrans':
        config = BasePointTransConfig
    else:
        raise ValueError('Invalid Model name {}.'.format(ModelType))
    config['feature_dim']=feature_dim
    
    return config

def getClusterModelConfig(ModelType:str='PointSwin', task:str='simclr', 
                          num_cluster:int=4, head:str='mlp', feature_dim:int=128, 
                          contrastive_feadim:int=128, nheads:int=1):
    config = {}
    backbone_config = getFeatureModelConfig(ModelType=ModelType, feature_dim=feature_dim)
    
    config['BackboneConfig'] = backbone_config
    config['BackboneType'] = ModelType
    config['num_cluster']=num_cluster
    config['task']=task
    config['head']=head
    config['nheads']=nheads
    config['contrastive_feadim']=contrastive_feadim
    
    return config

def getSupervisedModelConfig(ModelType:str='PointSwin',num_class:int=4, feature_dim:int=128):
    config = {}
    backbone_config = getFeatureModelConfig(ModelType=ModelType, feature_dim=feature_dim)
    backbone_config['num_class']=num_class
    config['BackboneConfig'] = backbone_config
    config['BackboneType'] = ModelType
    config['task'] = 'supervised'
    
    return config

def getPathConfig(task:str, root:str, db_name:str, BackboneType:str='PointSwin', pretext:str='simclr'):
    path = os.path.join(root, db_name, BackboneType)
    check_dirs(path)
    if task in ['simclr', 'byol', 'simsiam', 'moco']:
        config = getPretextConfig(path=path, pretext=task)
    elif task in ['scan', 'selflabel', 'deepcluster', 'spice']:
        config = getModelCheckpoint(path=path, task=task)
        pretext_config = getPretextConfig(path=path, pretext=pretext)
        for key in pretext_config.keys():
            config[key] = pretext_config[key]
    elif task == 'supervised':
        config = getModelCheckpoint(path=path, task=task)
        keys =['train_index', 'val_index']
        addpath = ['train_indexes.npy', 'val_indexes.npy']
        task_dir = os.path.join(path, task)
        for i in range(len(keys)):
            config[keys[i]] = os.path.join(task_dir, addpath[i])
    else:
        raise ValueError('Invalid task {}'.format(task))
        
    config['base_dir'] = path
    return config

def getTrainConfig(task):
    config = BaseTrainConfig
    config['task'] = task
    if task == 'scan':
        config['update_cluster_head_only'] = False
    else:
        config['update_cluster_head_only'] = False
    return config

def getDataConfig(task:str, data_path:str, point2img:bool=False,
                  val_rate:float=0.2, num_neighbors:int=16,
                  shuffle_data:bool=True, 
                  class_name:list=['圆环', '钻石', '混合', '其它']):
    config = BaseDataConfig
    config['val_rate'] = val_rate
    config['shuffle_data'] = shuffle_data
    config['point2img'] = point2img
    config['class_name'] = class_name
    config['num_neighbors'] = num_neighbors
    config['task'] = task
    config['data_path'] = data_path
    
    return config
 
def getModelCheckpoint(path, task):
    config = {}
    keywords = ['_dir', '_checkpoint', '_log', '_confusion_matrix']
    addpath = ['', 'checkpoint', 'log', 'confusion_matrix']
    for i in range(len(keywords)):
        k = '{}{}'.format(task, keywords[i])
        config[k]=os.path.join(path, task, addpath[i])
        check_dirs(config[k])
    return config

def getPretextConfig(path, pretext:str='simclr'):
    path = os.path.join(path, 'pretext', pretext)
    config = getModelCheckpoint(path=path, task=pretext)
    #pretext_dir = os.path.join(path, 'pretext')
    keys =['topk_neighbors_train_path', 'topk_neighbors_val_path', 'train_index', 'val_index']
    addpath = ['topk-train-neighbors.npy', 'topk-val-neighbors.npy', 'train_indexes.npy', 'val_indexes.npy']
    for i in range(len(keys)):
        config[keys[i]] = os.path.join(path, addpath[i])
    return config

def getOptimizerConfig(opti):
    config={}
    config['optimizer']=opti
    config['optimizer_kwargs'] = optimizer_param(opti)
    return config

def optimizer_param(optype):
    if optype == 'sgd':
        params = sgd_config
    elif optype == 'adam':
        params = adam_config
    elif optype == 'adamw':
        params = adamw_config
    else:
        raise ValueError('Invalid optimizer type {}'.format(optype))
    
    return params
        
def getSchedulerConfig(SchedulerType):
    if SchedulerType=='cosine':
        config = CosineSchedulerConfig
    else:
        raise ValueError('Invalid Scheduler name {}.'.format(SchedulerType))
    return config

def getCriterionConfig(task):
    config = criterion_config
    config['criterion']=task
    return config

sgd_config={
    'lr':0.4,
    'nesterov':False,
    'weight_decay':0.0001,
    'momentum':0.9
}

adam_config = {
    'lr':0.0001,
    'weight_decay':0.0001
}

adamw_config = {
    'lr':0.0001,
    'weight_decay':0.0001,
    'betas':(0.9, 0.999),
    'eps':1e-08
}

CosineSchedulerConfig = {
    'scheduler':'cosine',
    'update_cluster_head_only':False,
    'lr_decay_rate':0.1,
    'lr_decay_epochs':20
}

criterion_config = {
    # simclr
    'temperature':0.1,
    # scan
    'entropy_weight':2.0,
    # selflabel, spice
    'apply_class_balancing':True,
    'confidence_threshold':0.4
}

'''
# parameters of response proxy task

sgd_config={
    'lr':0.4,
    'nesterov':False,
    'weight_decay':0.0001,
    'momentum':0.9
}

adam_config = {
    'lr':0.0001,
    'weight_decay':0.0001
}

adamw_config = {
    'lr':0.0001,
    'weight_decay':0.001,
    'betas':(0.9, 0.999),
    'eps':1e-08
}

CosineSchedulerConfig = {
    'scheduler':'cosine',
    'update_cluster_head_only':False,
    'lr_decay_rate':0.01,
    'lr_decay_epochs':20
}

criterion_config = {
    # simclr
    'temperature':0.1,
    # scan
    'entropy_weight':5.0,
    # selflabel, spice
    'apply_class_balancing':True,
    'confidence_threshold':0.4
}

'''
# class ScanConfig:
#     def __init__(self, task='simclr', backbone_type='swin_t'):
#         self.data_path = './data'
#         self.result_path = './result/SCAN'
#         self.dataset_name = 'tube'
#         self.backbone_type = backbone_type
#         self.result_path = os.path.join(self.result_path, backbone_type)
#         assert task in ['simclr', 'scan', 'selflabel', 'spice', 'supervised']
#         self.task = task # simclr, scan, selflabel
        
#         self.config = self.get_config()
        
    
#     def get_config(self):
#         cfg={}
#         cfg['setup'] = self.task
#         cfg['dataset_path'] = self.data_path
#         cfg['root_dir'] = self.result_path
#         cfg['train_db_name'] = self.dataset_name
#         cfg['val_db_name'] = self.dataset_name
        
#         base_dir = os.path.join(cfg['root_dir'], cfg['train_db_name'])
#         pretext_dir = os.path.join(base_dir, 'pretext')
#         make_path(base_dir)
#         make_path(pretext_dir)
        
#         cfg['pretext_dir'] = pretext_dir
#         cfg['pretext_checkpoint'] = os.path.join(pretext_dir, 'checkpoint.pth.tar')
#         cfg['pretext_model'] = os.path.join(pretext_dir, 'model.pth.tar')
        
#         cfg['topk_neighbors_train_path'] = os.path.join(pretext_dir, 'topk-train-neighbors.npy')
#         cfg['topk_neighbors_val_path'] = os.path.join(pretext_dir, 'topk-val-neighbors.npy')
#         cfg['train_index'] = os.path.join(pretext_dir, 'train_indexes.npy')
#         cfg['val_index'] = os.path.join(pretext_dir, 'val_indexes.npy')
        
#         if cfg['setup'] in ['scan', 'selflabel', 'spice', 'supervised']:
#             base_dir = os.path.join(cfg['root_dir'], cfg['train_db_name'])
#             scan_dir = os.path.join(base_dir, 'scan')
#             selflabel_dir = os.path.join(base_dir, 'selflabel') 
#             spice_dir = os.path.join(base_dir, 'spice')
#             supervised_dir = os.path.join(base_dir, 'supervised')
#             make_path(base_dir)
#             make_path(scan_dir)
#             make_path(selflabel_dir)
#             make_path(spice_dir)
#             make_path(supervised_dir)
            
            
#             cfg['scan_dir'] = scan_dir
#             cfg['scan_checkpoint'] = os.path.join(scan_dir, 'checkpoint.pth.tar')
#             cfg['scan_model'] = os.path.join(scan_dir, 'model.pth.tar')
            
#             cfg['selflabel_dir'] = selflabel_dir
#             cfg['selflabel_checkpoint'] = os.path.join(selflabel_dir, 'checkpoint.pth.tar')
#             cfg['selflabel_model'] = os.path.join(selflabel_dir, 'model.pth.tar')
            
#             cfg['spice_dir'] = spice_dir
#             cfg['spice_checkpoint'] = os.path.join(spice_dir, 'checkpoint.pth.tar')
#             cfg['spice_model'] = os.path.join(spice_dir, 'model.pth.tar')
            
#             cfg['supervised_dir'] = supervised_dir
#             cfg['supervised_checkpoint'] = os.path.join(supervised_dir, 'checkpoint.pth.tar')
#             cfg['supervised_model'] = os.path.join(supervised_dir, 'model.pth.tar')
            
#         # data
#         cfg['point2img'] = True
#         if self.backbone_type == 'PointTransformer':
#             cfg['point2img']=False
#         cfg['weak_aug'] = True
#         cfg['num_neighbors'] = 15
#         cfg['shuffle_data'] = True
#         cfg['val_rate'] = 0.2
#         cfg['num_classes'] = 4
#         #cfg['class_name'] = ['Circular', 'Diamond', 'Mixed', 'Other']
#         cfg['class_name'] = ['圆环', '钻石', '混合', '其它']
        
#         # model
#         cfg['time_step'] = 4
#         cfg['backbone_type'] = self.backbone_type
#         cfg['feature_dim'] = 256
#         cfg['in_channels'] = 12
#         cfg['cluster_num'] = 4
#         cfg['head'] = 'mlp'
#         cfg['nheads'] = 1
#         cfg['contrastive_feadim']=128
#         cfg['num_heads']=1
        
        
#         # train
#         cfg['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         cfg['batch_size'] = 32
#         cfg['num_workers'] = 8
#         cfg['epochs'] = 200
        
#         # criterion
#         cfg['criterion'] = self.task
#         cfg['temperature'] = 0.1
#         cfg['entropy_weight'] = 5.0
#         cfg['apply_class_balancing'] = True
#         cfg['confidence_threshold'] = 0.4
        
#         # optimizer
#         cfg['optimizer'] = 'adamw'
#         cfg['optimizer_kwargs'] = optimizer_param(cfg['optimizer'])
        
#         # scheduler
#         cfg['scheduler'] = 'cosine'
#         cfg['update_cluster_head_only'] = False
#         cfg['lr_decay_rate'] =0.1
#         cfg['lr_decay_epochs'] = 20
        
#         return cfg


