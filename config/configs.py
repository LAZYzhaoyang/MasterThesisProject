# coding=utf-8
import os

from ..utils.toollib import check_dirs
from .base_config import *

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

def get_config(task='ResponseProxy', ncluster:int=4, 
               model_type='PointSwin', opti='adamw', 
               point2img=False, data_root=None,
               result_root=None, **kwargs):
    if task=='ResponseProxy':
        config = ResponseProxyConfig(ModelType=model_type, opti=opti, 
                                     data_root=data_root, result_root=result_root)
        config.data_config['point2img'] = point2img
        config.data_config['task'] = task
    elif task in ['simclr', 'byol', 'simsiam', 'deepcluster', 'scan', 'spice', 'supervised']:
        config = ClusterConfig(task=task, ncluster=ncluster, 
                               BackboneType=model_type, opti=opti, 
                               point2img=point2img, data_root=data_root, 
                               result_root=result_root, **kwargs)
    else:
        raise ValueError('Invalid Task name {}.'.format(task))
    return config

#=================================Response Proxy Model Config=================================#

class ResponseProxyConfig:
    def __init__(self, ModelType='PointSwin', opti='adamw', data_root=None, result_root=None):
        self.info_config = {'dataset_path':'./data',
                            'result_root':'./result/ResponseProxy{}'.format(ModelType),
                            'optimizer':opti}
        if result_root is not None:
            self.info_config['result_root'] = result_root
        if data_root is not None:
            self.info_config['dataset_path'] = data_root
            
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
            
    
#=================================Clustering Config=================================#

class ClusterConfig:
    def __init__(self, ncluster:int=4, task='simclr', BackboneType='PointSwin', 
                 DatsetName='tube', opti='adawm', scheduler='cosine', feature_dim=128,
                 point2img=False, pretext:str='simclr', data_root=None, result_root=None, 
                 is_train:bool=True):
        self.task = task
        assert task in ['simclr', 'byol', 'simsiam', 'scan', 'spice', 'deepcluster', 'supervised']
        self.info_config = getInfoConfig(task=task, backbone=BackboneType, 
                                         datasetname=DatsetName)
        self.model_type = BackboneType
        self.pretext = pretext
        self.point2img = point2img
        
        if result_root is not None:
            self.info_config['result_root']=result_root
        if data_root is not None:
            self.info_config['dataset_path']=data_root
        
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
            self.model_config = getClusterModelConfig(ModelType=BackboneType, task=task, 
                                                      num_cluster=ncluster, feature_dim=feature_dim)
            self.data_config['one_hot'] = False
            if task in ['selflabel', 'spice']:
                self.model_config['nheads']=1
        elif task == 'supervised':
            self.model_config = getSupervisedModelConfig(ModelType=BackboneType, feature_dim=feature_dim)
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
                          nheads:int=1):
    config = {}
    backbone_config = getFeatureModelConfig(ModelType=ModelType, feature_dim=feature_dim)
    
    config['BackboneConfig'] = backbone_config
    config['BackboneType'] = ModelType
    config['num_cluster']=num_cluster
    config['task']=task
    config['head']=head
    config['nheads']=nheads
    config['contrastive_feadim']=feature_dim
    
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
        config = getModelCheckpoint(path=path, task=task, pretext=pretext)
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
 
def getModelCheckpoint(path, task, pretext=None):
    config = {}
    keywords = ['_dir', '_checkpoint', '_log', '_confusion_matrix']
    addpath = ['', 'checkpoint', 'log', 'confusion_matrix']
    for i in range(len(keywords)):
        k = '{}{}'.format(task, keywords[i])
        if pretext is not None:
            config[k]=os.path.join(path, task, pretext, addpath[i])
        else:
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

#=================================Tube Optimizing Config=================================#

class TubeOptimizingConfig(object):
    def __init__(self, proxy_backbone:str='PointSwin', 
                 cluster_backbone:str='PointSwin', 
                 cluster_type:str='spice',
                 pretext:str='simclr',
                 ncluster:int=4,
                 pretrain_path:str=None,
                 point2img:bool=False):
        assert cluster_type in ['spice', 'scan', 'deepkmeans', 'supervised', 'deepcluster']
        assert pretext in ['simclr', 'byol', 'simsiam', 'supervised']
        assert proxy_backbone in ['PointSwin', 'PointTrans']
        assert cluster_backbone in ['PointSwin', 'PointTrans']
        self.info = {
            'cluster_type':cluster_type,
            'cluster_backbone':cluster_backbone,
            'proxy_backbone':proxy_backbone,
            'ncluster':ncluster,
            'pretext':pretext
        }
        
        self.ProxyConfig = get_config(task='ResponseProxy', model_type=proxy_backbone, point2img=point2img)
        if cluster_type == 'deepkmeans':
            self.ClustserModelConfig = {'ncluster':ncluster,
                                        'backbone':cluster_backbone,
                                        'pretext':pretext,
                                        'pretrain_path':pretrain_path,
                                        'point2img':point2img,}
        else:
            self.ClustserModelConfig = get_config(task=cluster_type, model_type=cluster_backbone,
                                                  point2img=point2img, ncluster=ncluster, pretext=pretext)
        self.TubeParams = TubeParamsConfig
        self.optimizingset = TubeOptimizingSet
        
    
