"""
# author: Zhaoyang Li
# 2022 08 15
# Central South University
"""

from .base_model.PointSwinTransformer import PointSwin
from .base_model.PointTransformer import PointTransformerCls
import torch
import os

def getSupervisedModel(config, ModelType:str='PointSwin'):
    if ModelType=='PointSwin':
        model = buildPointSwin(config=config)
    elif ModelType == 'PointTrans':
        model = buildPointTrans(config=config)
    else:
        raise ValueError('Invalid Model name {}.'.format(ModelType))
    
    return model

def buildPointSwin(config):
    model = PointSwin(in_channels=config['in_channels'],
                      num_class=config['num_class'],
                      feature_dim=config['feature_dim'],
                      hiddim=config['hiddim'],
                      heads=config['heads'],
                      headdim=config['headdim'],
                      embeddim=config['embeddim'],
                      stage_num=config['stage_num'],
                      downscale=config['downscale'],
                      layer_num=config['layer_num'],
                      window_size=config['window_size'],
                      attnlayer=config['attnlayer'])
    return model

def buildPointTrans(config):
    model = PointTransformerCls(in_channels=config['in_channels'],
                                num_class=config['num_class'],
                                feature_dim=config['feature_dim'],
                                embedding_dim=config['embedding_dim'],
                                npoints=config['npoints'],
                                nblocks=config['nblocks'],
                                nneighbor=config['nneighbor'],
                                transformer_dim=config['transformer_dim'])
    return model




def saveSupervisedModel(supervisednet, save_path:str, epoch:int, filename:str, optimizer=None):
    state = getSupervisedState(supervisednet)
    state['epoch'] = epoch
    if optimizer is not None:
        state['optimizer']=optimizer.state_dict()
    else:
        state['optimizer']=None
    
    filename = os.path.join(save_path, filename)
    
    torch.save(state, filename)

def loadSupervisedModel(net, save_path:str, file_class:str, model_name:str='PointSwin', epoch:int=0, 
                        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    class_list=['best', 'last', 'epochs']
    assert file_class in class_list, 'file class must be one of [best, last, epoch]'
    if file_class==class_list[0]:
        filename = os.path.join(save_path, '{}_best.pth'.format(model_name))
    elif file_class==class_list[1]:
        filename = os.path.join(save_path, '{}_last_epoch.pth'.format(model_name))
    else:
        filename = os.path.join(save_path, '{}_checkpoint-epoch{}.pth'.format(model_name, epoch))
        
    ckpt = torch.load(filename, map_location=device)
    
    epoch_start = ckpt['epoch']
    net = loadSupervisedParameters(model=net, ckpt=ckpt)
    
    return net, epoch_start

def getSupervisedState(model):
    if isinstance(model, PointSwin):
        state = {'backbone': model.backbone.state_dict(),
                 'backbone_dim': model.feature_dim,
                 'fc': model.clshead.state_dict()}
    elif isinstance(model, PointTransformerCls):
        state = {'backbone': model.backbone.state_dict(),
                 'backbone_dim':model.feature_dim,
                 'fc': model.clshead.state_dict()}
    else:
        raise ValueError('Invalid model type {}'.format(type(model)))
    
    return state

def loadSupervisedParameters(model, ckpt):
    if isinstance(model, PointSwin):
        model.feature_dim = ckpt['backbone_dim']
        model.backbone.load_state_dict(ckpt['backbone'])
        model.clshead.load_state_dict(ckpt['fc'])
    elif isinstance(model, PointTransformerCls):
        model.feature_dim = ckpt['backbone_dim']
        model.backbone.load_state_dict(ckpt['backbone'])
        model.clshead.load_state_dict(ckpt['fc'])
    else:
        raise ValueError('Invalid model type {}'.format(type(model)))
    
    return model