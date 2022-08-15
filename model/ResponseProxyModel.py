import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from torch import einsum

import numpy as np
import os

from .base_model.PointSwinTransformer import ResponsePointSwinTransformerProxyModel
from .base_model.PointTransformer import ResponsePointTransformerProxyModel
#==================================Point Swin==================================#
def getProxyModel(config, model_type='PointSwin'):
    if model_type == 'PointSwin':
        model = build_ResponsePointSwinTransformerProxyModel(config=config)
    elif model_type == 'PointTrans':
        model = build_ResponsePointTransformerProxyModel(config=config)
    else:
        raise ValueError('Invalid Model name {}.'.format(model_type))

    return model

#==================================Point Transformer==================================#

def build_ResponsePointTransformerProxyModel(config):
    model = ResponsePointTransformerProxyModel(in_channels=config['in_channel'],
                                               out_channels=config['out_channel'],
                                               param_dim=config['param_dim'],
                                               res_dim=config['res_dim'],
                                               embedding_dim=config['embedding_dim'],
                                               npoints=config['npoints'],
                                               nneighbor=config['nneighbor'],
                                               nblocks=config['nblocks'],
                                               transformer_dim=config['transformer_dim'])
    return model

def build_ResponsePointSwinTransformerProxyModel(config):
    model = ResponsePointSwinTransformerProxyModel(in_channels=config['in_channel'],
                                                   out_channels=config['out_channel'],
                                                   param_dim=config['param_dim'],
                                                   res_dim=config['res_dim'],
                                                   embed_dim=config['embed_dim'],
                                                   scale_factor=config['scale_factor'],
                                                   stage_num=config['stage_num'],
                                                   layers_num=config['layers_num'],
                                                   heads=config['heads'],
                                                   head_dims=config['head_dims'],
                                                   window_size=config['window_size'],
                                                   attn_layers=config['attn_layers'],
                                                   mlp_dim=config['mlp_dim'])
    return model



#==================================utils==================================#


def save_ResponseProxyModel(proxymodel,
                            save_path:str, epoch:int,
                            filename:str, optimizer=None):
    state = {
        'epoch':epoch,
        'embedding':proxymodel.embedding.state_dict(),
        'encoder':proxymodel.encoder.state_dict(),
        'decoder':proxymodel.decoder.state_dict(),
        'response_head':proxymodel.head.state_dict()
    }
    
    if optimizer is not None:
        state['optimizer']=optimizer.state_dict()
    
    filename = os.path.join(save_path, filename)
    
    torch.save(state, filename)

def load_ResponseProxyModel(net,
                            save_path:str, file_class:str, 
                            model_name:str='ResponseProxyModel', 
                            epoch:int=0):
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
    net.embedding.load_state_dict(ckpt['embedding'])
    net.encoder.load_state_dict(ckpt['encoder'])
    net.decoder.load_state_dict(ckpt['decoder'])
    net.head.load_state_dict(ckpt['response_head'])
    
    return net, epoch_start