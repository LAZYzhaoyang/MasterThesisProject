import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from model.PointSwinTransformer import PointSwinFeatureExtractor
from model.PointTransformer import PointTransformerBackbone

def resnet(in_channels=15, feature_dim=512, model_type='resnet18', **kwargs):
    #assert model_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    if model_type == 'resnet18':
        backbone = torchvision.models.resnet18(**kwargs)
    elif model_type == 'resnet34':
        backbone = torchvision.models.resnet34(**kwargs)
    elif model_type == 'resnet50':
        backbone = torchvision.models.resnet50(**kwargs)
    elif model_type == 'resnet101':
        backbone = torchvision.models.resnet101(**kwargs)
    elif model_type == 'resnet152':
        backbone = torchvision.models.resnet152(**kwargs)
    else:
        raise ValueError('Invalid model type {}'.format(model_type))
        
    old_conv1 = backbone.conv1
    new_conv1 = nn.Conv2d(
        in_channels=in_channels, 
        out_channels=old_conv1.out_channels,
        kernel_size=old_conv1.kernel_size,
        stride=old_conv1.stride,
        padding=old_conv1.padding,
        bias=True if old_conv1.bias else False)
    backbone.conv1 = new_conv1
    num_fc_ftr = backbone.fc.in_features
    backbone.fc = nn.Linear(num_fc_ftr, feature_dim)
    #return {'backbone': backbone, 'dim': feature_dim}
    return backbone


def getBackbone(config, ModelType:str='PointSwin'):
    if ModelType=='PointSwin':
        model = buildPointSwinBackbone(config=config)
    elif ModelType == 'PointTrans':
        model = buildPointTransBackbone(config=config)
    else:
        raise ValueError('Invalid Model name {}.'.format(ModelType))
    
    feature_dim = config['feature_dim']
    
    return {'backbone': model, 'dim': feature_dim}

def buildPointSwinBackbone(config):
    model = PointSwinFeatureExtractor(in_channels=config['in_channels'],
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

def buildPointTransBackbone(config):
    model = PointTransformerBackbone(in_channels=config['in_channels'],
                                     feature_dim=config['feature_dim'],
                                     embedding_dim=config['embedding_dim'],
                                     npoints=config['npoints'],
                                     nblocks=config['nblocks'],
                                     nneighbor=config['nneighbor'],
                                     transformer_dim=config['transformer_dim'])
    return model

