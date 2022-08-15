"""
# author: Zhaoyang Li
# 2022 05 01
# Central South University
"""
import torch

#=================================Response Proxy Model Config=================================#
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
    'train_loader':{'NumWorker':8,'BatchSize':16},
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
    'epochs':800,
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
    'train_loader':{'NumWorker':8,'BatchSize':16},
    'val_loader':{'NumWorker':4,'BatchSize':1}
}

PointTransformer_ResponseDataConfig = {
    'point2img':False,
    'one_hot':True,
    'npoint':1024
}

#=================================Clustering Config=================================#
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
    'train_loader':{'NumWorker':8,'BatchSize':128},
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
#=================================Tube Optimizing Config=================================#

TubeParamsConfig={
    'height':[90,185],
    'radius':[17.5,27.5],
    'thick':[1,4],
    'etan':[1000, 1450],
    'sigy':[200,500],
    'rho':[2.7e-9, 7.85e-9],
    'e':[69000, 206000],
    'ea':[1944700, 7251800],
    'mass':[2.7132919e-05, 0.00099261466],
    'pcf':[66656.05000159159, 8046749.491833895]
}

TubeOptimizingSet={
    'keys':['height', 'radius', 'thick', 'etan', 'sigy', 'rho', 'e', 'ea', 'mass', 'pcf'],
    'discrete':[0,0,0,1,1,1,1,0,0,0],
    'no_buttom':[1,1,1,0,0,0,0,1,1,1],
    'opti_keys':['height', 'radius', 'thick', 'etan', 'sigy', 'rho', 'e', 'material'],
    'pop_size':100,
}









