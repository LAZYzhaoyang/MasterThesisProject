from sklearn.utils import shuffle
from data_loader.base_dataset import Response_DataLoader, PointCloudDataset
from data_loader.subdataset import AugmentedDataset, NeighborsDataset, \
    SpiceDataset, SupervisedDataset

import numpy as np
import os
import collections

import torch
from torch.utils.data import SubsetRandomSampler
from torch._six import string_classes, int_classes

def getDataset(config):
    task = config['task']
    if task == 'ResponseProxy':
        dataset = Response_DataLoader(data_path=config['data_path'], 
                                      point2img=config['point2img'],
                                      npoint=config['npoint'])
    elif task in ['simclr', 'byol', 'simsiam', 'moco', 'scan', 'deepcluster', 'selflabel', 'spice','supervised']:
        BaseDataset = PointCloudDataset(data_path=config['data_path'],
                                        n_class=config['n_class'],
                                        one_hot=config['one_hot'])
        if task == 'supervised':
            dataset = SupervisedDataset(dataset=BaseDataset, point2img=config['point2img'])
        elif task == 'spice':
            dataset = SpiceDataset(dataset=BaseDataset, point2img=config['point2img'])
        elif task in ['simclr', 'byol', 'simsiam', 'deepcluster', 'moco', 'selflabel']:
            dataset = AugmentedDataset(dataset=BaseDataset, point2img=config['point2img'])
        elif task == 'scan':
            if os.path.exists(config['paths']['train_index']) and os.path.exists(config['paths']['val_index']):
                train_indices = np.load(config['paths']['train_index']).tolist()
                val_indices = np.load(config['paths']['val_index']).tolist()
                train_neibor_indices = np.load(config['paths']['topk_neighbors_train_path'])
                val_neibor_indices = np.load(config['paths']['topk_neighbors_val_path'])
                
                train_base_dataset = PointCloudDataset(data_path=config['data_path'],
                                                       n_class=config['n_class'],
                                                       one_hot=config['one_hot'],
                                                       indices=train_indices)
                val_base_dataset = PointCloudDataset(data_path=config['data_path'],
                                                     n_class=config['n_class'],
                                                     one_hot=config['one_hot'],
                                                     indices=val_indices)
                
                train_dataset = NeighborsDataset(dataset=train_base_dataset, 
                                                 indices=train_neibor_indices,
                                                 num_neighbors=config['num_neighbors'],
                                                 point2img=config['point2img'])
                val_dataset = NeighborsDataset(dataset=val_base_dataset, 
                                               indices=val_neibor_indices,
                                               num_neighbors=config['num_neighbors'],
                                               point2img=config['point2img'])
                dataset = {'train_dataset':train_dataset, 
                           'val_dataset':val_dataset,
                           'train_num':len(train_indices),
                           'val_num':len(val_indices),
                           'train_indices':train_indices,
                           'val_indices':val_indices}
            else:
                raise ValueError('You need a pre-trained model with train indices and train neibor indices')
    else:
        raise ValueError('Invalid task {}'.format(task))
    
    return dataset


def splitDataset(dataset, cfg, use_pretrain_indexes=False, indices_save_path=None):
    pathcfg = cfg.path_config
    traincfg = cfg.train_config
    
    val_rate=traincfg['val_rate']
    shuffle_data=traincfg['shuffle_data']
    
    if use_pretrain_indexes:
        if indices_save_path is not None:
            train_file = os.path.join(indices_save_path, 'train_indices.npy')
            val_file = os.path.join(indices_save_path, 'val_indices.npy')
        else:
            train_file = pathcfg['train_index']
            val_file = pathcfg['val_index']
        train_indices = np.load(train_file)
        val_indices = np.load(val_file)
        train_indices= train_indices.tolist()
        val_indices = val_indices.tolist()
    else:
        dataset_size = len(dataset)
        split = int(np.floor(val_rate * dataset_size))
        indices = list(range(dataset_size))
        if shuffle_data:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        #train_indices, val_indices = indices, indices[:split]
    
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    
    train_dataloader = getDataloader(cfg=traincfg['train_loader'], dataset=dataset, sampler=train_sampler)
    val_dataloader = getDataloader(cfg=traincfg['val_loader'], dataset=dataset, sampler=valid_sampler)
    
    out = {'train_dataloader':train_dataloader,
           'val_dataloader':val_dataloader,
           'train_num':len(train_indices),
           'val_num':len(val_indices),
           'train_indices':train_indices,
           'val_indices':val_indices}
    
    if not use_pretrain_indexes:
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        if indices_save_path is not None:
            np.save(os.path.join(indices_save_path, 'train_indices.npy'), train_indices)
            np.save(os.path.join(indices_save_path, 'val_indices.npy'), val_indices)
        else:
            np.save(pathcfg['train_index'], train_indices)
            np.save(pathcfg['val_index'], val_indices)
    
    return out


def getDataloader(cfg, dataset, sampler=None):
    '''return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'], 
            batch_size=p['batch_size'], sampler=sampler, collate_fn=collate_custom,
            shuffle=True)'''
    if sampler is not None:
        return torch.utils.data.DataLoader(dataset, 
                                           num_workers=cfg['NumWorker'], 
                                           batch_size=cfg['BatchSize'], 
                                           sampler=sampler, 
                                           collate_fn=collate_custom)
    else:
        return torch.utils.data.DataLoader(dataset, 
                                           num_workers=cfg['NumWorker'], 
                                           batch_size=cfg['BatchSize'], 
                                           collate_fn=collate_custom, 
                                           shuffle=True)
 
""" Custom collate function """
def collate_custom(batch):
    if isinstance(batch[0], np.int64):
        return np.stack(batch, 0)

    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)

    elif isinstance(batch[0], np.ndarray):
        return np.stack(batch, 0)

    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)

    elif isinstance(batch[0], float):
        return torch.FloatTensor(batch)

    elif isinstance(batch[0], string_classes):
        return batch

    elif isinstance(batch[0], collections.Mapping):
        batch_modified = {key: collate_custom([d[key] for d in batch]) for key in batch[0] if key.find('idx') < 0}
        return batch_modified

    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [collate_custom(samples) for samples in transposed]

    raise TypeError(('Type is {}'.format(type(batch[0]))))