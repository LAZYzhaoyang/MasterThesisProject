"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from .data_aug import get_transformers, get_supervised_transform, getAugmentedTransform, getNeighborTransform

from utils.toollib import squeeze_node, listnode2img

""" 
    AugmentedDataset
    Returns an image together with an augmentation.
"""


class AugmentedDataset(Dataset):
    def __init__(self, dataset, point2img=False, sq_ch=False, img_h=64):
        super(AugmentedDataset, self).__init__()
        transforms = getAugmentedTransform()
        dataset.transform = None
        self.dataset = dataset
        self.point2img = point2img
        self.img_h = img_h
        self.sq_ch = sq_ch
        
        self.node_transform = transforms['standard']
        self.augmentation_transform = transforms['augment']


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset.__getitem__(index)
        ori_node = sample['node']
        
        node = self.node_transform(ori_node)
        aug_node = self.augmentation_transform(ori_node)
        
        if self.point2img:
            node = listnode2img(node, self.img_h)
            aug_node = listnode2img(aug_node, self.img_h)
        
        if self.sq_ch:
            node = squeeze_node(node)
            aug_node = squeeze_node(aug_node)
        
            
        sample['node'] = node
        sample['node_augmented'] = aug_node
        
        sample['meta']['node_size'] = node.shape
        
        return sample
    
    


""" 
    NeighborsDataset
    Returns an image with one of its neighbors.
"""
class NeighborsDataset(Dataset):
    def __init__(self, dataset, indices, num_neighbors=None, sq_ch=False, point2img=False, img_h=64):
        super(NeighborsDataset, self).__init__()
        transforms = getNeighborTransform()
        dataset.transform = None
        self.anchor_transform = transforms['standard']
        self.neighbor_transform = transforms['augment']
       
        self.point2img = point2img
        self.dataset = dataset
        self.img_h = img_h
        self.sq_ch = sq_ch
        self.indices = indices # Nearest neighbor indices (np.array  [len(dataset) x k])
        if num_neighbors is not None:
            self.indices = self.indices[:, :num_neighbors+1]
        print(self.indices.shape[0],len(self.dataset))
        assert(self.indices.shape[0] == len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)
        
        neighbor_index = np.random.choice(self.indices[index], 1)[0]
        neighbor = self.dataset.__getitem__(neighbor_index)

        anchor['node'] = self.anchor_transform(anchor['node'])
        neighbor['node'] = self.neighbor_transform(neighbor['node'])
        
        if self.point2img:
            anchor['node'] = listnode2img(anchor['node'], self.img_h)
            neighbor['node'] = listnode2img(neighbor['node'], self.img_h)
            
        if self.sq_ch:
            anchor['node'] = squeeze_node(anchor['node'])
            neighbor['node'] = squeeze_node(neighbor['node'])

        output['anchor'] = anchor['node']
        output['neighbor'] = neighbor['node'] 
        output['possible_neighbors'] = torch.from_numpy(self.indices[index])
        output['label'] = anchor['label']
        
        return output
    



class SpiceDataset(Dataset):
    def __init__(self, dataset, point2img=False, img_h=64, sq_ch=False):
        super(SpiceDataset, self).__init__()
        transforms = get_transformers()
        dataset.transform = None
        self.anchor_transform = transforms['standard']
        self.weak_transform = transforms['weak_augment']
        self.strong_transform = transforms['strong_augment']
        
        self.point2img = point2img
        self.dataset = dataset
        self.img_h = img_h
        self.sq_ch = sq_ch

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)
        
        
        weak_trans = self.weak_transform(anchor['node'])
        strong_trans = self.strong_transform(anchor['node'])
        anchor['node'] = self.anchor_transform(anchor['node'])

        if self.point2img:
            anchor['node'] = listnode2img(anchor['node'], self.img_h)
            weak_trans = listnode2img(weak_trans, self.img_h)
            strong_trans = listnode2img(strong_trans, self.img_h)
            
        if self.sq_ch:
            anchor['node'] = squeeze_node(anchor['node'])
            weak_trans = squeeze_node(weak_trans)
            strong_trans = squeeze_node(strong_trans)

        output['node'] = anchor['node']
        output['weak_trans'] = weak_trans
        output['strong_trans'] = strong_trans
        output['label'] = anchor['label']
        #output['meta'] = anchor['meta']
        
        return output
    


class SupervisedDataset(Dataset):
    def __init__(self, dataset, point2img=False, sq_ch=False, img_h=64):
        super(SupervisedDataset, self).__init__()
        dataset.transform = None
        self.transform = get_supervised_transform()
        self.point2img = point2img
        self.dataset = dataset
        self.sq_ch = sq_ch
        self.img_h = img_h
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)
        anchor['node']= self.transform(anchor['node'])
        
        if self.point2img:
            anchor['node'] = listnode2img(anchor['node'], self.img_h)
            
        if self.sq_ch:
            anchor['node'] = squeeze_node(anchor['node'])
        
        output['node'] = anchor['node']
        output['label'] = anchor['label']
        
        return output
    
class MoCoDataset(Dataset):
    def __init__(self):
        super().__init__()
    
    
    
class SPICEDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()