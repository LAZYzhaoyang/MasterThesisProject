"""
# author: Zhaoyang Li
# 2021 12 15
# Central South University
"""

#import tensorflow as tf
from pickle import NONE
import numpy as np
import math
import random
import os

import torch
from torch.utils.data import Dataset
import albumentations as A

from .data_aug import get_transformers
from utils.toollib import squeeze_node, unsqueeze_node, num2onehot, random_index

class Response_DataLoader(Dataset):
    def __init__(self, data_path, point2img=True, npoint=4096):
        super(Response_DataLoader, self).__init__()
        self.dir_list = os.listdir(data_path)
        self.data_path = data_path
        self.get_max_min()
        self.point2img=point2img
        self.npoint = npoint
        
    def __getitem__(self, index):
        dir_name = self.dir_list[index]
        dir_path = os.path.join(self.data_path, dir_name)
        init_coordinate, node_coordinate, param, res = self.read_data(dir_path)
        out={'init_node':init_coordinate, 'out_node':node_coordinate, 'params':param, 'res':res}
        
        return out
        
    def read_data(self, path):
        param_file = os.path.join(path, 'parameter.npy')
        node_coord_file = os.path.join(path, 'node_coordinate.npy')
        init_coord_file = os.path.join(path, 'init_node_coordinate.npy')
        response_file = os.path.join(path, 'response.npy')
        
        params = np.load(param_file, allow_pickle=True).item()
        response = np.load(response_file, allow_pickle=True).item()
        node_coordinate = np.load(node_coord_file)
        init_coordinate = np.load(init_coord_file)
        
        init_coordinate, node_coordinate, param, res = \
            self.data_preprocessing(init_coordinate=init_coordinate,
                                    node_coordinate=node_coordinate,
                                    params=params,response=response)   
        return init_coordinate, node_coordinate, param, res
        

    def get_max_min(self):
        min_mass = 100000000000
        min_pcf = 100000000000
        min_ea = 100000000000

        max_mass = 0
        max_pcf = 0
        max_ea = 0

        data_path = self.data_path
        dir_list = os.listdir(data_path)
        for i in range(len(dir_list)):
            filename = os.path.join(data_path, dir_list[i], 'response.npy')
            data = np.load(filename, allow_pickle=True).item()
            mass = data['Mass']
            pcf = data['PCF']
            ea = data['EA']
            if mass>max_mass:
                max_mass=mass
            if mass<min_mass:
                min_mass=mass
            if pcf>max_pcf:
                max_pcf=pcf
            if pcf<min_pcf:
                min_pcf=pcf
            if ea>max_ea:
                max_ea=ea
            if ea<min_ea:
                min_ea=ea
        self.ea_range = [min_ea, max_ea]
        self.mass_range = [min_mass, max_mass]
        self.pcf_range = [min_pcf, max_pcf]

    def data_preprocessing(self, init_coordinate, node_coordinate, params, response):
        Height = params['Height']
        Radius = params['Radius']
        thick = params['thick']
        etan = params['ETAN']
        sigy = params['SIGY']
        rho = params['Rho']
        e = params['E']
        
        pcf = response['PCF']
        mass = response['Mass']
        ea =response['EA']
        
        init_coordinate = self.coordinate_normalized(coordinate=init_coordinate, H=185, R=27.5)
        node_coordinate = self.coordinate_normalized(coordinate=node_coordinate, H=185, R=27.5)
        
        H = Height/185
        R = Radius/27.5
        thick = thick/4
        etan = (etan-1000)/(1450-1000)
        sigy = (sigy-200)/(500-200)
        rho = (rho-2.7e-9)/(7.85e-9 - 2.7e-9)
        e = (e-69000)/(206000-69000)
        
        pcf = self.normalize(pcf, self.pcf_range)
        mass = self.normalize(mass, self.mass_range)
        ea = self.normalize(ea, self.ea_range)
        
        param = [H, R, thick, mass, etan, sigy, rho, e]
        res = [pcf, ea]
        
        t,c,nodenum = node_coordinate.shape
        
        if self.npoint<nodenum:
            sample_index = random_index(indexnum=self.npoint, up=nodenum-1, bottom=0)
            init_coordinate = init_coordinate[:,sample_index]
            node_coordinate = node_coordinate[:,:,sample_index]
        
        if self.point2img:
            init_coordinate = init_coordinate.reshape(c,64,-1)
            node_coordinate = node_coordinate.reshape(t,c,64,-1)
        
        param = np.array(param)
        res = np.array(res)
        
        node_coordinate = squeeze_node(node_coordinate)
        
        return init_coordinate, node_coordinate, param, res
    
    def normalize(self, d, r):
        b, u = r
        d = d/u
        return d
    
    def coordinate_normalized(self, coordinate, H, R):
        if len(coordinate.shape)==2:
            coordinate [0,:] = coordinate[0,:]/R
            coordinate [2,:] = coordinate[2,:]/R
            coordinate [1,:] = (370-coordinate[1,:])/H
        else:
            coordinate [:,0,:] = coordinate[:,0,:]/R
            coordinate [:,2,:] = coordinate[:,2,:]/R
            coordinate [:,1,:] = (370-coordinate[:,1,:])/H
        
        return coordinate
    
    def __len__(self):
        return len(self.dir_list)
    

class PointCloudDataset(Dataset):
    def __init__(self, data_path, 
                 transform=None, 
                 n_class=4, 
                 indices=None, 
                 one_hot=False,
                 key='NewLabel'):
        super(PointCloudDataset, self).__init__()
        self.dir_list = os.listdir(data_path)
        self.indices = indices
        self.data_path = data_path
        self.n_class = n_class
        
        self.transform = transform
        self.onehot = one_hot
        self.label_key = key
        #self.point2img = point2img
        #self.sq_chs = squeeze_channels
        
    def __getitem__(self, index):
        if self.indices is not None:
            new_index = self.indices[index]
            dir_name = self.dir_list[new_index]
        else:
            dir_name = self.dir_list[index]
        dir_path = os.path.join(self.data_path, dir_name)
        ori_node, label = self.read_data(dir_path)
        node_size = ori_node.shape
        if self.transform is not None:
            ori_node = self.transform(ori_node)
        out = {'node':ori_node, 'label':label, 'meta': {'node_size': node_size, 'index': index}}
        
        return out
    
    def read_data(self, path):
        param_file = os.path.join(path, 'parameter.npy')
        node_coord_file = os.path.join(path, 'node_coordinate.npy')
        params = np.load(param_file, allow_pickle=True).item()
        node_coordinate = np.load(node_coord_file) # [T, C, N]
        label = params[self.label_key]-1
        ori_node = self.data_preprocessing(node_coordinate=node_coordinate) 
        if self.onehot:
            label = num2onehot(label, self.n_class)  
        return ori_node, label
    
    def data_preprocessing(self, node_coordinate):
        node_coordinate = self.coordinate_normalized(coordinate=node_coordinate, H=185, R=27.5)
        #node_coordinate = self.sortnode(node_coordinate)
        #node_coordinate = node_coordinate[1:,:,:]
        return node_coordinate
    
    def coordinate_normalized(self, coordinate, H, R):
        if len(coordinate.shape)==2:
            coordinate [0,:] = coordinate[0,:]/R
            coordinate [2,:] = coordinate[2,:]/R
            coordinate [1,:] = (370-coordinate[1,:])/H
        else:
            coordinate [:,0,:] = coordinate[:,0,:]/R
            coordinate [:,2,:] = coordinate[:,2,:]/R
            coordinate [:,1,:] = (370-coordinate[:,1,:])/H
        
        return coordinate
    
    def sortnode(self, node):
        # node: [t,c,n]
        node_sortlist = np.lexsort((node[0,2,None], node[0,1,None],node[0,0,None]))
        node_sortlist = np.squeeze(node_sortlist, axis=0)
        node = node[:,:,node_sortlist]
        
        return node
    
    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        else:
            return len(self.dir_list)
        
        
        
        