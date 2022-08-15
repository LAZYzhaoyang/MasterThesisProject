"""
# author: Zhaoyang Li
# 2022 08 15
# Central South University
"""
import os
import pandas as pd
import numpy as np
import random

from tqdm import tqdm

from data_saver import save_data
from index_generator import random_index, sort_index, sample_from_multirange
from path_manager import read_doe_path, get_roots, get_all_doe_path, check_file
from data_reader import read_Doe_Info, read_elems, read_nodes, read_mass, read_matsum, read_rwforc, get_parameters, read_nodes_and_elems

def save_data_as_npy(root, save_path, rangelist, R, H, 
                     num=4096, 
                     sample_per_doe=5, 
                     node_keys = [0,1,2,4,5,6,8,9,10,12,13,14],
                     elem_keys = [0,1,3,4,5,6]):
    #########################################
    # root: data dir root, string
    # save_path: npy data save path, string
    # rangelist: time ranges, list
    # R: list of the radius, list
    # H: list of the height, list
    # num: number of sample nodes and elems, int
    # sample_per_doe: sample num of each doe, int
    #########################################
    roots, RH = get_roots(root=root, R=R, H=H)
    data_index = 0
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data_index = len(os.listdir(save_path))
    for i in range(len(roots)):
        doe_root = roots[i]
        print(doe_root)
        r,h = RH[i]
        doe_info = read_Doe_Info(doe_root)
        doe_num = len(doe_info)
        doe_path = os.path.join(doe_root,'approaches', 'doe_1')
        for j in tqdm(range(doe_num)):
            dir_name = 'run__{:0>5d}'.format(j+1)
            doe_dir = os.path.join(doe_path, dir_name, 'm_1')
            data = read_nodes_and_elems(data_path=doe_dir, 
                                        node_keys=node_keys, 
                                        elem_keys=elem_keys, 
                                        num=num,
                                        rangelist=rangelist,
                                        sample_per_doe=sample_per_doe)
            node_data, elem_data, node_indexes, elem_indexes, time_indexes = data
            
            mass = read_mass(data_path=doe_dir)
            PCF = read_rwforc(data_path=doe_dir)
            EA = read_matsum(data_path=doe_dir)
            params = get_parameters(doe_info=doe_info, index=j)
            
            for k in range(len(node_data)):
                save_dir_name = '{:0>6d}'.format(data_index)
                save_dir = os.path.join(save_path, save_dir_name)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_data(save_dir, node_data[k], elem_data[k], 
                          mass, PCF, EA, params, r=r, h=h, index=j, 
                          node_index=node_indexes[k], 
                          elem_index=elem_indexes[k],
                          time_index=time_indexes[k])
                data_index += 1
    print('end')
                
def save_coordinate_from_npy(data_path, coordinate_index):
    dir_list = os.listdir(data_path)
    for dirname in tqdm(dir_list):
        filename = os.path.join(data_path, dirname, 'node.npy')
        node = np.load(filename)
        node_coordinate = node[:,coordinate_index,:]
        
        init_node_coordinate = node_coordinate[0,:,:]
        node_coordinate = node_coordinate[1:,:,:]
        save_coordinate_file = os.path.join(data_path, dirname, 'node_coordinate.npy')
        np.save(save_coordinate_file, node_coordinate)
        save_init_file = os.path.join(data_path, dirname, 'init_node_coordinate.npy')
        np.save(save_init_file, init_node_coordinate)
    print('end')

if __name__ == '__main__':
    root = 'I:/MasterProgram/Thesis/data/tube'
    save_path = './data'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    rangelist = [[1,19], [20,39], [40,59], [60,79], [80,99]]
    R = [20, 25] 
    H = [800, 100, 120, 140, 160, 175]
    node_keys = [0,1,2,4,5,6,8,9,10,12,13,14]
    elem_keys = [0,1,3,4,5,6]
    num = 1024
    sample_per_doe = 5
    coordinate_index = [9,10,11]
    save_data_as_npy(root=root,
                     save_path=save_path,
                     rangelist=rangelist,
                     R=R, H=H, num=num,
                     sample_per_doe=sample_per_doe,
                     node_keys=node_keys,
                     elem_keys=elem_keys)
    save_coordinate_from_npy(data_path=save_path,
                             coordinate_index=coordinate_index)
    
    
            
            
