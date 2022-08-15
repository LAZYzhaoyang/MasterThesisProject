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

from index_generator import random_index, sample_from_multirange, sort_index

# read data

def read_nodes_and_elems(data_path, node_keys, elem_keys, num, rangelist, sample_per_doe=5):
    node_file = os.path.join(data_path, 'nodout.xls')
    elem_file = os.path.join(data_path, 'elout.xls')

    node = pd.read_excel(node_file, sheet_name=None, header=None)
    elem = pd.read_excel(elem_file, sheet_name=None, header=None)

    node_keywords = list(node)
    elem_keywords = list(elem)

    node_num, t1 = node[node_keywords[0]].to_numpy().shape
    elem_num, t2 = elem[elem_keywords[0]].to_numpy().shape

    node_indexes = []
    elem_indexes = []
    for i in range(sample_per_doe):
        node_index = random_index(num, node_num-1, 0)
        elem_index = random_index(num, elem_num-1, 0)
        node_indexes.append(node_index)
        elem_indexes.append(elem_index)
    
    time_indexes = sample_from_multirange(rangelist=rangelist, samplenum=sample_per_doe)
    for times in time_indexes:
        times.insert(0,0)

    node_data = read_nodes(node=node, 
                          keywords=node_keywords, 
                          keywords_index=node_keys, 
                          node_indexes=node_indexes,
                          time_indexes=time_indexes)
    elem_data = read_elems(elem=elem,
                           keywords=elem_keywords,
                           keywords_index=elem_keys,
                           elem_indexes=elem_indexes,
                           time_indexes=time_indexes)
    
    return node_data, elem_data, node_indexes, elem_indexes, time_indexes
    

def read_nodes(node, keywords, keywords_index, node_indexes, time_indexes):
    # read data without save
    assert len(node_indexes)==len(time_indexes)
    keywords = list(node)
    data = []
    for i in range(len(node_indexes)):
        data.append([])
        
    for i in keywords_index:
        sheet_data = node[keywords[i]].to_numpy()
        for j in range(len(node_indexes)):
            node_index = node_indexes[j]
            time_index = time_indexes[j]
            item_data = sheet_data[:, time_index]
            item_data = item_data[node_index, :].T
            item_data = item_data[:, np.newaxis, :]
            data[j].append(item_data)
            
    for i in range(len(data)):
        data[i] = np.concatenate(data[i], axis=1)
        
    return data

def read_elems(elem, keywords, keywords_index, elem_indexes, time_indexes):
    # read data without save
    assert len(elem_indexes)==len(time_indexes)
    data = []
    for i in range(len(elem_indexes)):
        data.append([])
        
    for i in keywords_index:
        sheet_data = elem[keywords[i]].to_numpy()
        for j in range(len(elem_indexes)):
            elem_index = elem_indexes[j]
            time_index = time_indexes[j]
            item_data = sheet_data[:, time_index]
            item_data = item_data[elem_index, :].T
            item_data = item_data[:, np.newaxis, :]
            data[j].append(item_data)
            
    for i in range(len(data)):
        data[i] = np.concatenate(data[i], axis=1)
        
    return data

def read_mass(data_path):
    data_file = os.path.join(data_path, 'mass.xls')
    mass = pd.read_excel(data_file,header=None)
    mass = mass.to_numpy()[0][0]
    return mass

def read_glstat():
    pass

def read_rwforc(data_path):
    data_file = os.path.join(data_path, 'rwforc.xls')
    rwforc = pd.read_excel(data_file)
    headers = list(rwforc.keys())
    keyword = 'StressRigidwall_resultant_force'
    assert keyword in headers
    PCF = np.max(np.abs(rwforc[keyword].to_numpy()))
    
    return PCF

def read_matsum(data_path):
    data_file = os.path.join(data_path, 'matsum.xls')
    matsum = pd.read_excel(data_file)
    headers = list(matsum.keys())
    keyword = headers[1]
    assert keyword in headers
    EA = np.max(np.abs(matsum[keyword].to_numpy()))
    
    return EA

def read_Doe_Info(data_path):
    data_file = os.path.join(data_path, 'DoeInfo.xlsx')
    doe_info = pd.read_excel(data_file)
    return doe_info

def get_parameters(doe_info, index):
    keywords = ['Height', 'H', 'Radius', 'R', 'thick', 'ETAN', 'SIGY', 'Rho', 'E', 'HumanLabel']
    params = {}
    for keyword in keywords:
        param = doe_info[keyword].to_numpy()[index].item()
        params[keyword] = param
    
    return params



