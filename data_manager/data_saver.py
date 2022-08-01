import os
import pandas as pd
import numpy as np
import random

from tqdm import tqdm

def save_data(save_path, nodes, elems, mass, PCF, EA, params, r,h, index, node_index, elem_index, time_index):
    filenames = ['node.npy', 'elements.npy', 
                 'parameter.npy', 'response.npy', 
                 'info.npy', 'node_index.npy', 
                 'elem_index.npy', 'time_index.npy']
    files = []
    for filename in filenames:
        file = os.path.join(save_path, filename)
        files.append(file)
    
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    SEA = EA/mass
    response = {'PCF':PCF, 'EA':EA, 'Mass':mass, 'SEA': SEA}
    
    info = {'R': r, 'H':h, 'index': index}
    
    data = [nodes, elems, params, response, info,  node_index, elem_index, time_index]
    
    for i in range(len(files)):
        filename = files[i]
        d = data[i]
        np.save(filename, d)

    #print('end')





