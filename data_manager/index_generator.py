import os
import pandas as pd
import numpy as np
import random

from tqdm import tqdm

# gen index

def random_index(indexnum, up, bottom):
    # 在一定范围内随机采indexnum个数
    # up: up of the range
    # bottom: bottom of the range
    # indexnum: number of sample index
    if bottom>up:
        up, bottom = bottom, up
    index = random.sample(range(bottom, up), indexnum)
    #print(index)
    index.sort()
    return index

def sort_index(index_list):
    # 将采样的序号排序好(转置)
    # index_list: the list of sample indexes
    sort_indexes = list(map(list, zip(*index_list)))
    return sort_indexes

def sample_from_multirange(rangelist, samplenum):
    # 在若干个范围内采样samplenum个序号
    # rangelist(n,2): list of ranges
    # samplenum: number of sample index
    index_list = []
    for r in rangelist:
        u,b = r
        indexes = random_index(samplenum, u, b)
        index_list.append(indexes)
    index = sort_index(index_list)
    return index


