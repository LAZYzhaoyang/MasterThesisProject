import os
import pandas as pd
import numpy as np
import random

from tqdm import tqdm

# read doe path

def read_doe_path(root):
    root = os.path.join(root, 'approaches', 'doe_1')
    doe_paths = []
    doe_dirs = os.listdir(root)
    for dirname in doe_dirs:
        doe_dirname = os.path.join(root, dirname)
        if os.path.isdir(doe_dirname):
            doe_path = os.path.join(doe_dirname, 'm_1')
            doe_paths.append(doe_path)
    return doe_paths

def get_roots(root, R=[20, 25], H=[800, 100, 120, 140, 160, 175]):
    roots = []
    rh = []
    for r in R:
        for h in H:
            rh.append([r,h])
            dirname = 'TubeR{}H{}'.format(r,h)
            dirroot = os.path.join(root, dirname)
            roots.append(dirroot)
    return roots, rh

def get_all_doe_path(root, R, H):
    roots = get_roots(root, R, H)
    paths = []
    for tube_root in roots:
        doe_paths = read_doe_path(tube_root)
        paths.extend(doe_paths)
    return paths 

def check_file(paths):
    for path in tqdm(paths):
        nodfile = os.path.join(path, 'nodout.xls')
        elfile = os.path.join(path, 'elout.xls')
        glstatfile =  os.path.join(path, 'glstat.xls')
        rwforcfile =  os.path.join(path, 'rwforc.xls')
        
        file_list = [nodfile, elfile, glstatfile, rwforcfile]
        for file in file_list:
            if not os.path.exists(file):
                print(file)
    print('checking end')




