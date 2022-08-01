import os
import pandas as pd
import numpy as np
import random

from tqdm import tqdm

def getNewLabel(ori_root, r, h, i, key='NewLabel'):
    excel_dir = 'TubeR{}H{}'.format(r,h)
    excel_file = os.path.join(ori_root, excel_dir, 'DoeInfo.xlsx')
    df = pd.read_excel(excel_file)
    newlabel = df[key][i]
    del df
    return newlabel

def addNewLabel2currentData(data_root, ori_root, key='NewLabel'):
    samples = os.listdir(data_root)
    for sam in tqdm(samples):
        infofile = os.path.join(data_root, sam, 'info.npy')
        paramfile = os.path.join(data_root, sam, 'parameter.npy')
        info = np.load(infofile, allow_pickle=True).item()
        params = np.load(paramfile, allow_pickle=True).item()
        R, H, i = info['R'], info['H'], info['index']
        newlabel = getNewLabel(ori_root, r=R, h=H, i=i, key=key)
        params[key] = newlabel
        np.save(paramfile, params)
    print('end')