"""
# author: Zhaoyang Li
# 2022 08 15
# Central South University
"""
import numpy as np
import torch
import os
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .toollib import ToNumpy, ToTensor, squeeze_node, unsqueeze_node,\
    flatten_node, check_dirs, getModelFileExternal
from .modellib import getModel, loadModel
from .datalib import getDataset, getDataloader, splitDataset

#=====================Default Font=====================#
DefaultFont = {'family': 'Times New Roman',
               'weight': 'normal',
               'size': 20,}
#=====================Plot Node Coordinate=====================#

def plot_node(node, savepath, figsize=(10,10), flatten=True):
    # node: [c, l] or [t,c,l]
    node = ToNumpy(node)
    if flatten:
        node = flatten_node(node)
    if len(node.shape)==2:
        x = node[0,:]
        y = node[1,:]
        z = node[2,:]
        plot_xy(x, y, save_path=savepath, figsize=figsize)
        plot_xz(x, z, save_path=savepath, figsize=figsize)
    elif len(node.shape)==3:
        t, c, _ = node.shape
        for i in range(t):
            x, y, z = node[i,0,:], node[i,1,:], node[i,2,:]
            plot_xy(x, y, index=i, save_path=savepath, figsize=figsize)
            plot_xz(x, z, index=i, save_path=savepath, figsize=figsize)
    else:
        ValueError('len node out of the range')

def plot_xy(x, y, index=0, save_path='', figsize=(10,10), font=DefaultFont):
    filename = 'time_step_{}_xy.png'.format(index)
    filename = os.path.join(save_path, filename)
    plt.figure(num=1,figsize=figsize)
    plt.scatter(x, y, alpha=0.8)
    plt.xlabel('x', font)
    plt.ylabel('y', font)
    plt.ylim(-0.05,1)
    plt.savefig(filename)
    plt.close(fig=1)

def plot_xz(x, z, index=0, save_path='', figsize=(10,10), font=DefaultFont):
    filename = 'time_step_{}_xz.png'.format(index)
    filename = os.path.join(save_path, filename)
    plt.figure(num=1,figsize=figsize)
    plt.scatter(x, z, alpha=0.8)
    plt.xlabel('x', font)
    plt.ylabel('z', font)
    plt.savefig(filename)
    plt.close(fig=1) 
    
def plot_loss(loss, t=None, save_path='', figsize=(10,10), font=DefaultFont):
    filename = 'loss.png'
    filename = os.path.join(save_path, filename)
    plt.figure(num=1,figsize=figsize)
    if t is not None:
        plt.plot(t,loss)
    else:
        plt.plot(loss)
    plt.xlabel('t', font)
    plt.ylabel('Loss', font)
    plt.savefig(filename)
    plt.close(fig=1)
    
def plotPointCloud(node, epoch:int, save_root:str, index:int=0, flatten_node:bool=False):
    # node [b,tc,l] or [tc,l]
    node = ToNumpy(node)
    if flatten_node:
        # node [n,c,h,w] or [c,h,w]
        node = flatten_node(node)
        # node [b,tc,l] or [tc,l]
    save_path = os.path.join(save_root, 'epoch_{}'.format(epoch))
    check_dirs(save_path)
    if len(node.shape)==3:
        n, tc, l = node.shape
        for i in range(n):
            multi_node = unsqueeze_node(node[i])
            index_dir = os.path.join(save_path, '{:0>5d}'.format(index*n+i))
            check_dirs(index_dir)
            plot_node(node=multi_node, savepath=index_dir, flatten=False)
    elif len(node.shape)==2:
        node = unsqueeze_node(node)       
        index_dir = os.path.join(save_path, '{:0>5d}'.format(index))
        check_dirs(index_dir)
        plot_node(node=node, savepath=index_dir, flatten=False)
        
#=====================Plot Response Result=====================#
def plotContrastNode(pred_node, gt_node, epoch:int,
                     save_root:str, index:int=0, flatten_node:bool=True):
    # pred_node, gt_node: [n, t*c, l] or [t*c, l]
    pred_node = ToNumpy(pred_node)
    gt_node = ToNumpy(gt_node)
    assert len(pred_node.shape)==len(gt_node.shape)
    if flatten_node:
        # pred_node, gt_node: [n, t*c, h, w] or [t*c, h, w]
        pred_node = flatten_node(pred_node)
        gt_node = flatten_node(gt_node)
        # pred_node, gt_node: [n, t*c, l] or [t*c, l]
    save_path = os.path.join(save_root, 'epoch_{}'.format(epoch))
    check_dirs(save_path)
    if len(pred_node.shape)==3:
        n = pred_node.shape[0]
        for i in range(n):
            pm_node = unsqueeze_node(pred_node[i,:,:])
            gm_node = unsqueeze_node(gt_node[i,:,:])
            # pm_node, gm_mode: [tc, l]
            index_dir = os.path.join(save_path, '{:0>5d}'.format(index*n+i))
            check_dirs(index_dir)
            contrast_node(pm_node, gm_node,saveroot=index_dir, flatten=False)
    elif len(pred_node.shape)==2:
        pm_node =  unsqueeze_node(pred_node)
        gm_node =  unsqueeze_node(gt_node)
        index_dir = os.path.join(save_path, '{:0>5d}'.format(index))
        check_dirs(index_dir)
        contrast_node(pm_node, gm_node,saveroot=index_dir, flatten=False)

def contrast_node(pred_node, gt_node, saveroot:str, index:int=0, flatten=True):
    # pred_node, gt_node: [n, t*c, h, w] or [t*c, h, w]
    if flatten:
        pred_node = flatten_node(pred_node)
        gt_node = flatten_node(gt_node)
    # pred_node, gt_node: [n, t*c, l] or [t*c, l]
    figlist = ['x','y','z']
    if len(pred_node.shape)==2:
        px = [pred_node[0,:], pred_node[1,:], pred_node[2,:]]
        gx = [gt_node[0,:], gt_node[1,:], gt_node[2,:]]
        files = [os.path.join(saveroot, 'time_step_{}_p{}g{}.png'.format(index, figlist[i], figlist[i])) for i in range(len(figlist))]
        for i in range(len(figlist)):
            contrast_xy(x=px[i], y=gx[i], 
                        filename=files[i], 
                        xlabel='Pred_{}'.format(figlist[i]),
                        ylabel='Truth_{}'.format(figlist[i]))
        
    elif len(pred_node.shape)==3:
        t, c, l = pred_node.shape
        for i in range(t):
            px = [pred_node[i,0,:], pred_node[i,1,:], pred_node[i,2,:]]
            gx = [gt_node[i,0,:], gt_node[i,1,:], gt_node[i,2,:]]
            files = [os.path.join(saveroot, 'time_step_{}_p{}g{}.png'.format(i, figlist[i], figlist[i])) for i in range(len(figlist))]
            for j in range(len(figlist)):
                contrast_xy(x=px[j], y=gx[j], 
                        filename=files[j], 
                        xlabel='Pred_{}'.format(figlist[j]),
                        ylabel='Truth_{}'.format(figlist[j]))
    else:
        ValueError('len pred node is out of range')


def contrast_xy(x:np.ndarray, y:np.ndarray, filename:str, xlabel:str='x', ylabel:str='y',figsize=(10,10)):
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 20,}
    maxx, minx = np.max(x), np.min(x)
    maxy, miny = np.max(y), np.min(y)
    plt.figure(num=1,figsize=figsize)
    plt.scatter(x, y, alpha=0.8)
    plt.xlabel(xlabel, font)
    plt.ylabel(ylabel, font)
    plt.ylim(miny-1, maxy+1)
    plt.xlim(minx-1, maxx+1)
    plt.plot([minx-1,maxx+1], [miny-1,maxy+1], color='m', linestyle='--')
    plt.savefig(filename)
    plt.close(fig=1)

def contrast_res(pred_res, gt_res, saveroot:str, epoch:int):
    # pred_res: [b, dim], gt_res: [b, dim]
    pred_res = ToNumpy(pred_res)
    gt_res = ToNumpy(gt_res)
    savepath = os.path.join(saveroot, 'epoch_{}'.format(epoch))
    check_dirs(savepath)
    figlist = ['PCF', 'SEA']
    for i in range(len(figlist)):
        filename = os.path.join(savepath, 'pred_{}_and_gt_{}.png'.format(figlist[i], figlist[i]))
        contrast_xy(x=pred_res[:,i], y=gt_res[:,i], 
                    filename=filename, 
                    xlabel='Pred_{}'.format(figlist[i]),
                    ylabel='Truth_{}'.format(figlist[i]))

# Polt all nodes API
def plotResponseResult(nodes:dict, paths:dict, index:int, epoch:int, istrain:bool=True, FlattenNode:bool=False):
    init_node, gt_node, pred_node = nodes['init_node'], nodes['gt_node'], nodes['pred_node']
    if istrain:
        dirname = 'train'
    else:
        dirname = 'val'
    plotPointCloud(node=init_node,
                   epoch=epoch,
                   save_root=os.path.join(paths['Input'], dirname),
                   index=index, 
                   flatten_node=FlattenNode)
    plotPointCloud(node=gt_node,
                   epoch=epoch,
                   save_root=os.path.join(paths['GT'], dirname),
                   index=index,
                   flatten_node=FlattenNode)
    plotPointCloud(node=pred_node,
                   epoch=epoch,
                   save_root=os.path.join(paths['Pred'], dirname),
                   index=index,
                   flatten_node=FlattenNode)
    plotContrastNode(pred_node=pred_node,
                     gt_node=gt_node, 
                     epoch=epoch, 
                     save_root=os.path.join(paths['contrast'], dirname),
                     index=index,
                     flatten_node=FlattenNode)
    

def loadPretrainModelPlotResponseResult(config, file_class:str='epochs', epoch:int=100):
    task='ResponseProxy'
    
    train_config = config.train_config
    paths = config.path_config
    model_config = config.model_config
    data_config = config.data_config
    optimizer_config = config.optimizer_config
    
    model_name = config.model_name
    # create and load model
    proxymodel = getModel(config)
    file_ex = getModelFileExternal(file_class=file_class, epoch=epoch)
    model_file = os.path.join(config.path_config['Model_Library'],'{}_{}.pth'.format(model_name, file_ex))
    if os.path.exists(model_file):
        proxymodel, e = loadModel(net=proxymodel,
                                  save_path=paths['Model_Library'],
                                  file_class=file_class,
                                  model_name=model_name, 
                                  task=task, epoch=epoch)
        print('load pretrain model')
    else:
        raise ValueError('File Could not found: {}'.format(model_file))
    epoch = e
    dataset = getDataset(data_config)
    # split dataset
    splited_out = splitDataset(dataset=dataset, cfg=config)
    train_loader, val_loader = splited_out['train_dataloader'], splited_out['val_dataloader']
    
    device = train_config['device']
    FlattenNode = False
    if data_config['point2img']:
        FlattenNode = True
        
    proxymodel.to(device=device)
    proxymodel.eval()
    idx=0
    train_img_index = 0
    val_img_index = 0
    for batch in tqdm(train_loader):
        if idx%50==0:
            init_node = batch['init_node']
            gt_node = batch['out_node']
            param = batch['params']
            
            init_node = ToTensor(init_node).to(device=device, dtype=torch.float32)
            gt_node = ToTensor(gt_node).to(device=device, dtype=torch.float32)
            param = ToTensor(param).to(device=device, dtype=torch.float32)
            
            with torch.no_grad():
                pred_node, _ = proxymodel(init_node, param)
            p_nodes = {'init_node':init_node,
                          'gt_node':gt_node, 
                          'pred_node':pred_node}
            plotResponseResult(nodes=p_nodes, paths=paths, index=train_img_index, epoch=epoch, istrain=True, FlattenNode=FlattenNode)
            train_img_index+=1
        idx+=1
    idx = 0
    for batch in tqdm(val_loader):
        if idx%10==0:
            init_node = batch['init_node']
            gt_node = batch['out_node']
            param = batch['params']
            # to device
            init_node = ToTensor(init_node).to(device=device, dtype=torch.float32)
            gt_node = ToTensor(gt_node).to(device=device, dtype=torch.float32)
            param = ToTensor(param).to(device=device, dtype=torch.float32)
            # predict
            with torch.no_grad():
                pred_node, _ = proxymodel(init_node, param)
            p_nodes = {'init_node':init_node,
                       'gt_node':gt_node, 
                       'pred_node':pred_node}
            plotResponseResult(nodes=p_nodes, paths=paths, index=val_img_index, epoch=epoch, istrain=False, FlattenNode=FlattenNode)
            val_img_index+=1
        idx+=1
    print('end')
    
    
    
    
