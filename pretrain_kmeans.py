from config.configs import get_config
from utils.modellib import getModel, loadModel
from utils.datalib import getDataset, splitDataset
from utils.toollib import ToTensor, ToNumpy

import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from termcolor import colored

import sklearn
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans

from scipy.optimize import linear_sum_assignment

def getFeatures(net, loader):
    features = []
    labels =[]
    net.eval()
    
    for batch in tqdm(loader):
        nodes = ToTensor(batch['node']).cuda()
        label = ToNumpy(batch['label'])
        labels.append(label)
        with torch.no_grad():
            fea = net.backbone(nodes)
            fea = fea.cpu().numpy()
        features.append(fea)
        
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    print('feature shape: {}, labels shape: {}'.format(features.shape, labels.shape))
    return features, labels

def getPretrainModelandFeatures(task='supervised', model_type='PointSwin', point2img=False, opti='adamw', load_pretrain=True):
    assert task in ['supervised', 'simclr', 'byol', 'simsiam', 'moco']
    
    config = get_config(task=task, model_type=model_type, point2img=point2img, opti=opti)
    train_config = config.train_config
    paths = config.path_config
    model_config = config.model_config
    data_config = config.data_config
    
    data_config['task'] = 'supervised'
    data_config['one_hot']=False
    
    epoch_start=0
    # print cfg
    print(colored('Info Config:', 'red'))
    print(colored(config.info_config, 'red'))
    print(colored('Backbone Config:', 'red'))
    print(colored(model_config, 'red'))

    # create and load model
    print(colored('Get backbone', 'blue'))
    net = getModel(config)
    if load_pretrain:
        last_model_file = os.path.join(config.path_config['{}_checkpoint'.format(task)],'{}_last_epoch.pth'.format(task+model_type))
        if os.path.exists(last_model_file):
            net, epoch_start = loadModel(net=net,
                                         save_path=paths['{}_checkpoint'.format(task)],
                                         file_class='last',
                                         model_name=task+model_type, 
                                         task=task)
            print(colored('Restart from checkpoint {}'.format(paths['{}_dir'.format(task)]), 'blue'))
        else:
            print(colored('Connot found any pre-train model', 'yellow'))
    else:
        print('Random init model')
    # create dataset
    dataset = getDataset(data_config)
    if os.path.exists(paths['train_index']) and os.path.exists(paths['val_index']):
        splited_out = splitDataset(dataset=dataset, cfg=config, use_pretrain_indexes=True)
    else:
        splited_out = splitDataset(dataset=dataset, cfg=config)
    train_loader, val_loader = splited_out['train_dataloader'], splited_out['val_dataloader']
    train_num, val_num = splited_out['train_num'], splited_out['val_num']
    print(colored('Get dataset and dataloaders', 'blue'))

    device = train_config['device']
    net = net.to(device=device)
    
    print('get train features:')
    train_feas, train_labels = getFeatures(net=net, loader=train_loader)
    print('get val features:')
    val_feas, val_labels = getFeatures(net=net, loader=val_loader)
    
    return net, train_feas, train_labels, val_feas, val_labels

def getConfusionMatrix(pred, target):
    pred_cls = np.unique(pred)
    targer_cls = np.unique(target)
    clusters = max(len(pred_cls), len(targer_cls))
    CMatrix = np.zeros((clusters, clusters))
    for index in range(len(target)):
        i,j = target[index], pred[index]
        CMatrix[i,j]+=1
    #print(CMatrix)
    return CMatrix

def KMeans_train_and_predict(ncluster, train_fea, val_fea):
    print('training Kmeans model...')
    KmeansModel = KMeans(n_clusters=ncluster, max_iter=10000)
    KmeansModel.fit(train_fea)
    
    y_pred_train = KmeansModel.predict(train_fea)
    y_pred_val = KmeansModel.predict(val_fea)
    
    return KmeansModel, y_pred_train, y_pred_val

def getMatch(pred, label):
    assert len(pred)==len(label)
    cmx = getConfusionMatrix(pred=pred, target=label)
    cost = len(label)-cmx
    matches = linear_sum_assignment(cost)
    
    return matches

def assignPred(matches, pred):
    assign_y_pred = np.zeros_like(pred)
    for i in range(len(matches[1])):
        assign_y_pred[pred==matches[1][i]] = i
    return assign_y_pred

def evaluate_result(features, labels, assign_y_pred):
    result_NMI=metrics.normalized_mutual_info_score(labels, assign_y_pred)
    result_ARI = metrics.adjusted_rand_score(labels, assign_y_pred)
    result_ACC = metrics.accuracy_score(labels, assign_y_pred)
    result_lab_CH = metrics.calinski_harabasz_score(features, labels)
    result_lab_SC = metrics.silhouette_score(features, labels)
    result_pred_CH = metrics.calinski_harabasz_score(features, assign_y_pred)
    result_pred_SC = metrics.silhouette_score(features, assign_y_pred)
    res = {'NMI':result_NMI, 'ARI':result_ARI, 'ACC':result_ACC, 
           'labCH':result_lab_CH, 'labSC':result_lab_SC, 
           'predCH':result_pred_CH, 'predSC':result_pred_SC}
    return res

def train_and_val_Kmeans(ncluster, train_feas, train_labels, val_feas, val_labels):
    KmeansModel, y_pred_train, y_pred_val = KMeans_train_and_predict(ncluster=ncluster, train_fea=train_feas, val_fea=val_feas)
    print('get matches')
    matches = getMatch(pred=y_pred_train, label=train_labels)
    print('Assigning pred...')
    assignYtrain = assignPred(matches=matches, pred=y_pred_train)
    assignYval = assignPred(matches=matches, pred=y_pred_val)
    print('Counting confusion matrix...')
    Tcmx = getConfusionMatrix(pred=assignYtrain, target=train_labels)
    Vcmx = getConfusionMatrix(pred=assignYval, target=val_labels)
    print('train confusion matrix:')
    print(Tcmx)
    print('val confusion matrix')
    print(Vcmx)
    
    train_res = evaluate_result(features=train_feas,labels=train_labels, assign_y_pred=assignYtrain)
    val_res = evaluate_result(features=val_feas, labels=val_labels, assign_y_pred=assignYval)
    print('train result:')
    print(train_res)
    print('val result:')
    print(val_res)
    
    return {'kmeans':KmeansModel, 'tcmx':Tcmx, 'vcmx':Vcmx, 'train_res':train_res, 'val_res':val_res, 'matches':matches}

def train_pretextKmeans(ncluster, task='supervised', model_type='PointSwin', point2img=False, opti='adamw'):
    net, train_feas, train_labels, val_feas, val_labels = getPretrainModelandFeatures(task=task, model_type=model_type, point2img=point2img, opti=opti)
    kmeans_result = train_and_val_Kmeans(ncluster, train_feas, train_labels, val_feas, val_labels)