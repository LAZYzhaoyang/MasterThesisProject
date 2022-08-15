from ..config.configs import get_config
from .modellib import getModel, loadModel
from .datalib import getDataset, splitDataset
from .toollib import ToTensor, ToNumpy, check_dirs

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
#from sklearn.externals import joblib
import joblib
from scipy.optimize import linear_sum_assignment

def getFeatures(net, loader, device=torch.device('cuda')):
    features = []
    labels =[]
    net.eval()
    
    for batch in tqdm(loader):
        nodes = ToTensor(batch['node']).to(device=device)
        label = ToNumpy(batch['label'])
        labels.append(label)
        with torch.no_grad():
            fea = net(nodes)
            fea = fea.cpu().numpy()
        features.append(fea)
        
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    print('feature shape: {}, labels shape: {}'.format(features.shape, labels.shape))
    return features, labels

def getPretrainModelandFeatures(task='supervised', model_type='PointSwin', point2img=False, opti='adamw', load_pretrain=True, return_config=False):
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
        net, epoch_start = loadPretrainModel(net=net, paths=paths, task=task, model_type=model_type,
                                             file_class='last')
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
    train_feas, train_labels = getFeatures(net=net.backbone, loader=train_loader, device=device)
    print('get val features:')
    val_feas, val_labels = getFeatures(net=net.backbone, loader=val_loader, device=device)
    
    if return_config:
        return net, config, train_feas, train_labels, val_feas, val_labels
    else:
        return net, train_feas, train_labels, val_feas, val_labels

def loadPretrainModel(net, paths, task='supervised', model_type='PointSwin', file_class='last', epoch:int=100, loadBackboneOnly=True):
    epoch_start=0
    class_list=['best', 'last', 'epochs']
    file_extension_name=['best', 'last_epoch', 'checkpoint-epoch{}'.format(epoch)]
    assert file_class in class_list, 'file class must be one of [best, last, epoch]'
    for i in range(len(class_list)):
        if file_class == class_list[i]:
            file_ex = file_extension_name[i]
            modelfilename = os.path.join(paths['{}_checkpoint'.format(task)],'{}_{}.pth'.format(task+model_type, file_ex))
    if os.path.exists(modelfilename):
        net, epoch_start = loadModel(net=net,
                                    save_path=paths['{}_checkpoint'.format(task)],
                                    file_class=file_class,
                                    model_name=task+model_type, 
                                    task=task,
                                    epoch=epoch,
                                    load_backbone_only=loadBackboneOnly)
        print(colored('Restart from checkpoint {}'.format(paths['{}_dir'.format(task)]), 'blue'))
    else:
        print(colored('Connot found any pre-train model', 'yellow'))
        
    return net, epoch_start
    

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
    result = val_Kmeans(y_pred_train=y_pred_train, y_pred_val=y_pred_val, train_feas=train_feas, train_labels=train_labels,
                        val_feas=val_feas, val_labels=val_labels)
    result['kmeans'] = KmeansModel
    return result

def val_Kmeans(y_pred_train, y_pred_val, train_feas, train_labels, val_feas, val_labels):
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
    
    return {'tcmx':Tcmx, 'vcmx':Vcmx, 'train_res':train_res, 'val_res':val_res, 'matches':matches}

def train_pretextKmeans(ncluster, task='supervised', model_type='PointSwin', point2img=False, opti='adamw'):
    net, train_feas, train_labels, val_feas, val_labels = getPretrainModelandFeatures(task=task, model_type=model_type, point2img=point2img, opti=opti)
    kmeans_result = train_and_val_Kmeans(ncluster, train_feas, train_labels, val_feas, val_labels)
    
    
    
class PretrainKmeans(object):
    def __init__(self, ncluster, backbone_type:str, pretext:str='supervised', point2img=False, opti='adamw', 
                 load_pretrain=True, result_root:str='./result/Clustering/tube'):
        super().__init__()
        assert pretext in ['supervised', 'simclr', 'byol', 'simsiam']
        self.ncluster = ncluster
        self.res_root = result_root
        
        self.backbone_type = backbone_type
        self.pretext = pretext
        
        self.point2img = point2img
        self.opti = opti
        
        self.make_config()
        
        print(colored('Get backbone', 'blue'))
        net = getModel(self.config)
        if load_pretrain:
            net, epoch_start = loadPretrainModel(net=net, paths=self.paths, task=pretext, model_type=backbone_type, file_class='last')
        else:
            print('Random init model')
        self.backbone = net.backbone
        
        if type(ncluster)==int:
            self.kmeans_head = KMeans(n_clusters=ncluster, max_iter=10000)
        elif type(ncluster)==list:
            self.kmeans_head = []
            for nc in ncluster:
                kmeans = KMeans(n_clusters=nc, max_iter=10000)
                self.kmeans_head.append(kmeans)
        else:
            raise ValueError('ncluster must be int or list[int].')
        
        self.istrain = False
        
        self.make_model_dir()
        
    def make_config(self):
        config = get_config(task=self.pretext, model_type=self.backbone_type, point2img=self.point2img, opti=self.opti)
        self.config = config
        self.train_config = config.train_config
        self.paths = config.path_config
        self.model_config = config.model_config
        self.data_config = config.data_config
    
        self.data_config['task'] = 'supervised'
        self.data_config['one_hot']=False
        self.device = self.train_config['device']
    
    def make_model_dir(self):
        model_dir = {}
        root = os.path.join(self.res_root, self.backbone_type, 'pretrain_kmeans', self.pretext)
        model_dir['root_dir'] = root
        model_dir['backbone'] = os.path.join(root, 'backbone')
        check_dirs(root)
        if type(self.ncluster)==int:
            model_dir['ncluster{}'.format(self.ncluster)] = os.path.join(root, 'ncluster{}'.format(self.ncluster))
        elif type(self.ncluster)==list:
            for i in range(len(self.ncluster)):
                nc = self.ncluster[i]
                model_dir['ncluster{}'.format(nc)] = os.path.join(root, 'ncluster{}'.format(nc))
        else:
            raise ValueError('ncluster must be int or list[int].')
        for k in model_dir.keys():
            check_dirs(model_dir[k])
        self.model_dir = model_dir
        
    def train(self):
        # create dataset
        dataset = getDataset(self.data_config)
        if os.path.exists(self.paths['train_index']) and os.path.exists(self.paths['val_index']):
            splited_out = splitDataset(dataset=dataset, cfg=self.config, use_pretrain_indexes=True)
        else:
            splited_out = splitDataset(dataset=dataset, cfg=self.config)
        train_loader, val_loader = splited_out['train_dataloader'], splited_out['val_dataloader']
        train_num, val_num = splited_out['train_num'], splited_out['val_num']
        print(colored('Get dataset and dataloaders', 'blue'))

        self.backbone = self.backbone.to(device=self.device)
        
        print('get train features:')
        train_feas, train_labels = getFeatures(net=self.backbone, loader=train_loader, device=self.device)
        print('get val features:')
        val_feas, val_labels = getFeatures(net=self.backbone, loader=val_loader, device=self.device)
        
        if type(self.ncluster)==int:
            self.kmeans_head.fit(train_feas)
            y_pred_train = self.kmeans_head.predict(train_feas)
            y_pred_val = self.kmeans_head.predict(val_feas)
            self.kmeans_result = val_Kmeans(y_pred_train=y_pred_train, y_pred_val=y_pred_val, 
                                            train_feas=train_feas, train_labels=train_labels,
                                            val_feas=val_feas, val_labels=val_labels)
            self.match = self.kmeans_result['matches']
        elif type(self.ncluster)==list:
            self.kmeans_result=[]
            self.match = []
            for i in range(len(self.kmeans_head)):
                self.kmeans_head[i].fit(train_feas)
                y_pred_train = self.kmeans_head[i].predict(train_feas)
                y_pred_val = self.kmeans_head[i].predict(val_feas)
                kmeans_result = val_Kmeans(y_pred_train=y_pred_train, y_pred_val=y_pred_val, 
                                            train_feas=train_feas, train_labels=train_labels,
                                            val_feas=val_feas, val_labels=val_labels)
                self.kmeans_result.append(kmeans_result)
                self.match.append(kmeans_result['matches'])
        else:
            raise ValueError('ncluster must be int or list[int].')
        self.istrain =True
        
    def __call__(self, x):
        if self.istrain:
            x = ToTensor(x).to(self.device)
            self.backbone = self.backbone.to(device=self.device)
            
            self.backbone.eval()
            fea = self.backbone(x)
            fea = ToNumpy(fea)
            if type(self.ncluster)==int:
                pred = self.kmeans_head.predict(fea)
                pred = assignPred(self.match, pred)
            elif type(self.ncluster)==list:
                pred = []
                for i in range(len(self.kmeans_head)):
                    prob =self.kmeans_head[i].predict(fea)
                    prob = assignPred(self.match[i], prob)
                    pred.append(prob)
            else:
                raise ValueError('ncluster must be int or list[int].')
            return pred
        else:
            raise ValueError('this model have not been trained.')
    
    def save(self):
        if self.istrain:
            backbonefile = os.path.join(self.model_dir['backbone'], '{}_backbone.pth'.format(self.backbone_type))
            torch.save(self.backbone, backbonefile)
            if type(self.ncluster)==int:
                path = self.model_dir['ncluster{}'.format(self.ncluster)]
                filename = os.path.join(path, 'kmeans.model')
                joblib.dump(self.kmeans_head, filename)
                matchfile = os.path.join(path, 'match.npy')
                np.save(matchfile, self.match, allow_pickle=True)
            elif type(self.ncluster)==list:
                for i in range(len(self.ncluster)):
                    nc = self.ncluster[i]
                    path = self.model_dir['ncluster{}'.format(nc)]
                    filename = os.path.join(path, 'kmeans.model')
                    joblib.dump(self.kmeans_head[i], filename)
                    matchfile = os.path.join(path, 'match.npy')
                    np.save(matchfile, self.match[i], allow_pickle=True)
            else:
                raise ValueError('ncluster must be int or list[int].')
        else:
            raise ValueError('this model have not been trained.')
    
    def load(self, root:str, backbone_type:str='PointSwin'):
        plist = os.listdir(root)
        ncluster = []
        kmeans_heads =[]
        matches = []
        for dirname in plist:
            if dirname == 'backbone':
                filename = os.path.join(root, dirname, '{}_backbone.pth'.format(backbone_type))
                if os.path.exists(filename):
                    self.backbone = torch.load(filename)
                    self.backbone_type = backbone_type
            elif 'ncluster' in dirname:
                num = int(dirname[len('ncluster'):])
                filename = os.path.join(root, dirname, 'kmeans.model')
                matchfile = os.path.join(root, dirname, 'match.npy')
                if os.path.exists(filename):
                    ncluster.append(num)
                    kmeansmodel =joblib.load(filename)
                    kmeans_heads.append(kmeansmodel)
                    match = np.load(matchfile, allow_pickle=True)
                    matches.append(match)
            else:
                continue
        if len(ncluster)==1:
            ncluster = ncluster[0]
            kmeans_heads = kmeans_heads[0]
            matches = matches[0]
        self.ncluster = ncluster
        self.kmeans_head = kmeans_heads
        self.match = matches
        self.make_config()
        self.make_model_dir()
        
        self.istrain = True
        
    def to(self, device):
        self.device = device
        self.backbone = self.backbone.to(device=device)
        
