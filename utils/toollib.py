'''
Author: Zhaoyang Li
Central South University, Changsha, Hunan

All tool functions are stored here.
'''

import os
import pandas as pd
import numpy as np
import random
import os
import torch
import math
import logging
from logging import handlers
import errno

# time utils
def time2hms(second_time):
    mins = second_time//60
    sec = second_time%60
    hours = int(mins//60)
    mins = int(mins%60)
    
    return hours, mins, sec

# tensor utils
def ToTensor(data):
    if type(data)==np.ndarray:
        data = torch.from_numpy(data)
    return data

def ToNumpy(data):
    if type(data)==torch.Tensor:
        try:
            data = data.cpu().numpy()
        except:
            data = data.detach().cpu().numpy()
    return data

# node utils
def squeeze_node(node):
    # node [t,c,...] to [tc,...]
    node = ToNumpy(node)
    n = node.shape[0]
    nodes = []
    for i in range(n):
        nod = node[i]
        nodes.append(nod)
    nodes = np.concatenate(nodes, axis=0)
    return nodes

def unsqueeze_node(node, c=3):
    # node [tc, ...] to [t,c,...]
    node = ToNumpy(node)
    tc = node.shape[0]
    nodes = []
    for i in range(tc//c):
        nod = node[c*i:c*(i+1)]
        nod = nod[np.newaxis, ...]
        nodes.append(nod)
    nodes = np.concatenate(nodes, axis=0)
    return nodes

def flatten_node(nodeimg):
    nodeimg = ToNumpy(nodeimg)
    if len(nodeimg.shape)==3:
        c, _, _ = nodeimg.shape
        flatten_node = nodeimg.reshape((c,-1))       
    elif len(nodeimg.shape)==4:
        t, c, _, _ = nodeimg.shape
        flatten_node = nodeimg.reshape((t,c,-1))
    else:
        ValueError('node shape should be [c,h,w] or [t,c,h,w]')
    
    return flatten_node

def listnode2img(node, img_h=64):
    # node [t,c,l] to [t,c,h,w] hw=l
    node = ToNumpy(node)
    t,c,_ = node.shape
    node = node.reshape(t,c,img_h,-1)
    node = squeeze_node(node)
    
    return node

# one hot label
def num2onehot(num, classnum):
    one_hot = np.zeros((classnum))
    one_hot[num]=1
    return one_hot

# index utils
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

# dir utils
def check_dirs(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def get_optimizer(p, model, nheads=1, cluster_head_only=False):
    if cluster_head_only: # Only weights in the cluster head will be updated 
        for name, param in model.named_parameters():
                if 'cluster_head' in name:
                    param.requires_grad = True 
                else:
                    param.requires_grad = False 
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert(len(params) == 2 * nheads)

    else:
        params = model.parameters()
                

    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])
    
    elif p['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(params, **p['optimizer_kwargs'])
    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer

def get_logger(path, filename):
    logger_name = os.path.join(path, filename)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    fmt = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    format_str = logging.Formatter(fmt)
    th = handlers.TimedRotatingFileHandler(filename=logger_name, 
                                           when='D',backupCount=3, 
                                           encoding='utf-8')
    th.setFormatter(format_str)
    logger.addHandler(th)
    
    return logger
    

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank):
    model.eval()
    memory_bank.reset()

    for i, batch in enumerate(loader):
        nodes = batch['node']
        targets = batch['label']
        nodes=ToTensor(nodes)
        targets=ToTensor(targets)
        nodes = nodes.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        with torch.no_grad():
            output = model(nodes)
        memory_bank.update(output, targets)
        if i % 100 == 99 or i%(len(loader)-1)==0:
            print('Fill Memory Bank [%d/%d]' %(i+1, len(loader)))

def confusion_matrix(predictions, gt, class_names, output_file=None):
    # Plot confusion_matrix and store result to output_file
    import sklearn.metrics
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    confusion_matrix = sklearn.metrics.confusion_matrix(gt, predictions)
    #confusion_matrix = confusion_matrix / np.sum(confusion_matrix, 1)
    print(confusion_matrix)
    fig, axes = plt.subplots(1)
    cluster_name = ['cluster {}'.format(i+1) for i in range(len(class_names))]
    #plt.imshow(confusion_matrix, cmap='Blues')
    #plt.imshow(confusion_matrix)
    plt.imshow(confusion_matrix, cmap='Blues_r')
    axes.set_xticks([i for i in range(len(class_names))])
    axes.set_yticks([i for i in range(len(class_names))])
    axes.set_yticklabels(class_names, ha='right', fontsize=15)
    axes.set_xticklabels(cluster_name, ha='right', fontsize=15)
    axes.set_xlabel('聚类簇', fontsize=20)
    axes.set_ylabel('实际类', fontsize=20)
    
    for (i, j), z in np.ndenumerate(confusion_matrix):
        axes.text(j, i, '%d' %(z), ha='center', va='center', color='black', fontsize=10)


    plt.tight_layout()
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def save_clustering_stats(clustering_stats, path, filename):
    file = os.path.join(path, '{}.npy'.format(filename))
    acc, ari, nmi= clustering_stats['ACC'], clustering_stats['ARI'], clustering_stats['NMI']
    match, confusion_matrix = clustering_stats['hungarian_match'], clustering_stats['confusion_matrix']
    match = [[pred_i, target_i] for pred_i, target_i in match]
    nmi = nmi.item()
    out = {'ACC':acc, 'ARI':ari, 'NMI':nmi, 'hungarian_match':match, 'confusion_matrix':confusion_matrix}
    np.save(file, out)

def adjust_learning_rate(p, optimizer, epoch):
    lr = p.optimizer_config['optimizer_kwargs']['lr']
    
    if p.scheduler_config['scheduler'] == 'cosine':
        eta_min = lr * (p.scheduler_config['lr_decay_rate'] ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p.train_config['epochs'])) / 2
         
    elif p.scheduler_config['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p.scheduler_config['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p.scheduler_config['lr_decay_rate'] ** steps)

    elif p.scheduler_config['scheduler'] == 'constant':
        lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p.scheduler_config['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def euler_distance(a,b, dim=2):
    # a:[n1, d], b [n2, d]
    # return distance [n1, n2]
    a = ToTensor(a)
    b = ToTensor(b)
    
    n1, d1 = a.size()
    n2, d2 = b.size()
    
    assert d1==d2
    a = a.view(n1,1,d1)
    b = b.view(1,n2,d2)
    
    distanceab = torch.pow(torch.sum(torch.pow((a-b),dim), dim=-1), 1/dim)
    if dim==1:
        distanceab = torch.abs(distanceab)
    
    return distanceab

class MemoryBank(object):
    def __init__(self, n, dim, num_classes, temperature):
        self.n = n
        self.dim = dim 
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.temperature = temperature
        self.C = num_classes
    

    def weighted_knn(self, predictions):
        predictions = predictions.to(self.device)
        # perform weighted knn
        retrieval_one_hot = torch.zeros(self.K, self.C).to(self.device) #[k, c]
        batchSize = predictions.shape[0] # b
        correlation = torch.matmul(predictions, self.features.t()) # [b, n]
        correlation = euler_distance(predictions, self.features)
        correlation = torch.exp(-torch.log(correlation+1))
        yd, yi = correlation.topk(self.K, dim=1, largest=True, sorted=True) # yd:[b, k], yi: [k, 1]
        candidates = self.targets.view(1,-1).expand(batchSize, -1) # [b,n,1]
        retrieval = torch.gather(candidates, 1, yi) # [b,k,1]
        retrieval_one_hot.resize_(batchSize * self.K, self.C).zero_() # [bk, c]
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1) # retrieval.view(-1, 1) [bk, 1] ; retrieval_one_hot[retrieval]=1
        yd_transform = yd.clone().div_(self.temperature).exp_() # exp(yd/temperature) [b, k] Similarity
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , self.C), # [b,k,c] mul [b,k,1]
                          yd_transform.view(batchSize, -1, 1)), 1) # [b, 1, c] 相似度求和，然后投票
        _, class_preds = probs.sort(1, True) # sort in 1 dim, descending=True, return sorted tensor, indices
        class_pred = class_preds[:, 0]

        return class_pred

    def knn(self, predictions):
        predictions = predictions.to(self.device)
        # perform knn
        correlation = torch.matmul(predictions, self.features.t())
        sample_pred = torch.argmax(correlation, dim=1)
        class_pred = torch.index_select(self.targets, 0, sample_pred)
        return class_pred

    def mine_nearest_neighbors(self, topk, calculate_accuracy=True):
        # mine the topk nearest neighbors for every sample
        import faiss
        features = self.features.cpu().numpy()
        n, dim = features.shape[0], features.shape[1]
        index = faiss.IndexFlatIP(dim)
        #index = faiss.index_cpu_to_all_gpus(index)
        index.add(features)
        distances, indices = index.search(features, topk+1) # Sample itself is included
        # 给每个feature找到离他最近的k个邻居
        # evaluate 
        if calculate_accuracy:
            targets = self.targets.cpu().numpy()
            neighbor_targets = np.take(targets, indices[:,1:], axis=0) # Exclude sample itself for eval
            anchor_targets = np.repeat(targets.reshape(-1,1), topk, axis=1)
            accuracy = np.mean(neighbor_targets == anchor_targets)
            return indices, accuracy
        
        else:
            return indices

    def reset(self):
        self.ptr = 0 
        
    def update(self, features, targets):
        b = features.size(0)
        
        assert(b + self.ptr <= self.n)
        #print(features.size(), targets.size())
        self.features[self.ptr:self.ptr+b].copy_(features.detach())
        self.targets[self.ptr:self.ptr+b].copy_(targets.detach())
        self.ptr += b

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')

#========================BYOL========================#
def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def update_moving_average(ma_model, current_model, beta=0.99):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        if old_weight is None:
            ma_params.data = up_weight
        else:
            ma_params.data = old_weight * beta + (1-beta)*up_weight