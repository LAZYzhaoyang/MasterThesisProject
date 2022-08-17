"""
# author: Zhaoyang Li
# 2022 08 15
# Central South University
"""

import os
import time
import numpy as np
from tqdm import tqdm
import logging
from logging import handlers
from termcolor import colored
import copy
from math import ceil

import torch
import torchvision
import torch.nn as nn
import torch.utils.data as D
from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.enabled = True

from torch import optim
from torch.utils.data import SubsetRandomSampler, RandomSampler
from torch.cuda.amp import autocast, GradScaler#need pytorch>1.6
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

# my lib
from .plotlib import plotPointCloud, plotContrastNode, contrast_res
from .toollib import check_dirs, time2hms, ToNumpy, ToTensor, AverageMeter, \
    ProgressMeter, get_logger, get_optimizer, adjust_learning_rate, \
    save_clustering_stats, MemoryBank, fill_memory_bank, set_requires_grad, update_moving_average
from .modellib import getModel, saveModel, loadModel
from ..config.configs import get_config, ClusterConfig
from .datalib import getDataset, getDataloader, splitDataset
from .losslib import ResponseLoss, get_criterion
from .evaluationlib import get_predictions, hungarian_evaluate, contrastive_evaluate, scan_evaluate
from ..model.ClusteringModel import DeepClusterCenter

#===================================Main Train Function===================================#
def main_train(task:str='ResponseProxy', model_type:str='PointSwin', 
               ncluster:int=4, point2img=False, opti='adamw', 
               pretext:str='byol', epochs:int=200, val_rate=0.2, batch_size:int=16, 
               num_worker:int=4, learn_rate=0.0001, feature_dim:int=256, 
               save_model_epoch:int=50, data_root=None, result_root=None):
    CFG = get_task_config(task=task, model_type=model_type, ncluster=ncluster,
                          point2img=point2img, opti=opti, pretext=pretext, 
                          epochs=epochs, val_rate=val_rate, batch_size=batch_size,
                          num_worker=num_worker, learn_rate=learn_rate, 
                          feature_dim=feature_dim, save_model_epoch=save_model_epoch, 
                          data_root=data_root, result_root=result_root)
    if task == 'ResponseProxy':
        train_proxy(config=CFG)
    elif task == 'supervised':
        train_supervised(config=CFG)
    elif task == 'scan':
        train_scan(config=CFG)
    elif task =='spice':
        train_spice(config=CFG)
    elif task == 'simclr':
        train_simclr(config=CFG)
    elif task == 'byol':
        train_byol(config=CFG)
    elif task == 'simsiam':
        train_simsiam(config=CFG)
    elif task == 'deepcluster':
        train_deepcluster(config=CFG)
    else:
        raise ValueError('Invalid task {}'.format(task))
        

def get_task_config(task:str='ResponseProxy', model_type:str='PointSwin', 
                    ncluster:int=4, point2img=False, opti='adamw', 
                    pretext:str='byol', epochs:int=200, val_rate=0.2, 
                    batch_size:int=16, num_worker:int=4, learn_rate=0.0001, 
                    feature_dim:int=256, save_model_epoch:int=50, 
                    data_root=None, result_root=None):
    config = get_config(task=task, ncluster=ncluster, model_type=model_type, opti=opti, 
                        point2img=point2img, data_root=data_root, result_root=result_root,
                        pretext=pretext, feature_dim=feature_dim)
    
    if task == 'ResponseProxy':
        config.train_config['lr']=learn_rate
    else:
        config.optimizer_config['optimizer_kwargs']['lr']=learn_rate
        
    config.train_config['train_loader']['BatchSize'] = batch_size
    config.train_config['val_loader']['BatchSize'] = batch_size
    config.train_config['train_loader']['NumWorker'] = num_worker
    config.train_config['val_loader']['NumWorker'] = num_worker
    
    config.train_config['epochs'] = epochs
    config.train_config['val_rate'] = val_rate
    config.train_config['save_model_epoch'] = save_model_epoch
    
    return config


#===================================Training of Response Proxy Model===================================#
def train_proxy(config):
    # create config
    task='ResponseProxy'
    
    train_config = config.train_config
    paths = config.path_config
    model_config = config.model_config
    data_config = config.data_config
    optimizer_config = config.optimizer_config
    
    print(colored('Info Config:', 'blue'))
    print(colored(config.info_config, 'yellow'))
    print(colored('Model Config:', 'blue'))
    print(colored(model_config, 'yellow'))
    print(colored('Train Config:', 'blue'))
    print(colored(train_config, 'yellow'))
    print(colored('Paths:', 'blue'))
    print(colored(paths, 'yellow'))
    print(colored('Data Config:', 'blue'))
    print(colored(data_config, 'yellow'))
    print(colored('Optimizer Config:', 'blue'))
    print(colored(optimizer_config, 'yellow'))

    # create a tensorboard
    writer = SummaryWriter(config.path_config['Log'])
    model_name = config.model_name
    epoch_start=0
    # create and load model
    proxymodel = getModel(config)
    last_model_file = os.path.join(config.path_config['Model_Library'],'{}_last_epoch.pth'.format(model_name))
    if os.path.exists(last_model_file):
        proxymodel, epoch_start = loadModel(net=proxymodel,
                                            save_path=paths['Model_Library'],
                                            file_class='last',
                                            model_name=model_name, 
                                            task=task)
        print('load pretrain model')
    
    # create dataset, criterion, optimizer and scheduler
    dataset = getDataset(data_config)
    criterion = ResponseLoss()
    optimizer = get_optimizer(p=optimizer_config, model=proxymodel)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=3, eta_min=1e-6, last_epoch=-1)
    
    # split dataset
    splited_out = splitDataset(dataset=dataset, cfg=config)
    train_loader, val_loader = splited_out['train_dataloader'], splited_out['val_dataloader']
    train_num, val_num = splited_out['train_num'], splited_out['val_num']
    # create logger
    logger = get_logger(path=paths['Log'], filename='{}Model_train.log'.format(task))
    
    scaler = GradScaler()
    # train setting
    train_step = 0
    val_step = 0
    train_losses = []
    val_losses = []
    best_loss = float('inf')
    start_time = time.time()
    best_epoch = 0
    device = train_config['device']
    FlattenNode = False
    if data_config['point2img']:
        FlattenNode = True
        
    proxymodel.to(device=device)
    
    for epoch in range(epoch_start, epoch_start+train_config['epochs']):
        print('training '+str(epoch)+'th step...')
        proxymodel.train()
        idx=0
        load_time = time.time()
        train_img_index = 0
        for batch in tqdm(train_loader):
            init_node = batch['init_node']
            gt_node = batch['out_node']
            param = batch['params']
            res = batch['res']
            train_time = time.time()
            train_loader_size = train_loader.__len__()
            
            init_node = ToTensor(init_node).to(device=device, dtype=torch.float32)
            gt_node = ToTensor(gt_node).to(device=device, dtype=torch.float32)
            param = ToTensor(param).to(device=device, dtype=torch.float32)
            res = ToTensor(res).to(device=device, dtype=torch.float32)
            #label = label.to(device=device, dtype=torch.float32)
            
            optimizer.zero_grad()
            if train_config['use_fp16']:
                with autocast():
                    pred_node, pred_res = proxymodel(init_node, param)
                    loss = criterion(pred_node, gt_node, pred_res, res)
                writer.add_scalar('{}_Loss/train'.format(model_name), loss.item(), train_step)
                if train_step % 10 == 0:
                    train_losses.append(loss.item())
                scaler.scale(loss).backward()
                scaler.step(optimizer=optimizer)
                scaler.update()
            else:
                pred_node, pred_res = proxymodel(init_node, param)
                loss = criterion(pred_node, gt_node, pred_res, res)
                writer.add_scalar('{}_Loss/train'.format(model_name), loss.item(), train_step)
                if train_step % 10 == 0:
                    train_losses.append(loss.item())
                loss.backward()
                optimizer.step()
            #print(pred)
            scheduler.step(epoch + idx / train_loader_size)
            #scheduler.step()
            total_time=time.time() - start_time
            hours, mins, sec = time2hms(total_time)
            
            infomation = 'Epoch: [{:>2d}/{:>2d}] || Time: {:>3d} H {:>2d} M {:.3f} s || Loss: {:.8f} || train item: {:>5d} / {:>5d} || item time: {:.3f} sec || train time: {:.3f} sec'.format(
                epoch, epoch_start + train_config['epochs'], hours, mins, sec, loss.item(), 
                train_step%((train_num)//train_config['train_loader']['BatchSize']), 
                (train_num)//train_config['train_loader']['BatchSize'], 
                time.time()-load_time, time.time()-train_time)
            if train_step % 100==0:
                logger.info(infomation)
                
            idx = idx+1
            train_step+=1
            load_time = time.time()
            
        # validation
        proxymodel.eval()
        val_img_index = 0
        pred_used_times = []
        pred_result = []
        gt_result = []
        for batch in tqdm(val_loader):
            init_node = batch['init_node']
            gt_node = batch['out_node']
            param = batch['params']
            res = batch['res']
            #optimizer.zero_grad()
            # save time
            
            val_time = time.time()
            # to device
            init_node = ToTensor(init_node).to(device=device, dtype=torch.float32)
            gt_node = ToTensor(gt_node).to(device=device, dtype=torch.float32)
            param = ToTensor(param).to(device=device, dtype=torch.float32)
            res = ToTensor(res).to(device=device, dtype=torch.float32)
            # predict
            with torch.no_grad():
                pred_node, pred_res = proxymodel(init_node, param)
                val_loss = criterion(pred_node, gt_node, pred_res, res)
            # count time
            pred_used_time = time.time()-val_time
            pred_used_times.append(pred_used_time)
            if (epoch+1)%train_config['save_model_epoch']==0 or epoch==train_config['epochs']-1:
                pred_res = ToNumpy(pred_res)
                res = ToNumpy(res)
                pred_result.append(pred_res)
                gt_result.append(res)
                
                val_img_index += 1
                
            
            writer.add_scalar('{}_Loss/val'.format(model_name), val_loss.item(), val_step)
            val_losses.append(val_loss.item())
            val_step = val_step+1 
        
        if len(pred_result)>0:
            pred_result = np.concatenate(pred_result, axis=0)
            gt_result = np.concatenate(gt_result, axis=0)
            contrast_res(pred_res=pred_result, gt_res=gt_result, saveroot=os.path.join(paths['contrast'], 'val'), epoch=epoch)
               
            
            
        mean_pred_time = np.mean(pred_used_times)
        mean_val_loss = np.mean(val_losses)
        logger.info('mean val loss: {:.8f} || mean predict time: {:.8f} sec'.format(mean_val_loss, mean_pred_time))
        print('mean val loss: {:.8f} || mean predict time: {:.8f} sec'.format(mean_val_loss, mean_pred_time))
        saveModel(net=proxymodel, save_path=paths['Model_Library'], epoch=epoch,
                  filename='{}_last_epoch.pth'.format(model_name), optimizer=optimizer, task=task)
        
        if mean_val_loss< best_loss:
            best_loss = mean_val_loss
            best_epoch = epoch
            saveModel(net=proxymodel, save_path=paths['Model_Library'], epoch=epoch,
                      filename='{}_best.pth'.format(model_name), optimizer=optimizer,
                      task=task)
        
        if (epoch+1)%train_config['save_model_epoch']==0:
            saveModel(net=proxymodel, save_path=paths['Model_Library'], epoch=epoch,
                      filename='{}_checkpoint-epoch{}.pth'.format(model_name, epoch),
                      optimizer=optimizer, task=task)
            
        
    trainlossfile = os.path.join(paths['Loss'], 'trainloss.npy')
    vallossfile = os.path.join(paths['Loss'], 'valloss.npy')
    
    np.save(trainlossfile, train_losses)
    np.save(vallossfile, val_losses)
    logger.info('best Loss: {:.6f} in epoch: {}'.format(best_loss, best_epoch))

#===================================Training of Supervised Model===================================#
def train_supervised(config):
    task='supervised'
    
    train_config = config.train_config
    paths = config.path_config
    model_config = config.model_config
    data_config = config.data_config
    optimizer_config = config.optimizer_config
    criterion_config = config.criterion_config
    
    epoch_start=0
    # print cfg
    print(colored('Info Config:', 'blue'))
    print(colored(config.info_config, 'yellow'))
    print(colored('Model Config:', 'blue'))
    print(colored(model_config, 'yellow'))
    print(colored('Train Config:', 'blue'))
    print(colored(train_config, 'yellow'))
    print(colored('Paths:', 'blue'))
    print(colored(paths, 'yellow'))
    print(colored('Data Config:', 'blue'))
    print(colored(data_config, 'yellow'))
    print(colored('Optimizer Config:', 'blue'))
    print(colored(optimizer_config, 'yellow'))
    print(colored('Criterion Config:', 'blue'))
    print(colored(criterion_config, 'yellow'))
    
    model_type = config.model_type
    # create and load model
    print(colored('Get model', 'blue'))
    supervisedmodel = getModel(config)
    last_model_file = os.path.join(config.path_config['{}_checkpoint'.format(task)],'{}_last_epoch.pth'.format(task+model_type))
    if os.path.exists(last_model_file):
        supervisedmodel, epoch_start = loadModel(net=supervisedmodel,
                                                 save_path=paths['{}_checkpoint'.format(task)],
                                                 file_class='last',
                                                 model_name=task+model_type, 
                                                 task=task)
        print(colored('Restart from checkpoint {}'.format(paths['{}_dir'.format(task)]), 'blue'))
    # create dataset
    dataset = getDataset(data_config)
    if os.path.exists(paths['train_index']) and os.path.exists(paths['val_index']):
        splited_out = splitDataset(dataset=dataset, cfg=config, use_pretrain_indexes=True)
    else:
        splited_out = splitDataset(dataset=dataset, cfg=config)
    train_loader, val_loader = splited_out['train_dataloader'], splited_out['val_dataloader']
    train_num, val_num = splited_out['train_num'], splited_out['val_num']
    print(colored('Get dataset and dataloaders', 'blue'))
    
    # loss
    print(colored('Get loss', 'blue'))
    criterion = get_criterion(criterion_config)
    criterion.cuda()
    # optimizer
    print(colored('Get optimizer', 'blue'))
    optimizer = get_optimizer(optimizer_config, supervisedmodel)
    # CUDNN
    torch.backends.cudnn.benchmark = True
    
    device = train_config['device']
    supervisedmodel = supervisedmodel.to(device=device)
    # create logger
    logger = get_logger(path=paths['{}_log'.format(task)], filename='{}Model_train.log'.format(task))
    
    # Main loop
    print(colored('Starting main loop', 'blue'))
    start_time = time.time()
    best_acc = 0
    best_epoch = 0
    
    stats_path = os.path.join(paths['{}_log'.format(task)], 'stats')
    check_dirs(stats_path)
    
    for epoch in range(epoch_start, epoch_start+train_config['epochs']):
        print(colored('Epoch %d/%d' %(epoch+1, train_config['epochs']), 'yellow'))
        print(colored('-'*15, 'yellow'))
        
        epoch_start_time = time.time()

        # Adjust lr
        lr = adjust_learning_rate(config, optimizer, epoch)
        print('Adjusted learning rate to {:.8e}'.format(lr))

        # Train
        print('Train ...')
        supervised_one_epoch(train_loader, supervisedmodel, criterion, optimizer, epoch, logger=logger)
        
        # Evaluate 
        print('Make prediction on training set ...')
        predictions = get_predictions(config, train_loader, supervisedmodel)

        clustering_stats = hungarian_evaluate(0, predictions, compute_confusion_matrix=False)
        print('acc:{:.8f}, ari:{:.8f}, nmi:{:.8f}'.format(clustering_stats['ACC'], clustering_stats['ARI'],clustering_stats['NMI']))
        print(clustering_stats['confusion_matrix'])
        train_info = 'Epoch: [{:>2d}/{:>2d}] || acc:{:.8f}, ari:{:.8f}, nmi:{:.8f}'.format(epoch, epoch_start + train_config['epochs'], clustering_stats['ACC'], clustering_stats['ARI'],clustering_stats['NMI'])
        logger.info(train_info)
        # Evaluate 
        print('Make prediction on validation set ...')
        predictions = get_predictions(config, val_loader, supervisedmodel)

        clustering_stats = hungarian_evaluate(0, predictions, compute_confusion_matrix=False)
        print('acc:{:.8f}, ari:{:.8f}, nmi:{:.8f}'.format(clustering_stats['ACC'], clustering_stats['ARI'],clustering_stats['NMI']))
        print(clustering_stats['confusion_matrix'])
        
        total_time=time.time() - start_time
        hours, mins, sec = time2hms(total_time)
        infomation = 'Epoch: [{:>2d}/{:>2d}] || Time: {:>3d} H {:>2d} M {:.3f} s || ACC: {:.8f} || ARI: {:.8f} || NMI: {:.8f} || epoch time: {:.3f} sec'.format(
                epoch, epoch_start + train_config['epochs'], hours, mins, sec, clustering_stats['ACC'], clustering_stats['ARI'], clustering_stats['NMI'], time.time()-epoch_start_time)
        
        logger.info(infomation)
        acc = clustering_stats['ACC']
        
        # Checkpoint
        print('Checkpoint ...')
        saveModel(net=supervisedmodel, save_path=paths['{}_checkpoint'.format(task)], epoch=epoch,
                  filename='{}_last_epoch.pth'.format(task+model_type), optimizer=optimizer, task=task)
        if (epoch+1)%train_config['save_model_epoch']==0:
            saveModel(net=supervisedmodel, save_path=paths['{}_checkpoint'.format(task)], epoch=epoch,
                      filename='{}_checkpoint-epoch{}.pth'.format(task+model_type, epoch), 
                      optimizer=optimizer, task=task)
            save_clustering_stats(clustering_stats=clustering_stats, path=stats_path, filename='stats_epoch{}'.format(epoch))
        if acc>=best_acc:
            saveModel(net=supervisedmodel, save_path=paths['{}_checkpoint'.format(task)], epoch=epoch,
                      filename='{}_best.pth'.format(task+model_type), optimizer=optimizer, task=task)
            save_clustering_stats(clustering_stats=clustering_stats, path=stats_path, filename='best_stats')
            best_acc = acc
            best_epoch = epoch
    
    # Evaluate and save the final model
    print(colored('Evaluate model at the end', 'blue'))
    
    last_model_file = os.path.join(config.path_config['{}_checkpoint'.format(task)],'{}_last_epoch.pth'.format(task+model_type))
    if os.path.exists(last_model_file):
        supervisedmodel, _ = loadModel(net=supervisedmodel,
                                       save_path=paths['{}_checkpoint'.format(task)],
                                       file_class='last',
                                       model_name=task+model_type, 
                                       task=task)
    
    predictions = get_predictions(config, val_loader, supervisedmodel)
    clustering_stats = hungarian_evaluate(0, predictions, 
                            class_names=data_config['class_name'], 
                            compute_confusion_matrix=True, 
                            confusion_matrix_file=os.path.join(paths['{}_confusion_matrix'.format(task)], 'confusion_matrix.png'))
    print(clustering_stats)
    save_clustering_stats(clustering_stats=clustering_stats, path=stats_path, filename='last_stats')
    
    bestinfo = 'best acc: {:.6f} in epoch {:>4d}'.format(best_acc, best_epoch)
    logger.info(bestinfo)
  
def supervised_one_epoch(train_loader, model, criterion, optimizer, epoch, logger, device=torch.device('cuda')):
    losses = AverageMeter('Loss', ':.8e')
    progress = ProgressMeter(len(train_loader), [losses],
                                prefix="Epoch: [{}]".format(epoch))
    
    model.train()
    
    for i, batch in enumerate(train_loader):
        nodes = ToTensor(batch['node']).to(device=device, non_blocking=True)
        label = ToTensor(batch['label']).to(device=device, non_blocking=True)

        pred = model(nodes)
        
        if type(pred)==list:
            pred=pred[0]
        #print(pred.shape, label.shape)
        loss = criterion(pred, label)
        losses.update(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        if i % 100 == 99 or i+1==len(train_loader):
            progress.display(i)
            infomation = 'Epoch: [{:>2d}] {:>4d}/{:>4d} items || Loss: {:.8e}'.format(epoch,i+1,len(train_loader), loss.item())
            logger.info(infomation)
            

#===================================Training of Scan Model===================================#
def train_scan(config):
    task = 'scan'
    
    train_config = config.train_config
    paths = config.path_config
    model_config = config.model_config
    data_config = config.data_config
    optimizer_config = config.optimizer_config
    criterion_config = config.criterion_config
    
    epoch_start=0
    # print cfg
    print(colored('Info Config:', 'blue'))
    print(colored(config.info_config, 'yellow'))
    print(colored('Model Config:', 'blue'))
    print(colored(model_config, 'yellow'))
    print(colored('Train Config:', 'blue'))
    print(colored(train_config, 'yellow'))
    print(colored('Paths:', 'blue'))
    print(colored(paths, 'yellow'))
    print(colored('Data Config:', 'blue'))
    print(colored(data_config, 'yellow'))
    print(colored('Optimizer Config:', 'blue'))
    print(colored(optimizer_config, 'yellow'))
    print(colored('Criterion Config:', 'blue'))
    print(colored(criterion_config, 'yellow'))
    # CUDNN
    torch.backends.cudnn.benchmark = True
    
    model_type = config.model_type
    pretext = config.pretext
    # create and load model
    print(colored('Get model', 'blue'))
    scanmodel = getModel(config)
    last_model_file = os.path.join(paths['{}_checkpoint'.format(task)],'{}_last_epoch.pth'.format(task+model_type))
    backbone_file = os.path.join(paths['{}_checkpoint'.format(pretext)], '{}_last_epoch.pth'.format(pretext+model_type))
    if os.path.exists(last_model_file):
        scanmodel, epoch_start = loadModel(net=scanmodel,
                                           save_path=paths['{}_checkpoint'.format(task)],
                                           file_class='last',
                                           model_name=task+model_type, 
                                           task=task)
        print(colored('Restart from checkpoint {}'.format(paths['{}_dir'.format(task)]), 'blue'))
    elif os.path.exists(backbone_file):
        scanmodel, _ = loadModel(net=scanmodel,
                                 save_path=paths['{}_checkpoint'.format(pretext)],
                                 file_class='last',
                                 model_name=pretext+model_type,
                                 task=task,
                                 load_backbone_only=True)
        print(colored('Load pre-train backbone from {} checkpoint {}'.format(pretext, paths['{}_dir'.format(pretext)]), 'blue'))
        print(colored('WARNING: SCAN will only update the cluster head', 'red'))
    else:
        print(colored('Train without pre-train backbone. ', 'red'))
    # to cuda
    device = train_config['device']
    scanmodel = scanmodel.to(device=device)
    
    # create logger
    logger = get_logger(path=paths['{}_log'.format(task)], filename='{}Model_train.log'.format(task))
    
    # loss
    print(colored('Get loss', 'blue'))
    criterion = get_criterion(criterion_config)
    criterion.cuda()
    # optimizer
    print(colored('Get optimizer', 'blue'))
    optimizer = get_optimizer(optimizer_config, scanmodel, nheads=model_config['nheads'], cluster_head_only=True)
    
    
    # create dataset
    dataset = getDataset(data_config)
    train_dataset, val_dataset = dataset['train_dataset'], dataset['val_dataset']
    train_num, val_num = dataset['train_num'], dataset['val_num']
    train_loader = getDataloader(cfg=train_config['train_loader'], dataset=train_dataset)
    val_loader = getDataloader(cfg=train_config['val_loader'], dataset=val_dataset)
    print(colored('Get dataset and dataloaders', 'blue'))
    
    # Main loop
    print(colored('Starting main loop', 'blue'))
    start_time = time.time()
    best_acc = 0
    best_epoch = 0
    
    stats_path = os.path.join(paths['{}_log'.format(task)], 'stats')
    check_dirs(stats_path)
    
    for epoch in range(epoch_start, epoch_start+train_config['epochs']):
        print(colored('Epoch %d/%d' %(epoch+1, train_config['epochs']), 'yellow'))
        print(colored('-'*15, 'yellow'))
        
        epoch_start_time = time.time()
        # Adjust lr
        lr = adjust_learning_rate(config, optimizer, epoch)
        print('Adjusted learning rate to {:.8e}'.format(lr))
        
        # Train
        print('Train ...')
        scan_one_epoch(train_loader=train_loader,
                       model=scanmodel,
                       criterion=criterion,
                       optimizer=optimizer,
                       epoch=epoch,
                       logger=logger,
                       device=device,
                       update_cluster_head_only=train_config['update_cluster_head_only'])
        
        # Evaluate 
        print('Make prediction on training set ...')
        train_prediction = get_predictions(config, train_loader, scanmodel)
        print('Get Train SCAN loss ...')
        scan_train_stats = scan_evaluate(train_prediction)
        print(scan_train_stats)
        lowest_train_loss = scan_train_stats['lowest_loss']
        lowest_train_loss_head = scan_train_stats['lowest_loss_head']
        
        train_clustering_stats = hungarian_evaluate(lowest_train_loss_head, train_prediction, compute_confusion_matrix=False)
        print('the lowest train loss is {:.8e}'.format(lowest_train_loss))
        print('acc:{:.8f}, ari:{:.8f}, nmi:{:.8f}'.format(train_clustering_stats['ACC'], train_clustering_stats['ARI'],train_clustering_stats['NMI']))
        print(train_clustering_stats['confusion_matrix'])
        
        train_info = 'Epoch: [{:>2d}/{:>2d}] || acc:{:.8f}, ari:{:.8f}, nmi:{:.8f}, lowest train loss: {:.8e}'.format(epoch+1, epoch_start + train_config['epochs'], train_clustering_stats['ACC'], train_clustering_stats['ARI'],train_clustering_stats['NMI'], lowest_train_loss)
        logger.info(train_info)
        
        # Evaluate 
        print('Make prediction on validation set ...')
        val_prediction = get_predictions(config, val_loader, scanmodel)
        
        print('Get Val SCAN loss ...')
        scan_val_stats = scan_evaluate(val_prediction)
        print(scan_val_stats)
        lowest_val_loss = scan_val_stats['lowest_loss']
        lowest_val_loss_head = scan_val_stats['lowest_loss_head']
        
        val_clustering_stats = hungarian_evaluate(lowest_val_loss_head, val_prediction, compute_confusion_matrix=False)
        print('the lowest val loss is {:.8e}'.format(lowest_val_loss))
        print('acc:{:.8f}, ari:{:.8f}, nmi:{:.8f}'.format(val_clustering_stats['ACC'], val_clustering_stats['ARI'],val_clustering_stats['NMI']))
        print(val_clustering_stats['confusion_matrix'])
        
        total_time=time.time() - start_time
        hours, mins, sec = time2hms(total_time)
        val_info = 'Epoch: [{:>2d}/{:>2d}] || Time: {:>3d} H {:>2d} M {:.3f} s || ACC: {:.8f} || ARI: {:.8f} || NMI: {:.8f} || epoch time: {:.3f} sec'.format(
                epoch, epoch_start + train_config['epochs'], hours, mins, sec, val_clustering_stats['ACC'], val_clustering_stats['ARI'], val_clustering_stats['NMI'], time.time()-epoch_start_time)
        
        logger.info(val_info)
        acc = val_clustering_stats['ACC']
        
        print('Checkpoint ...')
        saveModel(net=scanmodel, save_path=paths['{}_checkpoint'.format(task)], epoch=epoch,
                  filename='{}_last_epoch.pth'.format(task+model_type), optimizer=optimizer, task=task)
        
        if (epoch+1)%train_config['save_model_epoch']==0:
            saveModel(net=scanmodel, save_path=paths['{}_checkpoint'.format(task)], epoch=epoch,
                      filename='{}_checkpoint-epoch{}.pth'.format(task+model_type, epoch), 
                      optimizer=optimizer, task=task)
            save_clustering_stats(clustering_stats=val_clustering_stats, path=stats_path, filename='stats_epoch{}'.format(epoch+1))
        
        if acc>=best_acc:
            saveModel(net=scanmodel, save_path=paths['{}_checkpoint'.format(task)], epoch=epoch,
                      filename='{}_best.pth'.format(task+model_type), optimizer=optimizer, task=task)
            save_clustering_stats(clustering_stats=val_clustering_stats, path=stats_path, filename='best_stats')
            best_acc = acc
            best_epoch = epoch
    
    # Evaluate and save the final model
    print(colored('Evaluate model at the end', 'blue'))
    
    best_model_file = os.path.join(paths['{}_checkpoint'.format(task)],'{}_last_epoch.pth'.format(task+model_type))
    if os.path.exists(best_model_file):
        scanmodel, _ = loadModel(net=scanmodel,
                                 save_path=paths['{}_checkpoint'.format(task)],
                                 file_class='last',
                                 model_name=task+model_type, 
                                 task=task)
    
    predictions = get_predictions(config, val_loader, scanmodel)
    clustering_stats = hungarian_evaluate(0, predictions, 
                            class_names=data_config['class_name'], 
                            compute_confusion_matrix=True, 
                            confusion_matrix_file=os.path.join(paths['{}_confusion_matrix'.format(task)], 'confusion_matrix.png'))
    print(clustering_stats)
    save_clustering_stats(clustering_stats=clustering_stats, path=stats_path, filename='last_stats')
    
    bestinfo = 'best acc: {:.6f} in epoch {:>4d}'.format(best_acc, best_epoch)
    logger.info(bestinfo)

    
def scan_one_epoch(train_loader, model, criterion, optimizer, epoch, logger, device=torch.device('cuda'), update_cluster_head_only=True):
    """ 
    Train w/ SCAN-Loss
    """
    total_losses = AverageMeter('Total Loss', ':.6e')
    consistency_losses = AverageMeter('Consistency Loss', ':.6e')
    entropy_losses = AverageMeter('Entropy', ':.6e')
    progress = ProgressMeter(len(train_loader),
        [total_losses, consistency_losses, entropy_losses],
        prefix="Epoch: [{}]".format(epoch))

    if update_cluster_head_only:
        model.eval() # No need to update BN
    else:
        model.train() # Update BN

    for i, batch in enumerate(train_loader):
        # Forward pass
        anchors = ToTensor(batch['anchor']).to(device=device, non_blocking=True)
        neighbors = ToTensor(batch['neighbor']).to(device=device, non_blocking=True)
       
        if update_cluster_head_only: # Only calculate gradient for backprop of linear layer
            with torch.no_grad():
                anchors_features = model(anchors, forward_pass='backbone')
                neighbors_features = model(neighbors, forward_pass='backbone')
            anchors_output = model(anchors_features, forward_pass='head')
            neighbors_output = model(neighbors_features, forward_pass='head')

        else: # Calculate gradient for backprop of complete network
            anchors_output = model(anchors)
            neighbors_output = model(neighbors)     

        # Loss for every head
        total_loss, consistency_loss, entropy_loss = [], [], []
        for anchors_output_subhead, neighbors_output_subhead in zip(anchors_output, neighbors_output):
            total_loss_, consistency_loss_, entropy_loss_ = criterion(anchors_output_subhead,
                                                                         neighbors_output_subhead)
            total_loss.append(total_loss_)
            consistency_loss.append(consistency_loss_)
            entropy_loss.append(entropy_loss_)

        # Register the mean loss and backprop the total loss to cover all subheads
        total_losses.update(np.mean([v.item() for v in total_loss]))
        consistency_losses.update(np.mean([v.item() for v in consistency_loss]))
        entropy_losses.update(np.mean([v.item() for v in entropy_loss]))

        total_loss = torch.sum(torch.stack(total_loss, dim=0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 100 == 99 or i+1==len(train_loader):
            progress.display(i+1)
            infomation = 'Epoch: [{:>2d}] {:>4d}/{:>4d} items || Loss: {:.8e}'.format(epoch,i+1,len(train_loader), total_loss.item())
            logger.info(infomation)

#===================================Training of Spice Model===================================#
def train_spice(config):
    task = 'spice'
    
    train_config = config.train_config
    paths = config.path_config
    model_config = config.model_config
    data_config = config.data_config
    optimizer_config = config.optimizer_config
    criterion_config = config.criterion_config
    
    epoch_start=0
    # print cfg
    print(colored('Info Config:', 'blue'))
    print(colored(config.info_config, 'yellow'))
    print(colored('Model Config:', 'blue'))
    print(colored(model_config, 'yellow'))
    print(colored('Train Config:', 'blue'))
    print(colored(train_config, 'yellow'))
    print(colored('Paths:', 'blue'))
    print(colored(paths, 'yellow'))
    print(colored('Data Config:', 'blue'))
    print(colored(data_config, 'yellow'))
    print(colored('Optimizer Config:', 'blue'))
    print(colored(optimizer_config, 'yellow'))
    print(colored('Criterion Config:', 'blue'))
    print(colored(criterion_config, 'yellow'))
    
    # CUDNN
    torch.backends.cudnn.benchmark = True
    
    model_type = config.model_type
    pretext = config.pretext
    # create and load model
    print(colored('Get model', 'blue'))
    spicemodel = getModel(config)
    last_model_file = os.path.join(paths['{}_checkpoint'.format(task)],'{}_last_epoch.pth'.format(task+model_type))
    backbone_file = os.path.join(paths['{}_checkpoint'.format(pretext)], '{}_last_epoch.pth'.format(pretext+model_type))
    if os.path.exists(last_model_file):
        spicemodel, epoch_start = loadModel(net=spicemodel,
                                            save_path=paths['{}_checkpoint'.format(task)],
                                            file_class='last',
                                            model_name=task+model_type, 
                                            task=task)
        print(colored('Restart from checkpoint {}'.format(paths['{}_dir'.format(task)]), 'blue'))
    elif os.path.exists(backbone_file):
        spicemodel, _ = loadModel(net=spicemodel,
                                  save_path=paths['{}_checkpoint'.format(pretext)],
                                  file_class='last',
                                  model_name=pretext+model_type,
                                  task=task,
                                  load_backbone_only=True)
        print(colored('Load pre-train backbone from {} checkpoint {}'.format(pretext, paths['{}_dir'.format(pretext)]), 'blue'))
        #print(colored('WARNING: DeepCl will only update the cluster head', 'red'))
    else:
        print(colored('Train without pre-train backbone. ', 'red'))
    
    # to cuda
    device = train_config['device']
    spicemodel = spicemodel.to(device=device)
    
    # create logger
    logger = get_logger(path=paths['{}_log'.format(task)], filename='{}Model_train.log'.format(task))
    
    # loss
    print(colored('Get loss', 'blue'))
    criterion = get_criterion(criterion_config)
    criterion.cuda()
    # optimizer
    print(colored('Get optimizer', 'blue'))
    optimizer = get_optimizer(optimizer_config, spicemodel, nheads=model_config['nheads'], cluster_head_only=True)
    
    # create dataset
    dataset = getDataset(data_config)
    if os.path.exists(paths['train_index']) and os.path.exists(paths['val_index']):
        splited_out = splitDataset(dataset=dataset, cfg=config, use_pretrain_indexes=True)
    else:
        splited_out = splitDataset(dataset=dataset, cfg=config)
    train_loader, val_loader = splited_out['train_dataloader'], splited_out['val_dataloader']
    train_num, val_num = splited_out['train_num'], splited_out['val_num']
    train_indices, val_indices = splited_out['train_indices'], splited_out['val_indices']
    print(colored('Get dataset and dataloaders', 'blue'))
    
    # Main loop
    print(colored('Starting main loop', 'blue'))
    start_time = time.time()
    best_acc = 0
    best_epoch = 0
    
    stats_path = os.path.join(paths['{}_log'.format(task)], 'stats')
    check_dirs(stats_path)
    
    for epoch in range(epoch_start, epoch_start+train_config['epochs']):
        print(colored('Epoch %d/%d' %(epoch+1, train_config['epochs']), 'yellow'))
        print(colored('-'*15, 'yellow'))
        
        epoch_start_time = time.time()

        # Adjust lr
        lr = adjust_learning_rate(config, optimizer, epoch)
        print('Adjusted learning rate to {:.8e}'.format(lr))
        
        # Train
        print('Train ...')
        spice_one_epoch(dataset=dataset, train_indices=train_indices, model=spicemodel, criterion=criterion, 
                        optimizer=optimizer, device=device, epoch=epoch, logger=logger, num_repeat=4, nhead=1, 
                        batch_size=train_config['train_loader']['BatchSize'], device=device,
                        update_cluster_head_only=train_config['update_cluster_head_only'])
        
        # Evaluate 
        print('Make prediction on training set ...')
        predictions = get_predictions(config, train_loader, spicemodel)

        clustering_stats = hungarian_evaluate(0, predictions, compute_confusion_matrix=False)
        print('acc:{:.8f}, ari:{:.8f}, nmi:{:.8f}'.format(clustering_stats['ACC'], clustering_stats['ARI'],clustering_stats['NMI']))
        print(clustering_stats['confusion_matrix'])
        train_info = 'Epoch: [{:>2d}/{:>2d}] || acc:{:.8f}, ari:{:.8f}, nmi:{:.8f}'.format(epoch, epoch_start + train_config['epochs'], clustering_stats['ACC'], clustering_stats['ARI'],clustering_stats['NMI'])
        logger.info(train_info)
        # Evaluate 
        print('Make prediction on validation set ...')
        predictions = get_predictions(config, val_loader, spicemodel)

        clustering_stats = hungarian_evaluate(0, predictions, compute_confusion_matrix=False)
        print('acc:{:.8f}, ari:{:.8f}, nmi:{:.8f}'.format(clustering_stats['ACC'], clustering_stats['ARI'],clustering_stats['NMI']))
        print(clustering_stats['confusion_matrix'])
        
        total_time=time.time() - start_time
        hours, mins, sec = time2hms(total_time)
        infomation = 'Epoch: [{:>2d}/{:>2d}] || Time: {:>3d} H {:>2d} M {:.3f} s || ACC: {:.8f} || ARI: {:.8f} || NMI: {:.8f} || epoch time: {:.3f} sec'.format(
                epoch, epoch_start + train_config['epochs'], hours, mins, sec, clustering_stats['ACC'], clustering_stats['ARI'], clustering_stats['NMI'], time.time()-epoch_start_time)
        
        logger.info(infomation)
        acc = clustering_stats['ACC']
        
        # Checkpoint
        print('Checkpoint ...')
        saveModel(net=spicemodel, save_path=paths['{}_checkpoint'.format(task)], epoch=epoch,
                  filename='{}_last_epoch.pth'.format(task+model_type), optimizer=optimizer, task=task)
        if (epoch+1)%train_config['save_model_epoch']==0:
            saveModel(net=spicemodel, save_path=paths['{}_checkpoint'.format(task)], epoch=epoch,
                      filename='{}_checkpoint-epoch{}.pth'.format(task+model_type, epoch), 
                      optimizer=optimizer, task=task)
            save_clustering_stats(clustering_stats=clustering_stats, path=stats_path, filename='stats_epoch{}'.format(epoch))
        if acc>=best_acc:
            saveModel(net=spicemodel, save_path=paths['{}_checkpoint'.format(task)], epoch=epoch,
                      filename='{}_best.pth'.format(task+model_type), optimizer=optimizer, task=task)
            save_clustering_stats(clustering_stats=clustering_stats, path=stats_path, filename='best_stats')
            best_acc = acc
            best_epoch = epoch
    
    # Evaluate and save the final model
    print(colored('Evaluate model at the end', 'blue'))
    
    last_model_file = os.path.join(config.path_config['{}_checkpoint'.format(task)],'{}_last_epoch.pth'.format(task+model_type))
    if os.path.exists(last_model_file):
        spicemodel, _ = loadModel(net=spicemodel,
                                  save_path=paths['{}_checkpoint'.format(task)],
                                  file_class='last',
                                  model_name=task+model_type, 
                                  task=task)
    
    predictions = get_predictions(config, val_loader, spicemodel)
    clustering_stats = hungarian_evaluate(0, predictions, 
                            class_names=data_config['class_name'], 
                            compute_confusion_matrix=True, 
                            confusion_matrix_file=os.path.join(paths['{}_confusion_matrix'.format(task)], 'confusion_matrix.png'))
    print(clustering_stats)
    save_clustering_stats(clustering_stats=clustering_stats, path=stats_path, filename='last_stats')
    
    bestinfo = 'best acc: {:.6f} in epoch {:>4d}'.format(best_acc, best_epoch)
    logger.info(bestinfo)
        
def spice_one_epoch(dataset, train_indices, model, criterion, optimizer, 
                    device=torch.device('cuda'), epoch:int, logger, num_repeat:int=2, nhead:int=1, 
                    batch_size:int=8, update_cluster_head_only=False):
    # get pseudo label
    scores, features = getScoreAndFeature(dataset=dataset, indices=train_indices, model=model, 
                                          device=device, nhead=nhead, batch_size=batch_size)
    b, ncls = scores[0].size()
    _, dim = features.size()
    num_per_cluster = b//ncls
    # define center num and pseudo num
    center_ratio = 0.5
    select_ratio = 0.25
    center_k = int(num_per_cluster*select_ratio*center_ratio)
    pseudo_k = int(num_per_cluster*select_ratio)
    
    scores=scores[0] 
    centers = getPseudoCenter(scores=scores, features=features, k=center_k)
    pseudo_labs, pseudo_indices = getPseudoLabel(centers=centers, features=features, k=pseudo_k, train_indices=train_indices)
    pn, _ = pseudo_labs.size()
    left_num = pn%batch_size
    pseudo_labs = pseudo_labs[:-left_num]
    pseudo_indices = pseudo_indices[:-left_num]

    # prepare for train    
    losses = AverageMeter('Loss', ':.6e')
    num_pseudo = len(pseudo_indices)
    backward_num = num_repeat*ceil(num_pseudo//batch_size)
    progress = ProgressMeter(backward_num,
        [losses],
        prefix="Epoch: [{}]".format(epoch))
    # train
    model.train()
    back_i = 1
    for _ in range(num_repeat):
        shuffle_indices = list(range(num_pseudo))
        np.random.shuffle(shuffle_indices)
        batch = []
        batch_lab = []
        iter_num = 1
        for i in range(num_pseudo):
            truth_psei = pseudo_indices[shuffle_indices[i]]
            pse_lab = pseudo_labs[shuffle_indices[i]]
            pse_lab = torch.unsqueeze(pse_lab, dim=0)
            pse_lab = pse_lab.to(device=device, non_blocking=True)
            
            item = dataset.__getitem__(truth_psei)
            node = ToTensor(item['strong_trans']).to(device=device, non_blocking=True)
            node = torch.unsqueeze(node, dim=0)
            
            batch.append(node)
            batch_lab.append(pse_lab)
            
            if iter_num%batch_size==0 or iter_num==num_pseudo:
                batch = torch.cat(batch, dim=0)
                batch_lab = torch.cat(batch_lab, dim=0)
                #print(batch.size())
                if update_cluster_head_only:
                    with torch.no_grad():
                        fea = model(batch, forward_pass='backbone')
                    prob = model(fea, forward_pass='head')
                else:
                    prob = model(batch)
                prob = prob[0]
                loss = criterion(prob, batch_lab)
                losses.update(loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if back_i % 100 == 0 or back_i==backward_num:
                    progress.display(back_i)
                    infomation = 'Epoch: [{:>2d}] {:>4d}/{:>4d} items || Loss: {:.8e}'.format(epoch,back_i,backward_num, loss.item())
                    logger.info(infomation)
            
                batch=[]
                batch_lab=[]
                back_i+=1
            iter_num+=1
    print('Finish epoch {}'.format(epoch))
                
def getScoreAndFeature(dataset, indices, model, device=torch.device('cuda'), nhead:int=1, batch_size:int=8):
    # dataset: torch.utils.data.Dataset, data_loader.subdataset.SpiceDataset
    # indices: list or numpy.array, [train_num]
    # model: model.ClusteringModel.ClusteringModel
    # device: torch.device('cuda') or torch.device('cpu')
    # nhead: int
    # batch_size: int
    # return:
    #       scores: list [torch.Tensor [b, ncls], ...] len(scores)=nhead
    #       features: torch.Tensor [b, dim]
    model.eval()   
    num = 1
    #keys=['node', 'weak_trans', 'strong_trans']
    keys=['node', 'weak_trans']
    batch = [[] for _ in range(len(keys))]
    
    scores = [[] for _ in range(nhead)]
    features = []
    
    for i in indices:
        item = dataset.__getitem__(i)
        for j in range(len(keys)):
            node = item[keys[j]]
            node = ToTensor(node).to(device=device, non_blocking=True)
            node = torch.unsqueeze(node, dim=0)
            batch[j].append(node)
        if num%batch_size==0 or num==len(indices):
            for k in range(len(keys)):
                batch[k] = torch.cat(batch[k], dim=0)
            with torch.no_grad():
                sc = model(batch[0])
                fea = model(batch[1], forward_pass='backbone')
            features.append(fea)
            for isc in range(len(scores)):
                scores[isc].append(sc[isc])
            batch = [[] for _ in range(len(keys))]
            
        num+=1
    features = torch.cat(features,dim=0)
    for i in range(len(scores)):
        scores[i] = torch.cat(scores[i], dim=0)
    
    return scores, features

def getPseudoCenter(scores, features, k):
    b, ncls = scores.size()
    if k>=b:
        k=b//2
    _, topki = torch.topk(scores, k=k, dim=0)
    centers = []
    for i in range(ncls):
        c = torch.mean(features[topki[:,i]], dim=0).unsqueeze(dim=0)
        centers.append(c)
    centers = torch.cat(centers, dim=0)
    
    return centers

def getPseudoLabel(centers, features, train_indices, k:int):
    # centers: torch.Tensor [ncls, dim]
    # features: torch.Tensor [b, dim]
    # return:
    #    pseudo label: torch.Tensor [n, ncls]
    #    pseudo indices: np.array [n]
    b, _ = features.size()
    ncls, _ = centers.size()
    
    cls_center = torch.unsqueeze(centers, dim=0) # [1, ncls, dim]
    unsq_fea = torch.unsqueeze(features, dim=1) # [b, 1, dim]
    dis = torch.sqrt(torch.sum((unsq_fea-cls_center)**2, dim=-1)) # [b, ncls]
    _, topki = torch.topk(dis, k=k, dim=0, largest=False)
    
    pseudo_lab = torch.zeros((b,ncls))
    for i in range(ncls):
        pseudo_lab[topki[:,i],i]=1
    sum_pseudo = torch.sum(pseudo_lab, dim=-1)
    new_indices = ToNumpy(train_indices)[sum_pseudo!=0]
    pseudo_lab = pseudo_lab[sum_pseudo!=0]
    
    return pseudo_lab, new_indices

#===================================Training of KMeans Model===================================#
def train_kmeans():
    pass

#===================================Training of Simclr Model===================================#
def train_simclr(config):
    task='simclr'
    
    train_config = config.train_config
    paths = config.path_config
    model_config = config.model_config
    data_config = config.data_config
    optimizer_config = config.optimizer_config
    criterion_config = config.criterion_config
    
    epoch_start=0
    # print cfg
    print(colored('Info Config:', 'blue'))
    print(colored(config.info_config, 'yellow'))
    print(colored('Model Config:', 'blue'))
    print(colored(model_config, 'yellow'))
    print(colored('Train Config:', 'blue'))
    print(colored(train_config, 'yellow'))
    print(colored('Paths:', 'blue'))
    print(colored(paths, 'yellow'))
    print(colored('Data Config:', 'blue'))
    print(colored(data_config, 'yellow'))
    print(colored('Optimizer Config:', 'blue'))
    print(colored(optimizer_config, 'yellow'))
    print(colored('Criterion Config:', 'blue'))
    print(colored(criterion_config, 'yellow'))
    
    model_type = config.model_type
    point2img = config.point2img
    # create and load model
    print(colored('Get model', 'blue'))
    simclrmodel = getModel(config)
    last_model_file = os.path.join(config.path_config['{}_checkpoint'.format(task)],'{}_last_epoch.pth'.format(task+model_type))
    if os.path.exists(last_model_file):
        simclrmodel, epoch_start = loadModel(net=simclrmodel,
                                             save_path=paths['{}_checkpoint'.format(task)],
                                             file_class='last',
                                             model_name=task+model_type, 
                                             task=task)
        print(colored('Restart from checkpoint {}'.format(paths['{}_dir'.format(task)]), 'blue'))
    # create dataset
    dataset = getDataset(data_config)
    if os.path.exists(paths['train_index']) and os.path.exists(paths['val_index']):
        splited_out = splitDataset(dataset=dataset, cfg=config, use_pretrain_indexes=True)
    else:
        splited_out = splitDataset(dataset=dataset, cfg=config)
    train_loader, val_loader = splited_out['train_dataloader'], splited_out['val_dataloader']
    train_num, val_num = splited_out['train_num'], splited_out['val_num']
    print(colored('Get dataset and dataloaders', 'blue'))
    
    # Memory Bank
    print(colored('Build MemoryBank', 'blue'))
    
    memory_bank_base = MemoryBank(train_num, 
                                model_config['contrastive_feadim'],
                                data_config['n_class'], criterion_config['temperature'])
    memory_bank_base.cpu()
    memory_bank_val = MemoryBank(val_num,
                                model_config['contrastive_feadim'],
                                data_config['n_class'], criterion_config['temperature'])
    memory_bank_val.cpu()
    
    # loss
    print(colored('Get loss', 'blue'))
    criterion = get_criterion(criterion_config)
    print('Criterion is {}'.format(criterion.__class__.__name__))
    criterion = criterion.to(device=train_config['device'])
    
    # optimizer
    print(colored('Get optimizer', 'blue'))
    optimizer = get_optimizer(optimizer_config, simclrmodel)
    # CUDNN
    torch.backends.cudnn.benchmark = True
    
    device = train_config['device']
    simclrmodel = simclrmodel.to(device=device)
    
    # create logger
    logger = get_logger(path=paths['{}_log'.format(task)], filename='{}Model_train.log'.format(task))
    
    # Main loop
    print(colored('Starting main loop', 'blue'))
    start_time = time.time()
    best_acc = 0
    best_epoch = 0
    
    stats_path = os.path.join(paths['{}_log'.format(task)], 'stats')
    check_dirs(stats_path)
    
    for epoch in range(epoch_start, epoch_start+train_config['epochs']):
        print(colored('Epoch %d/%d' %(epoch+1, train_config['epochs']), 'yellow'))
        print(colored('-'*15, 'yellow'))
        
        epoch_start_time = time.time()

        # Adjust lr
        lr = adjust_learning_rate(config, optimizer, epoch)
        print('Adjusted learning rate to {:.8e}'.format(lr))
        
        # Train
        print('Train ...')
        simclr_one_epoch(train_loader=train_loader, model=simclrmodel, criterion=criterion, optimizer=optimizer, device=device, epoch=epoch, logger=logger, point2img=point2img)
        
        # Fill memory bank
        print('Fill memory bank for kNN...')
        fill_memory_bank(train_loader, simclrmodel, memory_bank_base)
        
        # Evaluate (To monitor progress - Not for validation)
        print('Evaluate ...')
        top1 = contrastive_evaluate(val_loader, simclrmodel, memory_bank_base)
        print('Result of kNN evaluation is %.4f' %(top1))
        total_time=time.time() - start_time
        hours, mins, sec = time2hms(total_time)
        val_info = 'Epoch: [{:>2d}/{:>2d}] || Time: {:>3d} H {:>2d} M {:.3f} s || Result of kNN acc is :{:.8f} || epoch time: {:.3f} sec'.format(epoch, epoch_start + train_config['epochs'], hours, mins, sec,  top1, time.time()-epoch_start_time)
        logger.info(val_info)
        
        # Checkpoint
        print('Checkpoint ...')
        saveModel(net=simclrmodel, save_path=paths['{}_checkpoint'.format(task)], epoch=epoch,
                  filename='{}_last_epoch.pth'.format(task+model_type), optimizer=optimizer, task=task)
        
        if (epoch+1)%train_config['save_model_epoch']==0:
            saveModel(net=simclrmodel, save_path=paths['{}_checkpoint'.format(task)], epoch=epoch,
                      filename='{}_checkpoint-epoch{}.pth'.format(task+model_type, epoch), 
                      optimizer=optimizer, task=task)
        
        if top1 >=best_acc:
            best_acc = top1
            best_epoch = epoch
            saveModel(net=simclrmodel, save_path=paths['{}_checkpoint'.format(task)], epoch=epoch,
                      filename='{}_best.pth'.format(task+model_type), optimizer=optimizer, task=task)
    
    # Mine the topk nearest neighbors at the very end (Train) 
    # These will be served as input to the SCAN loss.
    print(colored('Fill memory bank for mining the nearest neighbors (train) ...', 'blue'))
    fill_memory_bank(train_loader, simclrmodel, memory_bank_base)
    topk = data_config['num_neighbors']
    print('Mine the nearest neighbors (Top-%d)' %(topk)) 
    indices, acc = memory_bank_base.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on train set is %.2f' %(topk, 100*acc))
    np.save(paths['topk_neighbors_train_path'], indices)
    final_train_info = 'Accuracy of top-{} nearest neighbors on train set is {:.2f}' .format(topk, 100*acc)
    logger.info(final_train_info)
    
    # Mine the topk nearest neighbors at the very end (Val)
    # These will be used for validation.
    print(colored('Fill memory bank for mining the nearest neighbors (val) ...', 'blue'))
    fill_memory_bank(val_loader, simclrmodel, memory_bank_val)
    topk = 5
    print('Mine the nearest neighbors (Top-%d)' %(topk)) 
    indices, acc = memory_bank_val.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on val set is %.2f' %(topk, 100*acc))
    np.save(paths['topk_neighbors_val_path'], indices)
    final_val_info = 'Accuracy of top-{} nearest neighbors on val set is {:.2f}' .format(topk, 100*acc)
    logger.info(final_val_info)
    
    best_info = 'best acc:{:.6f} in epoch {:>4d}'.format(best_acc, best_epoch)
    logger.info(best_info)
    
def simclr_one_epoch(train_loader, model, criterion, optimizer, epoch, logger, device=torch.device('cuda'), point2img=False):
    losses = AverageMeter('Loss', ':.6e')
    progress = ProgressMeter(len(train_loader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    
    for i, batch in enumerate(train_loader):
        nodes = batch['node']
        nodes_augmented = batch['node_augmented']
        
        if point2img:
            b, c, h, w = nodes.shape
        else:
            b, c, l = nodes.shape
            
        if type(nodes)==np.ndarray:
            nodes = torch.from_numpy(nodes).to(device=device, non_blocking=True)
            nodes_augmented = torch.from_numpy(nodes_augmented).to(device=device, non_blocking=True)
        input_ = torch.cat([nodes.unsqueeze(1), nodes_augmented.unsqueeze(1)], dim=1)
        
        if point2img:
            input_ = input_.view(-1, c, h, w)
        else:
            input_ = input_.view(-1, c, l)
            
        input_ = input_.to(device=device, non_blocking=True)
        #targets = batch['label'].cuda(non_blocking=True)

        output = model(input_).view(b, 2, -1)
        loss = criterion(output)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 25 == 0 or i+1==len(train_loader):
            progress.display(i)
            infomation = 'Epoch: [{:>2d}] {:>4d}/{:>4d} items || Loss: {:.8e}'.format(epoch,i+1,len(train_loader), loss.item())
            logger.info(infomation)

#===================================Training of BYOL Model===================================#
def train_byol(config):
    task='byol'
    
    train_config = config.train_config
    paths = config.path_config
    model_config = config.model_config
    data_config = config.data_config
    optimizer_config = config.optimizer_config
    criterion_config = config.criterion_config
    
    epoch_start=0
    # print cfg
    print(colored('Info Config:', 'blue'))
    print(colored(config.info_config, 'yellow'))
    print(colored('Model Config:', 'blue'))
    print(colored(model_config, 'yellow'))
    print(colored('Train Config:', 'blue'))
    print(colored(train_config, 'yellow'))
    print(colored('Paths:', 'blue'))
    print(colored(paths, 'yellow'))
    print(colored('Data Config:', 'blue'))
    print(colored(data_config, 'yellow'))
    print(colored('Optimizer Config:', 'blue'))
    print(colored(optimizer_config, 'yellow'))
    print(colored('Criterion Config:', 'blue'))
    print(colored(criterion_config, 'yellow'))
    
    model_type = config.model_type
    point2img = config.point2img
    # create and load model
    print(colored('Get model', 'blue'))
    byolmodel = getModel(config)
    last_model_file = os.path.join(config.path_config['{}_checkpoint'.format(task)],'{}_last_epoch.pth'.format(task+model_type))
    if os.path.exists(last_model_file):
        byolmodel, epoch_start = loadModel(net=byolmodel,
                                           save_path=paths['{}_checkpoint'.format(task)],
                                           file_class='last',
                                           model_name=task+model_type, 
                                           task=task)
        print(colored('Restart from checkpoint {}'.format(paths['{}_dir'.format(task)]), 'blue'))
    # create dataset
    dataset = getDataset(data_config)
    if os.path.exists(paths['train_index']) and os.path.exists(paths['val_index']):
        splited_out = splitDataset(dataset=dataset, cfg=config, use_pretrain_indexes=True)
    else:
        splited_out = splitDataset(dataset=dataset, cfg=config)
    train_loader, val_loader = splited_out['train_dataloader'], splited_out['val_dataloader']
    train_num, val_num = splited_out['train_num'], splited_out['val_num']
    print(colored('Get dataset and dataloaders', 'blue'))
    
    # Memory Bank
    print(colored('Build MemoryBank', 'blue'))
    
    memory_bank_base = MemoryBank(train_num, 
                                model_config['contrastive_feadim'],
                                data_config['n_class'], criterion_config['temperature'])
    memory_bank_base.cpu()
    memory_bank_val = MemoryBank(val_num,
                                model_config['contrastive_feadim'],
                                data_config['n_class'], criterion_config['temperature'])
    memory_bank_val.cpu()
    
    # loss
    print(colored('Get loss', 'blue'))
    criterion = get_criterion(criterion_config)
    print('Criterion is {}'.format(criterion.__class__.__name__))
    criterion = criterion.to(device=train_config['device'])
    
    # optimizer
    print(colored('Get optimizer', 'blue'))
    optimizer = get_optimizer(optimizer_config, byolmodel)
    # CUDNN
    torch.backends.cudnn.benchmark = True
    
    target_model = copy.deepcopy(byolmodel)
    set_requires_grad(target_model, False)
    
    device = train_config['device']
    byolmodel = byolmodel.to(device=device)
    target_model = target_model.to(device=device)
    
    # create logger
    logger = get_logger(path=paths['{}_log'.format(task)], filename='{}Model_train.log'.format(task))
    
    # Main loop
    print(colored('Starting main loop', 'blue'))
    start_time = time.time()
    best_acc = 0
    best_epoch = 0
    
    stats_path = os.path.join(paths['{}_log'.format(task)], 'stats')
    check_dirs(stats_path)
    
    for epoch in range(epoch_start, epoch_start+train_config['epochs']):
        print(colored('Epoch %d/%d' %(epoch+1, train_config['epochs']), 'yellow'))
        print(colored('-'*15, 'yellow'))
        
        epoch_start_time = time.time()

        # Adjust lr
        lr = adjust_learning_rate(config, optimizer, epoch)
        print('Adjusted learning rate to {:.8e}'.format(lr))
        
        # Train
        print('Train ...')
        byolmodel, target_model = byol_one_epoch(train_loader=train_loader, model=byolmodel, 
                                                 target_model=target_model, criterion=criterion,
                                                 device=device, optimizer=optimizer, epoch=epoch, 
                                                 logger=logger, point2img=point2img)
        
        # Fill memory bank
        print('Fill memory bank for kNN...')
        fill_memory_bank(train_loader, byolmodel, memory_bank_base)
        
        # Evaluate (To monitor progress - Not for validation)
        print('Evaluate ...')
        top1 = contrastive_evaluate(val_loader, byolmodel, memory_bank_base)
        print('Result of kNN evaluation is %.4f' %(top1))
        total_time=time.time() - start_time
        hours, mins, sec = time2hms(total_time)
        val_info = 'Epoch: [{:>2d}/{:>2d}] || Time: {:>3d} H {:>2d} M {:.3f} s || Result of kNN acc is :{:.8f} || epoch time: {:.3f} sec'.format(epoch, epoch_start + train_config['epochs'], hours, mins, sec,  top1, time.time()-epoch_start_time)
        logger.info(val_info)
        
        # Checkpoint
        print('Checkpoint ...')
        saveModel(net=byolmodel, save_path=paths['{}_checkpoint'.format(task)], epoch=epoch,
                  filename='{}_last_epoch.pth'.format(task+model_type), optimizer=optimizer, task=task)
        
        if (epoch+1)%train_config['save_model_epoch']==0:
            saveModel(net=byolmodel, save_path=paths['{}_checkpoint'.format(task)], epoch=epoch,
                      filename='{}_checkpoint-epoch{}.pth'.format(task+model_type, epoch), 
                      optimizer=optimizer, task=task)
        
        if top1 >=best_acc:
            best_acc = top1
            best_epoch = epoch
            saveModel(net=byolmodel, save_path=paths['{}_checkpoint'.format(task)], epoch=epoch,
                      filename='{}_best.pth'.format(task+model_type), optimizer=optimizer, task=task)
    
    # Mine the topk nearest neighbors at the very end (Train) 
    # These will be served as input to the SCAN loss.
    print(colored('Fill memory bank for mining the nearest neighbors (train) ...', 'blue'))
    fill_memory_bank(train_loader, byolmodel, memory_bank_base)
    topk = data_config['num_neighbors']
    print('Mine the nearest neighbors (Top-%d)' %(topk)) 
    indices, acc = memory_bank_base.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on train set is %.2f' %(topk, 100*acc))
    np.save(paths['topk_neighbors_train_path'], indices)
    final_train_info = 'Accuracy of top-{} nearest neighbors on train set is {:.2f}' .format(topk, 100*acc)
    logger.info(final_train_info)
    
    # Mine the topk nearest neighbors at the very end (Val)
    # These will be used for validation.
    print(colored('Fill memory bank for mining the nearest neighbors (val) ...', 'blue'))
    fill_memory_bank(val_loader, byolmodel, memory_bank_val)
    topk = 5
    print('Mine the nearest neighbors (Top-%d)' %(topk)) 
    indices, acc = memory_bank_val.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on val set is %.2f' %(topk, 100*acc))
    np.save(paths['topk_neighbors_val_path'], indices)
    final_val_info = 'Accuracy of top-{} nearest neighbors on val set is {:.2f}' .format(topk, 100*acc)
    logger.info(final_val_info)
    
    best_info = 'best acc:{:.6f} in epoch {:>4d}'.format(best_acc, best_epoch)
    logger.info(best_info)
    
def byol_one_epoch(train_loader, model, target_model, criterion, optimizer, epoch, logger, device=torch.device('cuda'), point2img=False):
    losses = AverageMeter('Loss', ':.6e')
    progress = ProgressMeter(len(train_loader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    target_model.eval()
    
    for i, batch in enumerate(train_loader):
        nodes1 = ToTensor(batch['node']).to(device=device, non_blocking=True)
        nodes2 = ToTensor(batch['node_augmented']).to(device=device, non_blocking=True)
        
        pred1 = model(nodes1)
        pred2 = model(nodes2)
        
        with torch.no_grad():
            proj1 = target_model(nodes1, forward_pass='backbone')
            proj2 = target_model(nodes2, forward_pass='backbone')
            proj1 = proj1.detach_()
            proj2 = proj2.detach_()
            
        loss1 = criterion(pred1, proj2)
        loss2 = criterion(pred2, proj1)
        loss = loss1 + loss2
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_moving_average(target_model, model)

        if i % 100 == 99 or i+1==len(train_loader):
            progress.display(i)
            infomation = 'Epoch: [{:>2d}] {:>4d}/{:>4d} items || Loss: {:.8e}'.format(epoch,i+1,len(train_loader), loss.item())
            logger.info(infomation)
    return model, target_model
 
#===================================Training of Simsiam Model===================================#
def train_simsiam(config):
    task='simsiam'
    
    train_config = config.train_config
    paths = config.path_config
    model_config = config.model_config
    data_config = config.data_config
    optimizer_config = config.optimizer_config
    criterion_config = config.criterion_config
    
    epoch_start=0
    # print cfg
    print(colored('Info Config:', 'blue'))
    print(colored(config.info_config, 'yellow'))
    print(colored('Model Config:', 'blue'))
    print(colored(model_config, 'yellow'))
    print(colored('Train Config:', 'blue'))
    print(colored(train_config, 'yellow'))
    print(colored('Paths:', 'blue'))
    print(colored(paths, 'yellow'))
    print(colored('Data Config:', 'blue'))
    print(colored(data_config, 'yellow'))
    print(colored('Optimizer Config:', 'blue'))
    print(colored(optimizer_config, 'yellow'))
    print(colored('Criterion Config:', 'blue'))
    print(colored(criterion_config, 'yellow'))
    
    model_type = config.model_type
    # create and load model
    print(colored('Get model', 'blue'))
    simsiammodel = getModel(config)
    last_model_file = os.path.join(config.path_config['{}_checkpoint'.format(task)],'{}_last_epoch.pth'.format(task+model_type))
    if os.path.exists(last_model_file):
        simsiammodel, epoch_start = loadModel(net=simsiammodel,
                                              save_path=paths['{}_checkpoint'.format(task)],
                                              file_class='last',
                                              model_name=task+model_type, 
                                              task=task)
        print(colored('Restart from checkpoint {}'.format(paths['{}_dir'.format(task)]), 'blue'))
    # create dataset
    dataset = getDataset(data_config)
    if os.path.exists(paths['train_index']) and os.path.exists(paths['val_index']):
        splited_out = splitDataset(dataset=dataset, cfg=config, use_pretrain_indexes=True)
    else:
        splited_out = splitDataset(dataset=dataset, cfg=config)
    train_loader, val_loader = splited_out['train_dataloader'], splited_out['val_dataloader']
    train_num, val_num = splited_out['train_num'], splited_out['val_num']
    print(colored('Get dataset and dataloaders', 'blue'))
    
    # Memory Bank
    print(colored('Build MemoryBank', 'blue'))
    
    memory_bank_base = MemoryBank(train_num, 
                                model_config['contrastive_feadim'],
                                data_config['n_class'], criterion_config['temperature'])
    memory_bank_base.cpu()
    memory_bank_val = MemoryBank(val_num,
                                model_config['contrastive_feadim'],
                                data_config['n_class'], criterion_config['temperature'])
    memory_bank_val.cpu()
    
    # loss
    print(colored('Get loss', 'blue'))
    criterion = get_criterion(criterion_config)
    print('Criterion is {}'.format(criterion.__class__.__name__))
    criterion = criterion.to(device=train_config['device'])
    
    # optimizer
    print(colored('Get optimizer', 'blue'))
    optimizer = get_optimizer(optimizer_config, simsiammodel)
    # CUDNN
    torch.backends.cudnn.benchmark = True
    
    device = train_config['device']
    simsiammodel = simsiammodel.to(device=device)
    
    # create logger
    logger = get_logger(path=paths['{}_log'.format(task)], filename='{}Model_train.log'.format(task))
    
    # Main loop
    print(colored('Starting main loop', 'blue'))
    start_time = time.time()
    best_acc = 0
    best_epoch = 0
    
    stats_path = os.path.join(paths['{}_log'.format(task)], 'stats')
    check_dirs(stats_path)
    
    for epoch in range(epoch_start, epoch_start+train_config['epochs']):
        print(colored('Epoch %d/%d' %(epoch+1, train_config['epochs']), 'yellow'))
        print(colored('-'*15, 'yellow'))
        
        epoch_start_time = time.time()

        # Adjust lr
        lr = adjust_learning_rate(config, optimizer, epoch)
        print('Adjusted learning rate to {:.8e}'.format(lr))
        
        # Train
        print('Train ...')
        simsiam_one_epoch(train_loader=train_loader, model=simsiammodel, criterion=criterion, optimizer=optimizer, epoch=epoch, logger=logger, device=device)
        
        # Fill memory bank
        print('Fill memory bank for kNN...')
        fill_memory_bank(train_loader, simsiammodel, memory_bank_base)
        
        # Evaluate (To monitor progress - Not for validation)
        print('Evaluate ...')
        top1 = contrastive_evaluate(val_loader, simsiammodel, memory_bank_base)
        print('Result of kNN evaluation is %.4f' %(top1))
        total_time=time.time() - start_time
        hours, mins, sec = time2hms(total_time)
        val_info = 'Epoch: [{:>2d}/{:>2d}] || Time: {:>3d} H {:>2d} M {:.3f} s || Result of kNN acc is :{:.8f} || epoch time: {:.3f} sec'.format(epoch, epoch_start + train_config['epochs'], hours, mins, sec,  top1, time.time()-epoch_start_time)
        logger.info(val_info)
        
        # Checkpoint
        print('Checkpoint ...')
        saveModel(net=simsiammodel, save_path=paths['{}_checkpoint'.format(task)], epoch=epoch,
                  filename='{}_last_epoch.pth'.format(task+model_type), optimizer=optimizer, task=task)
        
        if (epoch+1)%train_config['save_model_epoch']==0:
            saveModel(net=simsiammodel, save_path=paths['{}_checkpoint'.format(task)], epoch=epoch,
                      filename='{}_checkpoint-epoch{}.pth'.format(task+model_type, epoch), 
                      optimizer=optimizer, task=task)
        
        if top1 >=best_acc:
            best_acc = top1
            best_epoch = epoch
            saveModel(net=simsiammodel, save_path=paths['{}_checkpoint'.format(task)], epoch=epoch,
                      filename='{}_best.pth'.format(task+model_type), optimizer=optimizer, task=task)
    
    # Mine the topk nearest neighbors at the very end (Train) 
    # These will be served as input to the SCAN loss.
    print(colored('Fill memory bank for mining the nearest neighbors (train) ...', 'blue'))
    fill_memory_bank(train_loader, simsiammodel, memory_bank_base)
    topk = data_config['num_neighbors']
    print('Mine the nearest neighbors (Top-%d)' %(topk)) 
    indices, acc = memory_bank_base.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on train set is %.2f' %(topk, 100*acc))
    np.save(paths['topk_neighbors_train_path'], indices)
    final_train_info = 'Accuracy of top-{} nearest neighbors on train set is {:.2f}' .format(topk, 100*acc)
    logger.info(final_train_info)
    
    # Mine the topk nearest neighbors at the very end (Val)
    # These will be used for validation.
    print(colored('Fill memory bank for mining the nearest neighbors (val) ...', 'blue'))
    fill_memory_bank(val_loader, simsiammodel, memory_bank_val)
    topk = 5
    print('Mine the nearest neighbors (Top-%d)' %(topk)) 
    indices, acc = memory_bank_val.mine_nearest_neighbors(topk)
    print('Accuracy of top-%d nearest neighbors on val set is %.2f' %(topk, 100*acc))
    np.save(paths['topk_neighbors_val_path'], indices)
    final_val_info = 'Accuracy of top-{} nearest neighbors on val set is {:.2f}' .format(topk, 100*acc)
    logger.info(final_val_info)
    
    best_info = 'best acc:{:.6f} in epoch {:>4d}'.format(best_acc, best_epoch)
    logger.info(best_info)
    
def simsiam_one_epoch(train_loader, model, criterion, optimizer, epoch, logger, device=torch.device('cuda')):
    losses = AverageMeter('Loss', ':.6e')
    progress = ProgressMeter(len(train_loader),
        [losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    
    for i, batch in enumerate(train_loader):
        nodes1 = batch['node']
        nodes2 = batch['node_augmented']
        
            
        if type(nodes1)==np.ndarray:
            nodes1 = torch.from_numpy(nodes1).to(device=device, non_blocking=True)
            nodes2 = torch.from_numpy(nodes2).to(device=device, non_blocking=True)
        else:
            nodes1 = nodes1.cuda()
            nodes2 = nodes2.cuda()
            
        #input_ = torch.cat([nodes.unsqueeze(1), nodes_augmented.unsqueeze(1)], dim=1)
        with torch.no_grad():
            proj1 = model(nodes1, forward_pass='backbone').detach()
            proj2 = model(nodes2, forward_pass='backbone').detach()
        pred1 = model(nodes1)
        pred2 = model(nodes2)
        
        loss = criterion(pred1, pred2, proj1, proj2)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 25 == 0 or i+1==len(train_loader):
            progress.display(i)
            infomation = 'Epoch: [{:>2d}] {:>4d}/{:>4d} items || Loss: {:.8e}'.format(epoch,i+1,len(train_loader), loss.item())
            logger.info(infomation)

#===================================Training of Deepcluster Model===================================#
def train_deepcluster(config):
    task = 'deepcluster'

    train_config = config.train_config
    paths = config.path_config
    model_config = config.model_config
    data_config = config.data_config
    optimizer_config = config.optimizer_config
    criterion_config = config.criterion_config
    
    epoch_start=0
    # print cfg
    print(colored('Info Config:', 'blue'))
    print(colored(config.info_config, 'yellow'))
    print(colored('Model Config:', 'blue'))
    print(colored(model_config, 'yellow'))
    print(colored('Train Config:', 'blue'))
    print(colored(train_config, 'yellow'))
    print(colored('Paths:', 'blue'))
    print(colored(paths, 'yellow'))
    print(colored('Data Config:', 'blue'))
    print(colored(data_config, 'yellow'))
    print(colored('Optimizer Config:', 'blue'))
    print(colored(optimizer_config, 'yellow'))
    print(colored('Criterion Config:', 'blue'))
    print(colored(criterion_config, 'yellow'))
    
    # CUDNN
    torch.backends.cudnn.benchmark = True
    
    model_type = config.model_type
    pretext = config.pretext
    # create and load model
    print(colored('Get model', 'blue'))
    deepclusmodel = getModel(config)
    last_model_file = os.path.join(paths['{}_checkpoint'.format(task)],'{}_last_epoch.pth'.format(task+model_type))
    backbone_file = os.path.join(paths['{}_checkpoint'.format(pretext)], '{}_last_epoch.pth'.format(pretext+model_type))
    if os.path.exists(last_model_file):
        deepclusmodel, epoch_start = loadModel(net=deepclusmodel,
                                               save_path=paths['{}_checkpoint'.format(task)],
                                               file_class='last',
                                               model_name=task+model_type, 
                                               task=task)
        print(colored('Restart from checkpoint {}'.format(paths['{}_dir'.format(task)]), 'blue'))
    elif os.path.exists(backbone_file):
        deepclusmodel, _ = loadModel(net=deepclusmodel,
                                     save_path=paths['{}_checkpoint'.format(pretext)],
                                     file_class='last',
                                     model_name=pretext+model_type,
                                     task=task,
                                     load_backbone_only=True)
        print(colored('Load pre-train backbone from {} checkpoint {}'.format(pretext, paths['{}_dir'.format(pretext)]), 'blue'))
        #print(colored('WARNING: DeepCl will only update the cluster head', 'red'))
    else:
        print(colored('Train without pre-train backbone. ', 'red'))
    # to cuda
    device = train_config['device']
    deepclusmodel = deepclusmodel.to(device=device)
    
    # create logger
    logger = get_logger(path=paths['{}_log'.format(task)], filename='{}Model_train.log'.format(task))
    
    # loss
    print(colored('Get loss', 'blue'))
    criterion = get_criterion(criterion_config)
    criterion.cuda()
    # optimizer
    print(colored('Get optimizer', 'blue'))
    optimizer = get_optimizer(optimizer_config, deepclusmodel, nheads=model_config['nheads'], cluster_head_only=True)
    
    
    # create dataset
    dataset = getDataset(data_config)
    if os.path.exists(paths['train_index']) and os.path.exists(paths['val_index']):
        splited_out = splitDataset(dataset=dataset, cfg=config, use_pretrain_indexes=True)
    else:
        splited_out = splitDataset(dataset=dataset, cfg=config)
    train_loader, val_loader = splited_out['train_dataloader'], splited_out['val_dataloader']
    train_num, val_num = splited_out['train_num'], splited_out['val_num']
    print(colored('Get dataset and dataloaders', 'blue'))
    
    # init cluster center...
    deepcenter = DeepClusterCenter(nclusters=model_config['num_cluster'], 
                                   loader=train_loader, 
                                   backbone_dim=model_config['BackboneConfig']['feature_dim'],
                                   beta=0.9)
    deepcenter.init_kmeans_center(net=deepclusmodel, loader=train_loader)
    deepcenter.to(device=device)
    
    print(colored('Get DeepClusterCenter', 'blue'))
    # Main loop
    print(colored('Starting main loop', 'blue'))
    start_time = time.time()
    best_acc = 0
    best_epoch = 0
    
    stats_path = os.path.join(paths['{}_log'.format(task)], 'stats')
    check_dirs(stats_path)
    
    for epoch in range(epoch_start, epoch_start+train_config['epochs']):
        print(colored('Epoch %d/%d' %(epoch+1, train_config['epochs']), 'yellow'))
        print(colored('-'*15, 'yellow'))
        
        epoch_start_time = time.time()
        # Adjust lr
        lr = adjust_learning_rate(config, optimizer, epoch)
        print('Adjusted learning rate to {:.8e}'.format(lr))
        
        # Train
        print('Train ...')
        train_deepcluster_one_epoch(train_loader=train_loader,
                                    model=deepclusmodel,
                                    deepcenter=deepcenter,
                                    criterion=criterion,
                                    optimizer=optimizer,
                                    epoch=epoch,
                                    device=device,
                                    logger=logger)

        # Evaluate 
        print('Make prediction on training set ...')
        train_prediction = get_predictions(config, train_loader, deepclusmodel)
        
        train_clustering_stats = hungarian_evaluate(0, train_prediction, compute_confusion_matrix=False)
        #print('the lowest train loss is {:.8e}'.format(lowest_train_loss))
        print('acc:{:.8f}, ari:{:.8f}, nmi:{:.8f}'.format(train_clustering_stats['ACC'], train_clustering_stats['ARI'],train_clustering_stats['NMI']))
        print(train_clustering_stats['confusion_matrix'])
        
        train_info = 'Epoch: [{:>2d}/{:>2d}] || acc:{:.8f}, ari:{:.8f}, nmi:{:.8f}'.format(epoch+1, epoch_start + train_config['epochs'], train_clustering_stats['ACC'], train_clustering_stats['ARI'],train_clustering_stats['NMI'])
        logger.info(train_info)
        
        # Evaluate 
        print('Make prediction on validation set ...')
        val_prediction = get_predictions(config, val_loader, deepclusmodel)
        
        val_clustering_stats = hungarian_evaluate(0, val_prediction, compute_confusion_matrix=False)
        #print('the lowest val loss is {:.8e}'.format(lowest_val_loss))
        print('acc:{:.8f}, ari:{:.8f}, nmi:{:.8f}'.format(val_clustering_stats['ACC'], val_clustering_stats['ARI'],val_clustering_stats['NMI']))
        print(val_clustering_stats['confusion_matrix'])
        
        total_time=time.time() - start_time
        hours, mins, sec = time2hms(total_time)
        val_info = 'Epoch: [{:>2d}/{:>2d}] || Time: {:>3d} H {:>2d} M {:.3f} s || ACC: {:.8f} || ARI: {:.8f} || NMI: {:.8f} || epoch time: {:.3f} sec'.format(
                epoch, epoch_start + train_config['epochs'], hours, mins, sec, val_clustering_stats['ACC'], val_clustering_stats['ARI'], val_clustering_stats['NMI'], time.time()-epoch_start_time)
        
        logger.info(val_info)
        acc = val_clustering_stats['ACC']
        
        print('Checkpoint ...')
        saveModel(net=deepclusmodel, save_path=paths['{}_checkpoint'.format(task)], epoch=epoch,
                  filename='{}_last_epoch.pth'.format(task+model_type), optimizer=optimizer, task=task)
        
        if (epoch+1)%train_config['save_model_epoch']==0:
            saveModel(net=deepclusmodel, save_path=paths['{}_checkpoint'.format(task)], epoch=epoch,
                      filename='{}_checkpoint-epoch{}.pth'.format(task+model_type, epoch), 
                      optimizer=optimizer, task=task)
            save_clustering_stats(clustering_stats=val_clustering_stats, path=stats_path, filename='stats_epoch{}'.format(epoch+1))
        
        if acc>=best_acc:
            saveModel(net=deepclusmodel, save_path=paths['{}_checkpoint'.format(task)], epoch=epoch,
                      filename='{}_best.pth'.format(task+model_type), optimizer=optimizer, task=task)
            save_clustering_stats(clustering_stats=val_clustering_stats, path=stats_path, filename='best_stats')
            best_acc = acc
            best_epoch = epoch
    
    # Evaluate and save the final model
    print(colored('Evaluate model at the end', 'blue'))
    
    best_model_file = os.path.join(paths['{}_checkpoint'.format(task)],'{}_best.pth'.format(task+model_type))
    if os.path.exists(best_model_file):
        deepclusmodel, _ = loadModel(net=deepclusmodel,
                                     save_path=paths['{}_checkpoint'.format(task)],
                                     file_class='best',
                                     model_name=task+model_type, 
                                     task=task)
    
    predictions = get_predictions(config, val_loader, deepclusmodel)
    clustering_stats = hungarian_evaluate(0, predictions, 
                            class_names=data_config['class_name'], 
                            compute_confusion_matrix=True, 
                            confusion_matrix_file=os.path.join(paths['{}_confusion_matrix'.format(task)], 'confusion_matrix.png'))
    print(clustering_stats)
    save_clustering_stats(clustering_stats=clustering_stats, path=stats_path, filename='last_stats')
    
    bestinfo = 'best acc: {:.6f} in epoch {:>4d}'.format(best_acc, best_epoch)
    logger.info(bestinfo)
    
def train_deepcluster_one_epoch(train_loader, model, deepcenter:DeepClusterCenter, criterion, optimizer, epoch, logger, device=torch.device('cuda')):
    losses = AverageMeter('Loss', ':.8e')
    progress = ProgressMeter(len(train_loader), [losses],
                                prefix="Epoch: [{}]".format(epoch))
    
    model.train()
    
    for i, batch in enumerate(train_loader):
        if epoch%2==0:
            nodes = ToTensor(batch['node']).to(device=device, non_blocking=True)
            node_aug = ToTensor(batch['node_augmented']).to(device=device, non_blocking=True)
        else:
            nodes = ToTensor(batch['node_augmented']).to(device=device, non_blocking=True)
            node_aug = ToTensor(batch['node']).to(device=device, non_blocking=True)
        
        pred = model(nodes, forward_pass='return_all')
        prob, fea = pred['output'], pred['features']
        with torch.no_grad():
            aug_fea = model(node_aug, forward_pass='backbone')
            
        fea = fea.detach()
        pseudo_label = deepcenter.selflabel(fea=fea, aug_fea=aug_fea)
        if type(prob)==list:
            prob = prob[0]
        #print(pred.shape, label.shape)
        loss = criterion(prob, pseudo_label)
        losses.update(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #deepcenter.update_center()

        
        if i % 100 == 99 or i+1==len(train_loader):
            progress.display(i)
            infomation = 'Epoch: [{:>2d}] {:>4d}/{:>4d} items || Loss: {:.8e}'.format(epoch,i+1,len(train_loader), loss.item())
            logger.info(infomation)
    
