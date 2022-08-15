from ..model.ClusteringModel import getClusterModel, saveClusterModel, loadClusterModel
from ..model.ResponseProxyModel import getProxyModel, save_ResponseProxyModel, load_ResponseProxyModel
from ..model.SupervisedModel import getSupervisedModel, saveSupervisedModel, loadSupervisedModel

def getModel(CFG):
    task = CFG.model_config['task']
    if task == 'ResponseProxy':
        model = getProxyModel(config=CFG.model_config, model_type=CFG.model_type)
    elif task in ['supervised']:
        model = getSupervisedModel(config=CFG.model_config['BackboneConfig'], ModelType=CFG.model_config['BackboneType'])
    elif task in ['simclr', 'byol', 'simsiam', 'moco', 'deepcluster', 'scan', 'selflabel', 'spice']:
        model = getClusterModel(config=CFG.model_config)
    else:
        raise ValueError('Invalid task {}'.format(task))
    
    return model


def saveModel(net, 
              save_path:str, 
              epoch:int, 
              filename:str, 
              optimizer=None, 
              task:str='ResponseProxy'):
    if task=='ResponseProxy':
        save_ResponseProxyModel(proxymodel=net, save_path=save_path,epoch=epoch, filename=filename, optimizer=optimizer)
    elif task in ['simclr', 'simsiam', 'byol', 'deepcluster', 'moco', 'scan', 'selflabel', 'spice']:
        saveClusterModel(clustermodel=net, save_path=save_path, epoch=epoch, filename=filename, optimizer=optimizer)
    elif task in ['supervised']:
        saveSupervisedModel(supervisednet=net, save_path=save_path, epoch=epoch, filename=filename, optimizer=optimizer)
    else:
        raise ValueError('Invalid task {}'.format(task))
    
def loadModel(net, save_path:str, 
              file_class:str, 
              model_name:str='PointSwin', 
              epoch:int=0, 
              task:str='ResponseProxy',
              load_backbone_only=False):
    if task=='ResponseProxy':
        loadnet, epoch_start = load_ResponseProxyModel(net=net, save_path=save_path, file_class=file_class, model_name=model_name, epoch=epoch)
    elif task in ['simclr', 'simsiam', 'byol', 'deepcluster', 'moco', 'scan', 'selflabel', 'spice']:
        loadnet, epoch_start = loadClusterModel(net=net, save_path=save_path, file_class=file_class, model_name=model_name, epoch=epoch, load_backbone_only=load_backbone_only)
    elif task in ['supervised']:
        loadnet, epoch_start = loadSupervisedModel(net=net, save_path=save_path, file_class=file_class, model_name=model_name, epoch=epoch)
    else:
        raise ValueError('Invalid task {}'.format(task))
    
    return loadnet, epoch_start
