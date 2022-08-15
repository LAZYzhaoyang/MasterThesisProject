import os
import numpy as np
import torch

import pymoo
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize

# my lib

#from .plotlib import plotPointCloud, plotContrastNode, contrast_res
from .toollib import ToNumpy, ToTensor, random_index, normalize, unnormalize
from .modellib import getModel, loadModel
from ..config.configs import get_config, TubeOptimizingConfig

from .kmeanslib import PretrainKmeans

#=========================gen point could=========================#
def genInitPointCould():
    pass

def genTubeInitPointCould(h, r, grid_size=1, npoint=1024):
    hlist = np.arange(0,h, grid_size)
    base_node_num = int(2*np.pi*r/grid_size)
    point_num = base_node_num*h//grid_size
    while point_num<npoint:
        base_node_num=base_node_num*2
        point_num = base_node_num*h//grid_size
    base_angle = np.linspace(0, 2*np.pi, num=base_node_num, endpoint=False)
    node = []
    for high in hlist:
        floor_node = [r*np.cos(base_angle)[np.newaxis,:], 
                      r*np.sin(base_angle)[np.newaxis,:],
                      high*np.ones_like(base_angle)[np.newaxis,:]]
        floor_node = np.concatenate(floor_node, axis=0)
        node.append(floor_node)
    node = np.concatenate(node, axis=1)
    _, n = node.shape
    indices = random_index(indexnum=npoint, up=n, bottom=0)
    node = node[:,indices]
    node = node[np.newaxis,:,:]
    return node


#=========================utils=========================#

def paramNormalization(p, p_range, keys, not_buttom):
    assert len(p)==len(keys)
    assert len(p)==len(not_buttom)
    
    for i in range(len(p)):
        p[i] = normalize(d=p[i], r=p_range[keys[i]], no_buttom=not_buttom[i])
    return p    
    
    

def paramUnnormalization(p, p_range, keys, not_buttom):
    assert len(p)==len(keys)
    assert len(p)==len(not_buttom)
    
    for i in range(len(p)):
        p[i] = unnormalize(d=p[i], r=p_range[keys[i]], no_buttom=not_buttom[i])
    return p 

def getTubeV(h,r,t):
    v1 = np.pi*(r-t/2)*(r-t/2)*h
    v2 = np.pi*(r+t/2)*(r+t/2)*h
    v = v2-v1
    return v

def getTubeMass(rho, h,r,t):
    v = getTubeV(h=h, r=r, t=t)
    return rho*v

def UnitConversion():
    # height: mm->mm
    # radius: mm->mm
    # thick: mm->mm
    # etan: MPa
    # sigy: MPa
    # rho: t/mm3->g/mm3
    # e: MPa
    # ea: N-mm->J
    # mass: ton->g
    # pcf: N->N
    pass


def zeroone2real(x:np.ndarray, x_range:dict, keys, is_discrete):
    # x: [d,] or [n, d]
    # len(keys): d
    if len(x.shape)==1:
        x = x[np.newaxis,:] #[1,d]
    elif len(x.shape)>2:
        raise ValueError('x.shape must be [d,] or [n,d]')
    assert x.shape[1]==len(keys)
    assert x.shape[1]==len(is_discrete)
    for i in range(len(keys)):
        r = x_range[keys[i]]
        if is_discrete[i]==1:
            x[:,i]=np.around(x[:,i])
        x[:,i] = unnormalize(x[:,i], r, no_buttom=False)
    if x.shape[0]==1:
        x=x[0]
    return x

def getModelFileExternal(file_class:str='last', epoch:int=100):
    class_list=['best', 'last', 'epochs']
    file_extension_name=['best', 'last_epoch', 'checkpoint-epoch{}'.format(epoch)]
    assert file_class in class_list, 'file class must be one of [best, last, epoch]'
    for i in range(len(class_list)):
        if file_class == class_list[i]:
            file_ex = file_extension_name[i]
    return file_ex

def getKeysandDiscrete(opti_keys, opti_setting:dict):
    real_keys = []
    real_discrete = []
    real_not_buttom = []
    keys = opti_setting['keys']
    discrete = opti_setting['discrete']
    buttom = opti_setting['no_buttom']
    keys_can_be_opti = opti_setting['opti_keys']
    for k in opti_keys:
        assert k in keys_can_be_opti
        if k == 'material':
            real_keys.extend(['etan', 'sigy','rho', 'e'])
        else:
            real_keys.append(k)
    real_keys = list(set(real_keys))
    
    for i in range(len(real_keys)):
        rk = real_keys[i]
        d = discrete[keys.index(rk)]
        b = buttom[keys.index(rk)]
        real_discrete.append(d)
        real_not_buttom.append(b)
    
    return real_keys, real_discrete, real_not_buttom

#=========================Proxy unit=========================#
class ProxyModel:
    def __init__(self, config:TubeOptimizingConfig):
        self.config = config
        self.proxy = getModel(config.ProxyConfig)
        
        clstype = config.info['cluster_type']
        clscfg = config.ClustserModelConfig
        self.cluster_type = clstype
        if clstype == 'deepkmeans':
            self.cluster = PretrainKmeans(ncluster=clscfg['ncluster'],
                                          backbone_type=clscfg['backbone'], 
                                          pretext=clscfg['pretext'],
                                          point2img=clscfg['point2img'])
            
            if clscfg['pretrain_path'] is not None:
                self.cluster.load(root=clscfg['pretrain_path'], backbone_type=clscfg['backbone'])
            else:
                self.cluster.train()
                self.cluster.save()
        else:
            self.cluster = getModel(clscfg)
            
        self.num_cluster = config.info['ncluster']
        self.device = config.ProxyConfig.train_config['device']
        self.proxy = self.proxy.to(device=self.device)
        self.cluster = self.cluster.to(device=self.device)
        
    
    def __call__(self, init_node, params):
        init_node = ToTensor(init_node).to(device=self.device, dtype=torch.float32)
        params = ToTensor(params).to(device=self.device, dtype=torch.float32)
        self.proxy.eval()
        pred_node, pred_res = self.proxy(init_node, params)
        
        if self.cluster_type != 'deepkmeans':
            self.cluster.eval()
            clsres = self.cluster(pred_node)
        else:
            clsres = self.cluster(pred_node)
            clsres = torch.argmax(clsres, dim=-1)
        
        clsres = ToNumpy(clsres)
        pred_node = ToNumpy(pred_node)
        pred_res = ToNumpy(pred_res)
        
        out = {'response':pred_res, 'cluster_res':clsres, 'pred_node':pred_node}
        return out
            
    def loadPretrainModel(self, file_class:str='last', epoch:int=0):
        proxy_paths = self.config.ProxyConfig.path_config
        
        proxy_name = self.config.ProxyConfig.model_name
        file_ex = getModelFileExternal(file_class=file_class, epoch=epoch)
        proxy_file = os.path.join(proxy_paths['Model_Library'], '{}_{}.pth'.format(proxy_name, file_ex))
        if os.path.exists(proxy_file):
            self.proxy, _ = loadModel(net=self.proxy,
                                      save_path=proxy_paths['Model_Library'],
                                      file_class=file_class,
                                      model_name=proxy_name, 
                                      task='ResponseProxy',
                                      epoch=epoch)
        else:
            print('Could not found model file {}'.format(proxy_name))
        
        cluster_paths = self.config.ClustserModelConfig.path_config
        task = self.cluster_type
        cluster_backbone = self.config.info['cluster_backbone']
        if task != 'deepkmeans':
            cluster_file = os.path.join(cluster_paths['{}_checkpoint'.format(task)], '{}_{}.pth'.format(task+cluster_backbone, file_ex))
            if os.path.exists(cluster_file):
                self.cluster, _ = loadModel(net=self.cluster,
                                            save_path=cluster_paths['{}_checkpoint'.format(task)],
                                            file_class=file_class,
                                            model_name=task+cluster_backbone,
                                            task=task, 
                                            epoch=epoch)
            else:
                print('Could not found model file {}'.format(cluster_file))
                
        print('The Proxy and Cluster have been loaded pre-train model')
    
    def to(self, device):
        self.device=device
        self.proxy = self.proxy.to(device=device)
        self.cluster = self.cluster.to(device=device)
        
#=========================Optimizing unit=========================#     

class TubeDeformationOptimizing(ElementwiseProblem):
    def __init__(self, config:TubeOptimizingConfig, opti_keys=['height', 'radius', 'thick', 'material']):
        self.config = config
        
        self.proxy_model = ProxyModel(config=config)
        self.proxy_model.loadPretrainModel()
        
        self.paramconfig = config.TubeParams
        self.opti_setting = config.optimizingset
        
        real_key, real_discrete, real_buttom = getKeysandDiscrete(opti_keys=opti_keys, opti_setting=config.optimizingset)
        self.real_key = real_key
        self.real_discrete = real_discrete
        self.real_buttom = real_buttom
        
        self.npoint = config.ProxyConfig.data_config['npoint']
        
        x_num = len(self.real_key)
        xl = np.zeros(len(self.real_key))
        xu = np.ones_like(xl)
        
        self.input_keys = ['height', 'radius', 'thick', 'mass', 'etan', 'sigy', 'rho', 'e']
        self.input_buttom = []
        for i in range(len(self.input_keys)):
            ik = self.input_keys[i]
            bi = self.opti_setting['keys'].index(ik)
            self.input_buttom.append(self.opti_setting['no_buttom'][bi])
            
        self.output_keys = ['pcf', 'ea']
        
        
        super().__init__(n_var=len(self.real_key),
                         n_obj=2,
                         n_constr=1,
                         xl=xl,
                         xu=xu)
        
    def _evaluate(self, x, out, *args, **kwargs):
        real_x = zeroone2real(x=x, x_range=self.paramconfig, keys=self.real_key, is_discrete=self.real_discrete)
        input_keys = self.input_keys
        real_p = []
        p_buttom = self.input_buttom
        for i in range(len(input_keys)):
            ik = input_keys[i]
            if ik in self.real_key:
                p_i = self.real_key.index(ik)
                real_p.append(real_x[p_i])
            else:
                p_r = self.paramconfig[ik]
                real_p.append(p_r[0])
        h, r, t, rho = real_x[0], real_x[1], real_x[2], real_x[-2]
        mass = getTubeMass(rho=rho, h=h, r=r, t=t)
        mass_i = input_keys.index('mass')
        real_p[mass_i]=mass
        real_p = paramNormalization(real_p, p_range=self.paramconfig, keys=input_keys, not_buttom=p_buttom)
        real_p = np.array(real_p)[np.newaxis,:] #原始数据，还没归一化
        
        init_node = genTubeInitPointCould(h=h, r=r, npoint=self.npoint)
        
        pred = self.proxy_model(init_node=init_node, params=real_p)
        
        res = pred['response']
        res = res[0].tolist()
        res = paramUnnormalization(p=res, p_range=self.paramconfig, keys=self.output_keys, not_buttom=[1,1])
        pcf, ea = res[0], res[1]
        sea = ea/mass
        
        clsres = pred['cluster_res']
        out['F']=[pcf, -sea]
        out['G']=clsres-3
        

def TubeOptimizing(config:TubeOptimizingConfig):
    problem = TubeDeformationOptimizing(config=config)
    algorithm = NSGA2(pop_size=config.optimizingset['pop_size'],
                      sampling=FloatRandomSampling(),
                      crossover=SBX(eta=15, prob=0.9),
                      mutation=PM(eta=20),
                      eliminate_duplicates=True)
    termination = get_termination("n_gen", 100)
    res = minimize(problem=problem,
                   algorithm=algorithm,
                   termination=termination,
                   seed=1,
                   save_history=True,
                   verbose=True)
    return res

        
        
        


