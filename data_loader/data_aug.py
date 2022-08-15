import imp
import numpy as np
import torch
import time
from ..utils.toollib import squeeze_node, unsqueeze_node


def fps(points, num):
    cids = []
    cid = np.random.choice(points.shape[0])
    cids.append(cid)
    id_flag = np.zeros(points.shape[0])
    id_flag[cid] = 1

    dist = torch.zeros(points.shape[0]) + 1e4
    dist = dist.type_as(points)
    while np.sum(id_flag) < num:
        dist_c = torch.norm(points - points[cids[-1]], p=2, dim=1)
        dist = torch.where(dist<dist_c, dist, dist_c)
        dist[id_flag == 1] = 1e4
        new_cid = torch.argmin(dist)
        id_flag[new_cid] = 1
        cids.append(new_cid)
    cids = torch.Tensor(cids)
    return cids

def angle_axis(angle, axis):
    # type: (float, np.ndarray) -> float
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle
    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about
    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                                [u[2], 0.0, -u[0]],
                                [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )
    # yapf: enable
    return R.float()


class PointcloudScale(object):
    def __init__(self, lo=0.8, hi=1.25, p=1):
        self.lo, self.hi = lo, hi
        self.p = p

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        scaler = np.random.uniform(self.lo, self.hi)
        points *= scaler
        return points


class PointcloudRotate(object):
    def __init__(self, axis=np.array([0.0, 1.0, 0.0]), p=1):
        self.axis = axis
        self.p = p

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        if self.axis is None:
            angles = np.random.uniform(size=3) * 2 * np.pi
            Rx = angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))
            Ry = angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))
            Rz = angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))

            rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)
        else:
            rotation_angle = np.random.uniform() * 2 * np.pi
            rotation_matrix = angle_axis(rotation_angle, self.axis)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points
        

class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05, p=1):
        self.std, self.clip = std, clip
        self.p = p

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        jittered_data = (
            points.new(points.size(0), 3)
            .normal_(mean=0.0, std=self.std)
            .clamp_(-self.clip, self.clip)
        )
        points[:, 0:3] += jittered_data
        return points


class PointcloudTranslate(object):
    def __init__(self, translate_range=0.1, p=1):
        self.translate_range = translate_range
        self.p = p

    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points

        points = points.numpy()
        coord_min = np.min(points[:,:3], axis=0)
        coord_max = np.max(points[:,:3], axis=0)
        coord_diff = coord_max - coord_min
        translation = np.random.uniform(-self.translate_range, self.translate_range, size=(3)) * coord_diff
        points[:, 0:3] += translation
        return torch.from_numpy(points).float()
    
    
class PointcloudToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).float()

class PointcloudToNumpy(object):
    def __call__(self, points):
        return points.numpy()
    

class PointcloudNormalize(object):
    def __init__(self, radius=1):
        self.radius = radius

    def pc_normalize(self, pc):
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def __call__(self, points):
        pc = points.numpy()
        pc[:, 0:3] = self.pc_normalize(pc[:, 0:3])
        return torch.from_numpy(pc).float()


class PointcloudShuffle(object):
    def __init__(self, point_num=1024, p=1):
        self.point_num = point_num
        self.p = p
    def __call__(self, points):
        if np.random.uniform(0, 1) > self.p:
            return points
        
        if type(points)==torch.Tensor:
            points = points.numpy()
        
        if len(points.shape) == 2:
            tc, _ = points.shape
            if tc%3==0:
                points=unsqueeze_node(points, c=3)
            elif tc%self.point_num==0:
                points=unsqueeze_node(points, c=self.point_num).transpose(0,2,1)
            else:
                ValueError('points shape must be [t*c, n] or [t*n, c]')
        
        index = np.arange(self.point_num)
        np.random.shuffle(index)
        
        points = points[:,:,index]
        
        points = points.transpose(0,2,1)
        points = squeeze_node(points)
        
        return torch.from_numpy(points).float()
   
class PointcloudSample(object):
    def __init__(self, point_num=1024, sample_num=1024):
        self.point_num = point_num
        self.sample_num = sample_num
    def __call__(self, points):
        if type(points)==torch.Tensor:
            points = points.numpy()
        
        if len(points.shape) == 2:
            tc, _ = points.shape
            if tc%3==0:
                points=unsqueeze_node(points, c=3)
            elif tc%self.point_num==0:
                points=unsqueeze_node(points, c=self.point_num).transpose(0,2,1)
            else:
                ValueError('points shape must be [t*c, n] or [t*n, c]')
        
        index = np.arange(self.point_num)
        np.random.shuffle(index)
        index = index[:self.sample_num]
        
        points = points[:,:,index]
        
        points = points.transpose(0,2,1)
        points = squeeze_node(points)
        
        return torch.from_numpy(points).float() 

class PointcloudRandomCrop(object):
    def __init__(self, point_num=1024, axis=1, x_min=0.3, x_max=0.7, ar_min=0.75, ar_max=1.33, p=1, min_num_points=1024, max_try_num=10):
        self.x_min = x_min
        self.x_max = x_max

        self.ar_min = ar_min
        self.ar_max = ar_max

        self.p = p
        self.axis=axis
        
        self.point_num = point_num

        self.max_try_num = max_try_num
        self.min_num_points = min_num_points

    def __call__(self, points):
        #print(points.shape)
        if np.random.uniform(0, 1) > self.p:
            return points
        
        if type(points)==torch.Tensor:
            points = points.numpy()
        
        ori_points=points
        
        if len(points.shape) == 2:
            tc, _ = points.shape
            if tc%3==0:
                ori_points=unsqueeze_node(points, c=3).transpose(0,2,1)
            elif tc%self.point_num==0:
                ori_points=unsqueeze_node(points, c=self.point_num)
            else:
                ValueError('points shape must be [t*c, n] or [t*n, c]')
            # points [t, n, c]
        points = ori_points[0]
        
        isvalid = False
        try_num = 0
        while not isvalid:
            coord_min = np.min(points[:,self.axis], axis=0)
            coord_max = np.max(points[:,self.axis], axis=0)
            coord_diff = coord_max - coord_min
            # resampling later, so only consider crop here
            
            
            #new_coord_range = np.zeros(3)
            new_coord_range = np.random.uniform(self.x_min, self.x_max)
            ar = np.random.uniform(self.ar_min, self.ar_max)
            # new_coord_range[1] = np.random.uniform(self.ar_min, self.ar_max) * new_coord_range[0]
            # new_coord_range[2] = np.random.uniform(self.ar_min, self.ar_max) * new_coord_range[0]
            new_coord_min = new_coord_range * ar
            new_coord_max = new_coord_range / ar
            # new_coord_range = np.where(new_coord_range>1, 1, new_coord_range)

            if ar>1:
                new_coord_min, new_coord_max = new_coord_max, new_coord_min

            new_coord_min = coord_min + coord_diff * new_coord_min
            new_coord_max = coord_min + coord_diff * new_coord_max

            

            new_indices = (points[:, self.axis] > new_coord_min) & (points[:, self.axis] < new_coord_max)
            #new_indices = np.sum(new_indices, axis=1) == 3
            new_points = points[new_indices]

            # other_num = points.shape[0] - new_points.shape[0]
            # if new_points.shape[0] > 0:
            #     isvalid = True
            if new_points.shape[0] >= self.min_num_points and new_points.shape[0] < points.shape[0]:
                isvalid = True

            try_num += 1
            if try_num > self.max_try_num:
                ori_points = squeeze_node(ori_points)
                return torch.from_numpy(ori_points).float()
        new_points = ori_points[:, new_indices, :]
        new_points = squeeze_node(new_points)
        # other_indices = np.random.choice(np.arange(new_points.shape[0]), other_num)
        # other_points = new_points[other_indices]
        # new_points = np.concatenate([new_points, other_points], axis=0)

        # new_points[:,:3] = (new_points[:,:3] - new_coord_min) / (new_coord_max - new_coord_min) * coord_diff + coord_min
        return torch.from_numpy(new_points).float()
    
class PointcloudUpSampling(object):
    def __init__(self, max_num_points=1024, time_step=5, radius=0.1, nsample=5, centroid="random"):
        self.max_num_points = max_num_points
        # self.radius = radius
        self.centroid = centroid
        self.nsample = nsample
        self.time_step = time_step

    def __call__(self, points):
        t0 = time.time()
        
        if type(points)==torch.Tensor:
            points = points.numpy()
        
        ori_points=points
        
        
        if len(points.shape) == 2:
            tc, _ = points.shape
            if tc/self.time_step==3:
                ori_points=unsqueeze_node(points, c=3).transpose(0,2,1)
            else:
                ori_points=unsqueeze_node(points, c=tc//self.time_step)
            # points [t, n, c]
        points = ori_points[0]
        points = torch.from_numpy(points).float()

        p_num = ori_points.shape[1]
        
        ori_points = torch.from_numpy(ori_points).float()
        
        
        
        if p_num >= self.max_num_points:
            return ori_points

        c_num = self.max_num_points - p_num

        if self.centroid == "random":
            cids = np.random.choice(np.arange(p_num), c_num)
        else:
            assert self.centroid == "fps"
            fps_num = c_num / self.nsample
            fps_ids = fps(points, fps_num)
            cids = np.random.choice(fps_ids, c_num)

        xyzs = points[:, :3]
        loc_matmul = torch.matmul(xyzs, xyzs.t())
        loc_norm = xyzs * xyzs
        r = torch.sum(loc_norm, -1, keepdim=True)

        r_t = r.t()  # 转置
        dist = r - 2 * loc_matmul + r_t
        # adj_matrix = torch.sqrt(dist + 1e-6)

        dist = dist[cids]
        # adj_sort = torch.argsort(adj_matrix, 1)
        adj_topk = torch.topk(dist, k=self.nsample*2, dim=1, largest=False)[1]

        uniform = np.random.uniform(0, 1, (cids.shape[0], self.nsample*2))
        median = np.median(uniform, axis=1, keepdims=True)
        # choice = adj_sort[:, 0:self.nsample*2][uniform > median]  # (c_num, n_samples)
        choice = adj_topk[uniform > median]  # (c_num, n_samples)

        choice = choice.reshape(-1, self.nsample)
        
        sample_points = ori_points[:,choice,:] # (t, c_num, n_samples, 3)
        
        new_points = torch.mean(sample_points, dim=2)
        new_points = torch.cat([ori_points, new_points], 1)
        
        new_points = new_points.numpy()
        new_points = squeeze_node(new_points)

        return torch.from_numpy(new_points).float()
    
class Compose(object):
    def __init__(self, transformers:tuple, time_step:int=5):
        self.transformers = transformers
        self.time_step=time_step
    def __call__(self, points:np.array):
        # points [t, c, n]
        #print('ori point: {}'.format(points.shape))
        points = points.transpose(0,2,1)
        points = squeeze_node(points)
        for transformer in self.transformers:
            points = transformer(points)
        #print(len(points.shape))
        tn = points.shape[0]
        num = tn//self.time_step
        points = unsqueeze_node(points, c=num)# points [t, n, c]
        #print(points.shape)
        points = points.transpose(0,2,1)
        points = squeeze_node(points)
        #print('new point: {}'.format(points.shape))
        return points
    
def get_transformers():
    ori = [PointcloudToTensor(),
           PointcloudNormalize(),
           PointcloudJitter(p=0.5),
           PointcloudShuffle(p=0.5),
           PointcloudToNumpy()]
    
    weak = [PointcloudToTensor(),
            PointcloudNormalize(),
            PointcloudJitter(p=0.5),
            PointcloudShuffle(p=0.5),
            PointcloudTranslate(p=0.5), 
            PointcloudScale(p=0.5),
            PointcloudRotate(p=0.5),
            PointcloudToNumpy()]
    
    strong = [PointcloudToTensor(),
              #PointcloudNormalize(),
              #PointcloudShuffle(),
              PointcloudRandomCrop(min_num_points=128, p=1),
              PointcloudJitter(p=0.5),
              PointcloudTranslate(p=0.5), 
              PointcloudScale(p=0.5),
              PointcloudUpSampling(time_step=5),
              PointcloudToNumpy()]
    
    ori_transformer = Compose(transformers=ori)
    weak_transformer = Compose(transformers=weak)
    strong_transformer = Compose(transformers=strong)
    
    transformers = {'standard': ori_transformer,
                    'weak_augment': weak_transformer,
                    'strong_augment':strong_transformer}
    
    return transformers


def get_supervised_transform():
    transforms = [PointcloudToTensor(),
                  #PointcloudNormalize(),
                  PointcloudJitter(p=0.5),
                  PointcloudShuffle(p=0.5),
                  PointcloudTranslate(p=0.5), 
                  PointcloudScale(p=0.5),
                  PointcloudRotate(p=0.5),
                  PointcloudToNumpy()]
    
    trans = Compose(transforms)
    
    return trans

def getAugmentedTransform():
    stand = [PointcloudToTensor(),
             #PointcloudNormalize(),
             PointcloudJitter(p=0.5),
             PointcloudShuffle(p=0.5),
             PointcloudScale(p=0.5),
             PointcloudRotate(p=0.5),
             PointcloudToNumpy()]
    
    aug = [PointcloudToTensor(),
           #PointcloudNormalize(),
           PointcloudJitter(p=0.5),
           PointcloudShuffle(p=0.5),
           PointcloudTranslate(p=0.5), 
           PointcloudScale(p=0.5),
           PointcloudRotate(p=0.5),
           PointcloudToNumpy()]
    
    stand_transform = Compose(transformers=stand)
    aug_transform = Compose(transformers=aug)
    
    transforms = {'standard':stand_transform,
                  'augment':aug_transform}
    
    return transforms

def getNeighborTransform():
    stand = [PointcloudToTensor(),
             #PointcloudNormalize(),
             PointcloudJitter(p=0.5),
             PointcloudShuffle(p=0.5),
             PointcloudScale(p=0.5),
             PointcloudRotate(p=0.5),
             PointcloudToNumpy()]
    
    aug = [PointcloudToTensor(),
           #PointcloudNormalize(),
           PointcloudJitter(p=0.5),
           PointcloudShuffle(p=0.5),
           PointcloudTranslate(p=0.5), 
           PointcloudScale(p=0.5),
           PointcloudRotate(p=0.5),
           PointcloudToNumpy()]
    
    stand_transform = Compose(transformers=stand)
    aug_transform = Compose(transformers=aug)
    
    transforms = {'standard':stand_transform,
                  'augment':aug_transform}
    
    return transforms

def getSpiceTransform():
    ori = [PointcloudToTensor(),
           #PointcloudNormalize(),
           PointcloudJitter(p=0.5),
           PointcloudShuffle(p=0.5),
           PointcloudTranslate(p=0.5), 
           PointcloudRotate(p=0.5),
           PointcloudToNumpy()]
    
    aug1 = [PointcloudToTensor(),
            #PointcloudNormalize(),
            PointcloudJitter(p=0.5),
            PointcloudShuffle(p=0.5),
            PointcloudTranslate(p=0.5), 
            PointcloudScale(p=0.5),
            PointcloudToNumpy()]
    
    aug2 = [PointcloudToTensor(),
            #PointcloudNormalize(),
            #PointcloudRandomCrop(min_num_points=128, p=1),
            PointcloudJitter(p=0.5),
            PointcloudTranslate(p=0.5), 
            PointcloudScale(p=0.5),
            PointcloudRotate(p=0.5),
            PointcloudShuffle(p=0.5),
            #PointcloudUpSampling(time_step=5),
            PointcloudToNumpy()]
    
    ori_transformer = Compose(transformers=ori)
    aug1_transformer = Compose(transformers=aug1)
    aug2_transformer = Compose(transformers=aug2)
    
    transformers = {'standard': ori_transformer,
                    'weak_augment': aug1_transformer,
                    'strong_augment':aug2_transformer}
    
    return transformers
