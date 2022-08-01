from utils.trainlib import train_proxy, train_supervised, \
    train_simclr, train_scan, train_simsiam, train_byol, \
        train_deepcluster


if __name__ == "__main__":
    #warnings.filterwarnings('ignore')
    #train_proxy()
    
    #train_supervised(model_type='PointSwin', opti='adamw')
    #train_supervised(model_type='PointTrans', opti='adamw')
    
    #train_simclr(model_type='PointTrans', point2img=False, opti='adamw')
    #train_simclr(model_type='PointSwin', point2img=False, opti='adamw')
    
    #train_scan(model_type='PointTrans', point2img=False, opti='adamw', pretext='simclr')
    #train_scan(model_type='PointSwin', point2img=False, opti='adamw', pretext='byol')
    
    #train_simsiam(model_type='PointSwin', point2img=False, opti='adamw')
    #train_simsiam(model_type='PointTrans', point2img=False, opti='adamw')
    
    #train_byol(model_type='PointSwin', point2img=False, opti='adamw')
    #train_byol(model_type='PointTrans', point2img=False, opti='adamw')
    
    train_deepcluster(model_type='PointSwin', point2img=False, opti='adamw', pretext='byol')
    #train_deepcluster(model_type='PointSwin', point2img=False, opti='adamw', pretext='supervised')