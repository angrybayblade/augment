import numpy as np 
import pandas as pd
import cv2

from scipy import ndimage


"""
Augmenatioan Tools For Object Detection And Recognition
"""

class RandomZoom(object):
    """
    Perform Random Zoom On Given Image
    """
    def __init__(self,img_size:int=1024,center_zoom:bool=False,resize:int=False):        
        self.img_size = img_size
        self.lower = img_size //2
        self.upper = img_size //1.25 
        self.center_zoom = center_zoom
        self.resize = resize

    def __call__(self,img:np.ndarray,box:pd.DataFrame,center_zoom:bool=False,resize:bool=False):
        r = np.random.randint(self.lower,self.upper)
        r += r%2
        pad = int((self.img_size - r) / 2)

        c_x_adj = np.random.randint(-pad,pad)
        c_y_adj = np.random.randint(-pad,pad)

        if center_zoom or center_zoom:
            c_x_adj = 0
            c_y_adj = 0

        c_x = self.lower + c_x_adj
        c_y = self.lower + c_y_adj
        c_xmin = c_x - (r//2)
        c_xmax = c_x + (r//2)
        c_ymin = c_y - (r//2)
        c_ymax = c_y + (r//2)

        img_ = img.copy()[c_ymin:c_ymax,c_xmin:c_xmax]

        box_ = box.copy()
        box_['w'] = (box_.xmax.values - box_.xmin)
        box_['h'] = (box_.ymax.values - box_.ymin)
        box_['x'] = box_.xmin.values + (box_.w)//2
        box_['y'] = box_.ymin.values + (box_.h)//2
        
        y_check = np.logical_and( box_['y'].values > c_ymin , box_['y'].values < c_ymax)
        x_check = np.logical_and( box_['x'].values > c_xmin , box_['x'].values < c_xmax)
        check = np.logical_and(y_check,x_check)

        box_ = box_[check]
        box_['y'] = np.maximum(box_['y'].values - c_ymin,0)
        box_['x'] = np.maximum(box_['x'].values - c_xmin,0)
        box_['xmin'] = box_['x'].values - (box_['w'].values // 2)
        box_['xmax'] = box_['x'].values + (box_['w'].values // 2)
        box_['ymin'] = box_['y'].values - (box_['h'].values // 2)
        box_['ymax'] = box_['y'].values + (box_['h'].values // 2)

        if self.resize or resize:
            size = self.resize if self.resize else resize
            img_ = cv2.resize(img_,(size,size),interpolation=cv2.INTER_CUBIC)
            box_ = box_ * (size/r)
            return img_,box_[['xmin','ymin','xmax','ymax']] 
        
        return img_,box_[['xmin','ymin','xmax','ymax']]


class RandomHsv(object):
    """
    RandomBrightness
    
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5 ,1.6 ]
        
    """
    def __init__(self,img_size:int=1024,resize:int=0):
        self.img_size = img_size
        self.resize = resize
        self.choices_0 = list(np.arange(-30,-18)) + list(np.arange(12,24))            
        
    def __call__(self,img:np.ndarray,boxes:pd.DataFrame,resize:int=0):
        
        img = cv2.cvtColor(img.copy(),cv2.COLOR_RGB2HSV).astype(np.int)
        img[:,:,2] = img[:,:,2] + np.random.choice(self.choices_0)
        img = np.maximum(np.minimum(img,255),0).astype(np.uint8)
        img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
        
        if self.resize or resize:
            size = self.resize if self.resize else resize
            img = cv2.resize(img,(size,size),interpolation=cv2.INTER_CUBIC)
            boxes = boxes * (size/self.img_size)
            return img,boxes[['xmin','ymin','xmax','ymax']] 
        
        return img,boxes


class RandomRotate(object):
    """
    RandomRotate
    """
    def __init__(self,img_size:int=1024,resize:int=0):
        self.img_size = img_size
        self.angles = [15,75,90,105,165,180,195,255,270,285,345]
        self.offset = 0.5
        self.angle = None
        self.resize = resize
        
    """
    y_ = -x*sin + y*cos 
    x_ =  x*cos + y*sin
    """


    def get_new_loc(self,x,y,angle=None):
        x -= self.offset
        y -= self.offset
        theta = np.deg2rad(angle if angle else self.angle) 
        sin = np.sin(theta)
        cos = np.cos(theta)
        y_ = (-x * sin + y * cos ) + self.offset
        x_ = ( x * cos + y * sin ) + self.offset
        return x_,y_
    
    def get_new_box(self,box):
        xmin, ymin, xmax, ymax = box
        p = np.array([
            self.get_new_loc(XX,YY) 
            for 
                XX,YY 
            in 
                [
                    (xmin,ymin),
                    (xmin,ymax),
                    (xmax,ymin),
                    (xmax,ymax)
                ]
        ])

        xmin = p[:,0].min()
        xmax = p[:,0].max()
        ymin = p[:,1].min()
        ymax = p[:,1].max()
        
        return xmin, ymin, xmax, ymax

    def __call__(self,img:np.ndarray,box:pd.DataFrame,resize:int=0):
        box_ = box[['xmin', 'ymin', 'xmax', 'ymax']].copy() / self.img_size
        self.angle = np.random.choice(self.angles)
        box_[['xmin', 'ymin', 'xmax', 'ymax']] = (np.maximum(np.apply_along_axis(self.get_new_box,axis=1,arr=box_),0)*self.img_size).astype(int)
        img_ = ndimage.rotate(img.copy(),self.angle,reshape=False)
        
        if self.resize or resize:
            size = self.resize if self.resize else resize
            img_ = cv2.resize(img_,(size,size),interpolation=cv2.INTER_CUBIC)
            box_ = box_ * (size/self.img_size)
            return img_.astype(np.uint8),box_[['xmin','ymin','xmax','ymax']]
        
        return img_,box_[['xmin', 'ymin', 'xmax', 'ymax']]