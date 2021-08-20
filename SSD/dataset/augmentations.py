import cv2
import numpy as np
import torch
from SSD.augmentation_params import AUGMENTATION_PARAMS

class Transforms():
    def __init__(self,train = False, normalized_coord_boxes = True):
        """
        if train = false , only normalize and resize image
        """
        self.train = train
        self.normalize_coord_boxes = normalized_coord_boxes
    
    def __call__(self,img,boxes):
        """
        img : np.array
        boxes : torch tensor with shape [num_boxes,4]
        """
        if self.normalized_coord_boxes != True:
            boxes[:,0] /= img.shape[1]
            boxes[:,1] /= img.shape[0]
            boxes[:,2] /= img.shape[1]
            boxes[:,3] /= img.shape[0]
        
        pass

    
    def normalize(self,img,mean,std):
        pass

    def resize(self,img,boxes):
        img = cv2.resize(img,(AUGMENTATION_PARAMS.IMG_NEW_WIDTH,AUGMENTATION_PARAMS.IMG_NEW_HEIGHT),cv2.INTER_LINEAR)
        
        pass

    def randomValue(self,img):
        pass

    def randomSaturation(self,img):
        pass
    
    def randomHue(self,img):
        pass
    
    def randomVerticalFlip(self,img,boxes):
        pass

    def randomHorizontalFlip(self,img,boxes):
        pass

