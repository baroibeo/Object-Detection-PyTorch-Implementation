import torch
import numpy as np
import cv2
import random
from torchvision import transforms

import os,sys
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from config import NEW_HEIGHT, NEW_WIDTH


class Transform():
    def __init__(self,img,bboxes,train = False):
        self.img = img
        self.bboxes = bboxes # [xmin,ymin,xmax,ymax]
        self.train = train
    
    def take_transform(self):
        img = cv2.resize(self.img, (NEW_HEIGHT,NEW_WIDTH),cv2.INTER_LINEAR)
        bboxes = self.bboxes

        if self.train == False:
            img = self.normalize(img)
            return img, bboxes
        
        else:
            img = self.randomBlur(img)
            img = self.randomHue(img)
            img = self.randomSaturation(img)
            img = self.randomValue(img)
            img,bboxes = self.randomHorizontalFlip(img,bboxes)
            img,bboxes = self.randomVerticalFlip(img,bboxes)
            img = self.normalize(img)
            return img,bboxes
    
    def normalize(self,img):
        #mean 0.5 , std 0.5 
        return (img - 128)/255.0
    
    def randomBlur(self,img):
        if random.random() < 0.5 :
            img = cv2.blur(img,ksize = (3,3))
        
        return img

    def randomHue(self,img):
        if random.random() < 0.5:
            hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
            h,s,v = cv2.split(hsv)
            k = random.uniform(0.5,1.5)
            h = h*k
            h = np.clip(h,0,255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            img = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
        return img

    def randomSaturation(self,img):
        if random.random() < 0.5:
            hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
            h,s,v = cv2.split(hsv)
            k = random.uniform(0.5,1.5)
            s = s*k
            s = np.clip(s,0,255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            img = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
        return img

    def randomValue(self,img):
        if random.random() < 0.5:
            hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
            h,s,v = cv2.split(hsv)
            k = random.uniform(0.5,1.5)
            v = v*k
            v = np.clip(v,0,255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            img = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
        
        return img

    def randomHorizontalFlip(self,img,bboxes):
        if random.random() < 0.5:
            img = np.fliplr(img)
            xmin = bboxes[:,0]
            xmax = bboxes[:,2]
            xmin_new = 1 - xmax
            xmax_new = 1 - xmin
            bboxes[:,0] = xmin_new
            bboxes[:,2] = xmax_new

        return img,bboxes

    def randomVerticalFlip(self,img,bboxes):
        if random.random() < 0.5:
            img = np.flipud(img)
            ymin = bboxes[:,1]
            ymax = bboxes[:,3]
            ymin_new = 1 - ymax
            ymax_new = 1 - ymin
            bboxes[:,1] = ymin_new
            bboxes[:,3] = ymax_new
        
        return img,bboxes

