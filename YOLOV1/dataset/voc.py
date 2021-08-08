import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from data_augmentation import Transform
from PIL import Image


class VOC(Dataset):
    def __init__(self, img_root, anno_dir, num_classes, grid_size, nboxes, train=False):
        """
        img_root : directory contains images
        anno_dir : directory contains txt files 
        num_classes : number of class need to classify after detecting
        grid_size : size to divide img into grids 
        train : default False for test dataset (not involved transform)  , if true , this will be for train dataset (involved transform)
        """
        super(VOC, self).__init__()
        self.img_root = img_root
        self.anno_dir = anno_dir
        self.anno_filenames = os.listdir(anno_dir)
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.nboxes = nboxes
        self.train = train

    def __len__(self):
        return len(self.anno_filenames)

    def __getitem__(self, idx):
        anno_filename = self.anno_filenames[idx]
        anno = open(os.path.join(self.anno_dir, anno_filename),
                    'r').readline().split(' ')
        img = np.array(Image.open(os.path.join(self.img_root, anno[0])))

        #Get bndboxes and labels
        num_boxes = int(anno[3])
        # xmin, ymin ,xmax ,ymax
        bndboxes = np.zeros(shape=(num_boxes, 4), dtype=np.float32)
        labels = np.zeros(shape=(num_boxes,), dtype=np.int)
        for i in range(num_boxes):
            bndboxes[i, 0] = float(anno[4 + 5*i])
            bndboxes[i, 1] = float(anno[4 + 5*i + 1])
            bndboxes[i, 2] = float(anno[4 + 5*i + 2])
            bndboxes[i, 3] = float(anno[4 + 5*i + 3])
            labels[i] = int(anno[4 + 5*i + 4])

        transform = Transform(img, bndboxes, self.train)
        # img,bndboxes = transform.take_transform()
        target = self.encode(bndboxes, labels)
        return torch.from_numpy(img), torch.from_numpy(target)

    def encode(self, bboxes, labels):
        """
        encode bboxes vs labels -> [grid_size , grid_size , 5 x num_boxes + C]
        5 : [obj_score,x , y ,w ,h ]
        """
        target = np.zeros((self.grid_size, self.grid_size, 5 *
                          self.nboxes + self.num_classes), dtype=np.float32)
        boxes_xy = (bboxes[:, 2:]+bboxes[:, :2])/2
        boxes_wh = bboxes[:, 2:] - bboxes[:, :2]
        print(boxes_xy)
        for k in range(len(bboxes)):
            xy = boxes_xy[k]
            wh = boxes_wh[k]
            j, i = int((xy[0]) * self.grid_size), int((xy[1])*self.grid_size)
            print(j, i)
            x_cell = xy[0]*self.grid_size - j
            y_cell = xy[1]*self.grid_size - i
            width_cell = wh[0]*self.grid_size
            height_cell = wh[1]*self.grid_size
            if target[i, j, self.num_classes] == 0:
                target[i, j, self.num_classes] = 1
                target[i, j, self.num_classes + 1] = x_cell
                target[i, j, self.num_classes + 2] = y_cell
                target[i, j, self.num_classes + 3] = width_cell
                target[i, j, self.num_classes + 4] = height_cell
                target[i, j, labels[k]] = 1
        return target


def test():
    img_root = "E:/Datasets/VOC2007/train_val/VOCdevkit/VOC2007/JPEGImages"
    anno_dir = "E:/ObjectDetection_PyTorch_Implementation/YOLOV1/voc_2007_annos_txt/train_val"
    dataset = VOC(img_root, anno_dir, num_classes=20,
                  grid_size=7, nboxes=2, train=True)
    print(dataset[0][1][5][0])


if __name__ == '__main__':
    test()
