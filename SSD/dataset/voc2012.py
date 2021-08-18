import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)

def parse_xml(xml_path):
    """
    convert infos in xml file into an object contains : 
    img_name width height num_boxes [box_1] [box_2] ...
    [box] : xmin ymin xmax ymax label (class index)
    """
    tree = ET.parse(xml_path)
    infos = {}
    infos["filename"] = tree.find("filename").text
    size = tree.find("size")
    infos["width"] = size.find("width").text
    infos["height"] = size.find("height").text
    infos["num_boxes"] = len(tree.findall("object"))
    for i,obj in enumerate(tree.findall("object")):
        class_name = obj.find("name").text.lower().strip()
        label = VOC_CLASSES.index(class_name)
        xmin = int(obj.find("bndbox").find("xmin").text)
        ymin = int(obj.find("bndbox").find("ymin").text)
        xmax = int(obj.find("bndbox").find("xmax").text)
        ymax = int(obj.find("bndbox").find("ymax").text)
        infos["box_"+str(i)] = [xmin,ymin,xmax,ymax,label]
    
    return infos

class VOC(Dataset):
    def __init__(self,img_root,anno_root,transform = None):
        self.img_root = img_root
        self.anno_root = anno_root
        self.anno_lists = os.listdir(anno_root)
        self.transform = transform
    
    def __len__(self):
        return len(self.anno_lists)
    
    def __getitem__(self,idx):
        anno_path = self.anno_lists[idx]
        anno = parse_xml(os.path.join(self.anno_root,anno_path))
        img_filename = anno["filename"]
        img = cv2.imread(os.path.join(self.img_root,img_filename))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        boxes = []
        for i in range(anno["num_boxes"]):
            boxes.append(anno["box_"+str(i)])
        boxes = torch.Tensor(boxes)
        img = VOC.visualize_bndbox(img,boxes)
        plt.imshow(img)
        plt.show( )
    
    @staticmethod
    def visualize_bndbox(img,bndboxes,coord_mode = ""):
        """
        img : np.arrray input image needed to visualize
        bndboxes : shape (n_boxes,4)
        coord_mode = "normalized" that means xmin = original_xmin / img_width, ymin = original_ymin / img_width,...
        coord_mode = "other" visualize as original shape
        return img after visualized
        """
        if (len(bndboxes.shape)) == 1:
            bndboxes = bndboxes.view(1,-1)
        if coord_mode == "normalized":
            bndboxes[:,0] *= img.shape[1]
            bndboxes[:,1] *= img.shape[0]
            bndboxes[:,2] *= img.shape[1]
            bndboxes[:,3] *= img.shape[0]
        
        for box in bndboxes:
            if torch.is_tensor(box):
                box = box.cpu().detach().numpy()
            print(box)
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            pts = np.array([[box[0],box[1]],[box[0]+box_width,box[1]],[box[0]+box_width,box[1]+box_height],[box[0],box[1]+box_height]],dtype=np.int32)
            isClosed = True
            color = (255,0,0)
            thickness = 1 
            img = cv2.polylines(img,[pts],isClosed,color,thickness)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,VOC_CLASSES[int(box[4])],(box[0],box[1]),font,0.5,(0,0,0),1)
        
        return img

def test():
    xml_path = "D:\\datasets\\VOC2012\\VOCdevkit\\VOC2012\\Annotations\\2007_000027.xml"
    print(parse_xml(xml_path))

    img_root = "D:\\datasets\\VOC2012\\VOCdevkit\\VOC2012\\JPEGImages"
    anno_root = "D:\\datasets\\VOC2012\\VOCdevkit\\VOC2012\\Annotations"
    ds = VOC(img_root,anno_root)
    ds[1]

if __name__ == "__main__":
    test()
