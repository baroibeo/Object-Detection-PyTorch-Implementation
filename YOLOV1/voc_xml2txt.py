"""
This code to simplify voc annotations from ".xml" to ".txt" with format
file_name img_height img_width num_bboxes [xmin_1,ymin_1,xmax_1,ymax_1,class_box_1] [xmin_2,ymin_2,xmax_2,ymax_2,class_box_2] ...
[xmin,ymin,xmax,ymax] could be kept as their original value or could be normalized based on img size
"""

import os
import argparse
import xml.etree.ElementTree as ET
from tqdm import tqdm

VOC_CLASSES = (
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annos_dir",help="Directory contains xml annotations")
    parser.add_argument("--output_dir",help="Path for saving simplified txt annotation")
    parser.add_argument("--normalize_box",type=bool,default=True,help="To decide whether save x_coords/img_width , y_coords/img_height or keep as the originals")
    return parser.parse_args()

def parse_xml(annos_dir, xml_filename, normalize_box = True):
    """
    annos_dir : dir contains xml anno files
    xml_filename : specifi xml file name
    normalize_box : with the use described in the arguments function
    output will return a string with the format above in the txt file output 
    """
    anno_txt = ""
    anno_txt += xml_filename+".jpg" + " "
    xml_path = os.path.join(annos_dir, xml_filename+".xml")
    tree = ET.parse(xml_path)

    #Get img_height, img_width
    size = tree.find('size')
    img_height = size.find('height').text
    img_width = size.find('width').text
    anno_txt += img_height + " " + img_width + " "

    #Get num boxes
    num_boxes = len(tree.findall('object'))
    anno_txt += str(num_boxes) + " "

    #Get bndboxes properties
    for obj in tree.findall('object'):
        class_idx = VOC_CLASSES.index(obj.find('name').text)
        bndbox = obj.find('bndbox')
        xmin = bndbox.find('xmin').text
        ymin = bndbox.find('ymin').text
        xmax = bndbox.find('xmax').text
        ymax = bndbox.find('ymax').text

        if normalize_box == True:
            xmin = str(float(xmin)/float(img_width))
            ymin = str(float(ymin)/float(img_height))
            xmax = str(float(xmax)/float(img_width))
            ymax = str(float(ymax)/float(img_height))

        anno_txt += xmin + " " + ymin + " " + xmax + " " + ymax + " " + str(class_idx) + " "

    return anno_txt


if __name__=='__main__':
    args = arguments()
    xml_filenames = os.listdir(args.annos_dir)
    for xml_filename in tqdm(xml_filenames):
        xml_filename = xml_filename.split('.')[0]
        anno_txt = parse_xml(args.annos_dir,xml_filename,args.normalize_box)
        with open(args.output_dir +"/" + xml_filename+".txt",'w') as f:
            f.write(anno_txt)
        f.close()
    

    # python voc_xml2txt.py --annos_dir=E:\Datasets\VOC2007\train_val\VOCdevkit\VOC2007\Annotations\ --output_dir=voc_2007_annos_txt\train_val\
    # python voc_xml2txt.py --annos_dir=E:\Datasets\VOC2007\test\VOCdevkit\VOC2007\Annotations\ --output_dir=voc_2007_annos_txt\test\


    



