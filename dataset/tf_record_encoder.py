from __future__ import division
import tensorflow as tf
import numpy as np
import os
import random
from PIL import Image, ImageDraw
from util.dataset_util import int64_feature, int64_list_feature, float_feature, bytes_feature, bytes_list_feature

RANDOM_SEED = 1024
IMAGE_SHAPE = [512,640,3] ##Height, Width, Depth
DATASET_PATH = "../KAIST"
ANNOTATION_NAME ="annotations"

LABEL_MAP = {"person":1,
             "people":2,
             "cyclist":2,
             "person?":2
             }
def _difficult_condition(line):

    label = line[0]
    bbox = [int(i) for i in line[1:5]]
    occ = int(line[5])
    if label != 'person' or bbox[3]<50 or occ >0:
        return 1
    else:
        return 0

def _annotation_paser(annotation_dir):
    labels_text = []
    labels =[]
    bboxes = []
    difficults = []
    with open(annotation_dir) as f:
        for line in f:
            line = line.strip().split()
            if line[0] == "%":
                continue
            else:
                difficult = _difficult_condition(line)
                box = [int(i) for i in line[1:5]]
                ##bbox format "xywh"
                ## convert to "xmin,ymin, xmax, ymax"
                x,y,w,h = box
                xmin = float(x)/float(IMAGE_SHAPE[1])
                ymin = float(y)/float(IMAGE_SHAPE[0])
                xmax = np.minimum(1.0,float(x + w)/float(IMAGE_SHAPE[1]))
                ymax = np.minimum(1.0,float(y + h)/float(IMAGE_SHAPE[0]))
                box = [ymin,xmin,ymax,xmax]

                bboxes.append(box)
                labels_text.append("person".encode("ascii"))
                labels.append(int(LABEL_MAP[line[0]]))
                difficults.append(difficult)
    return labels_text ,labels,difficults , bboxes

def _write_to_tf_record(image_dir,annotation_dir,tf_record_writer):

    image_data = tf.gfile.FastGFile(image_dir,'r').read()

    labels_text, labels,difficults, bboxes = _annotation_paser(annotation_dir)

    xmin = []
    ymin = []
    xmax = []
    ymax = []

    # split the bboxes into individual
    for box in bboxes:
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], box)]

    example = tf.train.Example(features=tf.train.Features(feature = {
        'image/height': int64_feature(IMAGE_SHAPE[0]),
        'image/width': int64_feature(IMAGE_SHAPE[1]),
        'image/channels': int64_feature(IMAGE_SHAPE[2]),
        'image/shape': int64_list_feature(IMAGE_SHAPE),
        'image/object/bbox/xmin': float_feature(xmin),
        'image/object/bbox/xmax': float_feature(xmax),
        'image/object/bbox/ymin': float_feature(ymin),
        'image/object/bbox/ymax': float_feature(ymax),
        'image/object/bbox/label': int64_list_feature(labels),
        'image/object/bbox/label_text': bytes_list_feature(labels_text),
        'image/object/bbox/difficult': int64_list_feature(difficults),
        'image/format': bytes_feature(b"JPEG"),
        'image/encoded': bytes_feature(image_data)}))

    tf_record_writer.write(example.SerializeToString())

    ######Visualization Test#############
    # if lables != []:
    #     with Image.open(image_dir) as img:
    #         draw = ImageDraw.Draw(img)
    #         draw.rectangle([(bboxes[0][0],bboxes[0][1]),(bboxes[0][2],bboxes[0][3])])
    #         draw.text((bboxes[0][0]-10,bboxes[0][1]-10),lables[0])
    #         print img.size
    #         img.show()

def encoding(imagelist_dir, output_dir="../KAIST", stage = "train", img_type = "visible",shuffling = False, skip = 20):

    if not tf.gfile.Exists(imagelist_dir):
        raise Exception("no such imagelist path")
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MkDir(output_dir)

    imagelist = []
    with open(imagelist_dir) as f:
     imagelist = [ i.strip() for i in f]

    ### sample data
    imagelist = [v for i, v in enumerate(imagelist) if i%skip==0]

    ### shulffe data
    if shuffling:
        random.seed(RANDOM_SEED)
        random.shuffle(imagelist)
    imagelist_dir ,_ = os.path.splitext(imagelist_dir)
    file_name = imagelist_dir.split('/')[-1]
    tf_file_name = "{}/{}_skip{}.tf".format(output_dir,file_name,skip)

    with tf.python_io.TFRecordWriter(tf_file_name) as tf_record_writer:
        for index, name in enumerate(imagelist):
            if index%100==0:
                print "{}/{}".format(index,len(imagelist))
            name = name.split('/')
            image_dir = os.path.join(DATASET_PATH,name[0],name[1],img_type,name[2]+".jpg")
            annotation_dir = os.path.join(DATASET_PATH,ANNOTATION_NAME,name[0],name[1],name[2]+".txt")
            _write_to_tf_record(image_dir,annotation_dir,tf_record_writer)


    print tf_file_name

if __name__ == "__main__":

    encoding("../KAIST/imageSets/test01_valid.txt","../TF_DATA",shuffling=True,skip=20)