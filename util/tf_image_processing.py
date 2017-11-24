import numpy as np
import tensorflow as tf
import os
from PIL import Image


def image_whiten(image,means = [123,117,104]):
    """
    Subtracts the mean value
    :param image:  tf value in image style
    :param means:  the means value of three diffrent channals R G B
    :return: the subtracted image in tf tensor
    """
    image = tf.cast(image,tf.float32)
    tf_mean = tf.constant(means,dtype=image.dtype)
    image =  image - tf_mean
    return image

def image_resize(image,size = (300,300)):
    """
    Resize the image into fixed scale
    :param image: tf tensor
    :param size:
    :return: resized image
    """
    image = tf.image.resize_images(image,size=size)

    return image

def image_draw_bboxes(image, bboxes):
    """
    Draw bounding boxes on the image
    :param image: a tensor image in shape of (width,height,channel)
    :param bboxes: in shape of (num of bboxes, [y_min,x_min,y_max,x_max])
    :return: the image drawed with bounding boxes
    """
    ## increase the dimension
    print "tesor image shape", image.get_shape().ndims
    image = tf.expand_dims(image,dim=0)
    bboxes = tf.expand_dims(bboxes,dim=0)

    image_with_box = tf.image.draw_bounding_boxes(image,bboxes)
    ## downsize the dimension
    image_with_box = tf.squeeze(image_with_box,axis=0)
    return image_with_box

def image_show(image):
    """
    show the image in float32 format
    :param image:
    :return:
    """
    image = image.astype(np.uint8)
    image = np.clip(image,0,255)
    img = Image.fromarray(image)
    img.show()



if __name__ == "__main__":

    input_imgs = tf.placeholder(tf.float32,shape=(None,None,3))
    input_bbox = tf.placeholder(tf.float32,shape=(None,4))
    image = image_draw_bboxes(input_imgs,input_bbox)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ##load image and transform into narray format
        imgs_path = "../images/"
        names = os.listdir(imgs_path)
        img = Image.open(imgs_path + names[-5])
        np_img = np.asarray(img,dtype=np.float32)

        ##run the function
        np_boxes = np.array([[0,0,0.2,0.2],[0.1,0.1,0.5,0.5]],np.float32)
        result = sess.run(image, feed_dict={input_imgs:np_img,input_bbox:np_boxes})



        image_show(result)
