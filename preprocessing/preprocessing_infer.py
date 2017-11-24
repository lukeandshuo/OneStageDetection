from util import tf_image_processing
import os
import tensorflow as tf
import numpy as np
from PIL import Image

def preprocessing_fn(image,size = (300,300)):
    """
    conduct a preprocssing step for SSD inference, it includes image whiten and resize
    :param image: should be (None,None,None,3), it's better to have a float32 dtype
    :param size: default is 300
    :return: the preprocessing tensor (None,None,3)
    """
    whiten_imgs = tf_image_processing.image_whiten(image)
    resized_imgs = tf_image_processing.image_resize(whiten_imgs,size)
    return resized_imgs

if __name__ == "__main__":

    size=(300,300)
    input_imgs = tf.placeholder(tf.float32,shape=(None,None,3))
    expand_imgs = tf.expand_dims(input_imgs,0)
    resized_imgs = preprocessing_fn(expand_imgs,size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ##load image and transform into narray format
        imgs_path = "../images/"
        names = sorted(os.listdir(imgs_path))
        img = Image.open(imgs_path + names[-5])
        np_img = np.asarray(img,dtype=np.float32)

        ##run the function
        result = sess.run(resized_imgs, feed_dict={input_imgs:np_img})

        tf_image_processing.image_show(result)