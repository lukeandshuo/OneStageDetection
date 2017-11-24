import numpy as np
import tensorflow as tf
from Bboxes import tf_bboxes_resize,tf_bboxes_filter_overlap
from tensorflow.python.framework import ops,tensor_shape
from tensorflow.python.ops import array_ops,check_ops,control_flow_ops,math_ops,random_ops,variables
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

def image_resize(image,size = (300,300),method=tf.image.ResizeMethod.BILINEAR,align_corners=False):
    """
    Resize the image into fixed scale
    :param image: tf tensor
    :param size:
    :return: resized image
    """
    image = tf.image.resize_images(image,size=size,method=method,align_corners=align_corners)

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

def distorted_bounding_box_crop(image,
                                labels,
                                bboxes,
                                min_object_covered=0.3,
                                aspect_ratio_range=(0.5, 2),
                                area_range=(0.1, 1.0),
                                max_attempts=400,
                                clip_bboxes=True,
                                scope=None):
    """Generates cropped_image using a one of the bboxes randomly distorted.

    See `tf.image.sample_distorted_bounding_box` for more documentation.

    Args:
        image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
            where each coordinate is [0, 1) and the coordinates are arranged
            as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
            image.
        min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
            area of the image must contain at least this fraction of any bounding box
            supplied.
        aspect_ratio_range: An optional list of `floats`. The cropped area of the
            image must have an aspect ratio = width / height within this range.
        area_range: An optional list of `floats`. The cropped area of the image
            must contain a fraction of the supplied image within in this range.
        max_attempts: An optional `int`. Number of attempts at generating a cropped
            region of the image of the specified constraints. After `max_attempts`
            failures, return the entire image.
        scope: Optional scope for name_scope.
    Returns:
        A tuple, a 3-D Tensor cropped_image and the distorted bbox
    """
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bboxes]):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].
        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
                tf.shape(image),
                bounding_boxes=tf.expand_dims(bboxes, 0),
                min_object_covered=min_object_covered,
                aspect_ratio_range=aspect_ratio_range,
                area_range=area_range,
                max_attempts=max_attempts,
                use_image_if_no_bounding_boxes=True)
        distort_bbox = distort_bbox[0, 0]

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        # Restore the shape since the dynamic slice loses 3rd dimension.
        cropped_image.set_shape([None, None, 3])

        # Update bounding boxes: resize and filter out.
        bboxes = tf_bboxes_resize(distort_bbox, bboxes)
        labels, bboxes = tf_bboxes_filter_overlap(labels, bboxes)
        return cropped_image, labels, bboxes

def image_flip_left_right(image,bboxes):
    def flip_bboxes(bboxes):
        """Flip bounding boxes coordinates.
        """
        bboxes = tf.stack([bboxes[:, 0], 1 - bboxes[:, 3],
                           bboxes[:, 2], 1 - bboxes[:, 1]], axis=-1)
        return bboxes

    # Random flip. Tensorflow implementation.
    with tf.name_scope('lip_left_right'):
        image = ops.convert_to_tensor(image, name='image')
        # _Check3DImage(image, require_static=False)

        # Flip image.
        image_flip = array_ops.reverse_v2(image, [1])

        # Flip bboxes.
        bboxes =  flip_bboxes(bboxes)

        def fix_image_flip_shape(image, result):
            """Set the shape to 3 dimensional if we don't know anything else.
            Args:
              image: original image size
              result: flipped or transformed image
            Returns:
              An image whose shape is at least None,None,None.
            """
            image_shape = image.get_shape()
            if image_shape == tensor_shape.unknown_shape():
                result.set_shape([None, None, None])
            else:
                result.set_shape(image_shape)
            return result

        return fix_image_flip_shape(image, image_flip), bboxes

def random_flip_left_right(image, bboxes, seed=None):
    """Random flip left-right of an image and its bounding boxes.
    """
    def flip_bboxes(bboxes):
        """Flip bounding boxes coordinates.
        """
        bboxes = tf.stack([bboxes[:, 0], 1 - bboxes[:, 3],
                           bboxes[:, 2], 1 - bboxes[:, 1]], axis=-1)
        return bboxes

    # Random flip. Tensorflow implementation.
    with tf.name_scope('random_flip_left_right'):
        image = ops.convert_to_tensor(image, name='image')
        # _Check3DImage(image, require_static=False)
        uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
        mirror_cond = math_ops.less(uniform_random, .5)
        # Flip image.
        result = control_flow_ops.cond(mirror_cond,
                                       lambda: array_ops.reverse_v2(image, [1]),
                                       lambda: image)
        # Flip bboxes.
        bboxes = control_flow_ops.cond(mirror_cond,
                                       lambda: flip_bboxes(bboxes),
                                       lambda: bboxes)

        def fix_image_flip_shape(image, result):
            """Set the shape to 3 dimensional if we don't know anything else.
            Args:
              image: original image size
              result: flipped or transformed image
            Returns:
              An image whose shape is at least None,None,None.
            """
            image_shape = image.get_shape()
            if image_shape == tensor_shape.unknown_shape():
                result.set_shape([None, None, None])
            else:
                result.set_shape(image_shape)
            return result

        return fix_image_flip_shape(image, result), bboxes


def apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].

    Args:
        x: input Tensor.
        func: Python function to apply.
        num_cases: Python int32, number of cases to sample sel from.

    Returns:
        The result of func(x, sel), where func receives the value of the
        selector as a python integer, but sel is sampled dynamically.
    """
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
            func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
            for case in range(num_cases)])[0]

def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    """Distort the color of a Tensor image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    Args:
        image: 3-D Tensor containing single image in [0, 1].
        color_ordering: Python int, a type of distortion (valid values: 0-3).
        fast_mode: Avoids slower ops (random_hue and random_contrast)
        scope: Optional scope for name_scope.
    Returns:
        3-D Tensor color-distorted image on range [0, 1]
    Raises:
        ValueError: if color_ordering not in [0, 3]
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')
        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 1.0)

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
