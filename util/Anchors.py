import multibox_parameters as mp
import numpy as np
import math
def layers_anchors(img_size,feat_shape,anchor_sizes,anchor_ratios,
                   anchor_steps,offset=0.5,dtype=np.float32):
    """
    generate anchors for one feature layer;

    the anchor will be in shape of (y,x,h,w),where x, y are the relative position grid of the centers
    and h,w are the relative width and height

    :return:
    """
    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = (y.astype(dtype) + offset) * anchor_steps / img_size[0]
    x = (x.astype(dtype) + offset) * anchor_steps / img_size[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    # Compute relative height and width.
    # Tries to follow the original implementation of SSD for the order.
    num_anchors = len(anchor_sizes) + len(anchor_ratios)

    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)

    # Add first anchor boxes with ratio=1.
    h[0] = anchor_sizes[0] / img_size[0]
    w[0] = anchor_sizes[0] / img_size[1]
    di = 1
    if len(anchor_sizes) > 1:
        h[1] = math.sqrt(anchor_sizes[0] * anchor_sizes[1]) / img_size[0]
        w[1] = math.sqrt(anchor_sizes[0] * anchor_sizes[1]) / img_size[1]
        di += 1
    for i, r in enumerate(anchor_ratios):
        h[i+di] = anchor_sizes[0] / img_size[0] / math.sqrt(r)
        w[i+di] = anchor_sizes[0] / img_size[1] * math.sqrt(r)
    return y, x, h, w


def multibox_anchors(img_size=(300,300)):
    """
    Generate all the anchors based on the size of input image
    :param img_size:
    :return:
    """
    multibox_params = mp.get_multibox_parameters(image_size=img_size)

    anchors = []
    for i,shape in enumerate(multibox_params.feat_shapes):
        anchors_bboxes = layers_anchors(img_size,shape,multibox_params.anchor_sizes[i],
                                        multibox_params.anchor_ratios[i],multibox_params.anchor_steps[i],
                                        offset = multibox_params.anchor_offset,dtype=np.float32)
        anchors.append(anchors_bboxes)
    return anchors


if __name__ == "__main__":

    anchors = multibox_anchors()
    print anchors