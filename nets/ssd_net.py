import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.python.ops import nn
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib.framework.python.ops import variables
from util import multibox_parameters as mp
from util.tf_util import get_shape
import numpy as np
slim = tf.contrib.slim

def abs_smooth(x):
    """Smoothed absolute function. Useful to compute an L1 smooth error.

    Define as:
        x^2 / 2         if abs(x) < 1
        abs(x) - 0.5    if abs(x) > 1
    We use here a differentiable definition using min(x) and abs(x). Clearly
    not optimal, but good enough for our purpose!
    """
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)
    return r
def weighted_smooth_l1(prediction,target,weights):
    '''
    adopt from Tensorflow Object Detection API

        Define as:
        abs(x)^2 / 2         if abs(x) < 1
        abs(x) - 0.5    if abs(x) > 1

    :param prediction:
    :param target:
    :param weights:
    :return:
    '''
    with tf.name_scope("weighted_smooth_l1"):
        diff = prediction - target
        abs_diff = tf.abs(diff)
        abs_diff_lt_1 = tf.less(abs_diff, 1)
        anchorwise_smooth_l1norm = tf.reduce_sum(
            tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5)* weights)
    return anchorwise_smooth_l1norm

@add_arg_scope
def pad2d_layer(inputs, pad=(0,0),scope=None, data_format = 'NHWC'):
    """
    2D padding layer, to mimic padding in Caffe.
    :param inputs: 4D tensor
    :param pad:  2 padding value
    :param scope:
    :return:
    """
    with tf.name_scope(scope,"pad2d",[inputs]):
        paddings = [[0,0],[pad[0],pad[0]],[pad[1],pad[1]],[0,0]]
        net = tf.pad(inputs,paddings)
        return net

@add_arg_scope
def l2_normalization_layer(
        inputs,
        scaling=False,
        scale_initializer=init_ops.ones_initializer(),
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        data_format='NHWC',
        trainable=True,
        scope=None):
    """Implement L2 normalization on every feature (i.e. spatial normalization).

    Should be extended in some near future to other dimensions, providing a more
    flexible normalization framework.

    Args:
      inputs: a 4-D tensor with dimensions [batch_size, height, width, channels].
      scaling: whether or not to add a post scaling operation along the dimensions
        which have been normalized.
      scale_initializer: An initializer for the weights.
      reuse: whether or not the layer and its variables should be reused. To be
        able to reuse the layer scope must be given.
      variables_collections: optional list of collections for all the variables or
        a dictionary containing a different list of collection per variable.
      outputs_collections: collection to add the outputs.
      data_format:  NHWC or NCHW data format.
      trainable: If `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
      scope: Optional scope for `variable_scope`.
    Returns:
      A `Tensor` representing the output of the operation.
    """

    with variable_scope.variable_scope(
            scope, 'L2Normalization', [inputs], reuse=reuse) as sc:
        inputs_shape = inputs.get_shape()
        inputs_rank = inputs_shape.ndims
        dtype = inputs.dtype.base_dtype
        if data_format == 'NHWC':
            # norm_dim = tf.range(1, inputs_rank-1)
            norm_dim = tf.range(inputs_rank-1, inputs_rank)
            params_shape = inputs_shape[-1:]
        elif data_format == 'NCHW':
            # norm_dim = tf.range(2, inputs_rank)
            norm_dim = tf.range(1, 2)
            params_shape = (inputs_shape[1])

        # Normalize along spatial dimensions.
        outputs = nn.l2_normalize(inputs, norm_dim, epsilon=1e-12)
        # Additional scaling.
        if scaling:
            scale_collections = utils.get_variable_collections(
                variables_collections, 'scale')
            scale = variables.model_variable('gamma',
                                             shape=params_shape,
                                             dtype=dtype,
                                             initializer=scale_initializer,
                                             collections=scale_collections,
                                             trainable=trainable)
            if data_format == 'NHWC':
                outputs = tf.multiply(outputs, scale)
            elif data_format == 'NCHW':
                scale = tf.expand_dims(scale, axis=-1)
                scale = tf.expand_dims(scale, axis=-1)
                outputs = tf.multiply(outputs, scale)
                # outputs = tf.transpose(outputs, perm=(0, 2, 3, 1))

        return utils.collect_named_outputs(outputs_collections,
                                           sc.original_name_scope, outputs)


def ssd_arg_scope(weight_decay=0.0005, data_format='NHWC'):
    """Defines the VGG arg scope.

    Args:
      weight_decay: The l2 regularization coefficient.

    Returns:
      An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            data_format=data_format):
            with slim.arg_scope([pad2d_layer,
                                 l2_normalization_layer],
                                data_format=data_format) as sc:
                return sc

def tensor_shape(inputs):

    if inputs.get_shape().is_fully_defined():
        return inputs.get_shape().as_list()
    else:
        raise ValueError("the tensor shape should be fully defined")

@add_arg_scope
def classification_fn(inputs):
    """
    transform the logits into predictions using softmax
    :param inputs:
    :return:
    """
    net = slim.softmax(inputs)

    return net

def multibox_head(inputs,num_classes,anchor_sizes,anchor_ratios,normalization = -1):
    """
    This is a head mounted on each branch for transforming feature layer
    into localization and classification layer
    :return: localizations and classification
    """
    net = inputs
    if normalization > 0:
        net = l2_normalization_layer(net,scaling=True)

    # number of anchors
    num_anchors = len(anchor_ratios) + len(anchor_sizes)

    #num predicting location value in each coordinates
    num_loc_pred = num_anchors * 4
    loc_pred = slim.conv2d(net,num_loc_pred,[3,3],activation_fn=None, scope="conv_loc")
    loc_pred = tf.reshape(loc_pred, tensor_shape(loc_pred)[:-1]+[num_anchors,4])

    #num predicting classification vlaue in each coordinates
    num_cls_pred = num_anchors * num_classes
    cls_pred = slim.conv2d(net,num_cls_pred,[3,3],activation_fn=None, scope="conv_cls")
    cls_pred = tf.reshape(cls_pred, tensor_shape(cls_pred)[:-1]+[num_anchors,num_classes])

    return loc_pred,cls_pred

def multibox_layer(end_points,size = (300,300)):
    """
    Generate several branches from the net trunk, each branch has a multibox head
     which can transform feature layer into localizations and classification layer.
    :param end_points: the collections from the main stream network
    :param size: the mark of this type of ssd
    :return: localizations and classification from each branch
    """
    multibox_params = mp.get_multibox_parameters(size)
    classifications = []
    logits = []
    localizations = []
    for i, layer in enumerate(multibox_params.feat_layers):
        with tf.variable_scope(layer+ "_box"):
            l,c = multibox_head(end_points[layer],num_classes=multibox_params.num_classes,
                                  anchor_sizes=multibox_params.anchor_sizes[i],anchor_ratios=multibox_params.anchor_ratios[i],normalization=multibox_params.normalizations[i])
            logits.append(c)
            classifications.append(classification_fn(c))
            localizations.append(l)
    return localizations,classifications,logits


def ssd_net(image,size=(300,300),batch_norm_trainable=False):
    _dropout_rate = 0.5
    _batch_norm_is_training = batch_norm_trainable
    end_points = {}
    with tf.variable_scope("ssd_300_vgg","ssd_300_vgg",[image],reuse=None,):

        #basic network from VGG-16
        net = slim.repeat(image,2,slim.conv2d,64,[3,3],scope="conv1")
        end_points["block1"] = net
        net = slim.max_pool2d(net,[2,2],scope="pool1")

        net = slim.repeat(net,2,slim.conv2d,128,[3,3],scope="conv2")
        end_points["block2"] = net
        net = slim.max_pool2d(net,[2,2], scope="pool2")

        net = slim.repeat(net,3,slim.conv2d,256,[3,3],scope="conv3")
        end_points["block3"] = net
        net = slim.max_pool2d(net, [2,2], scope="pool3")


        net = slim.repeat(net,3, slim.conv2d,512,[3,3],scope="conv4")
        end_points["block4"] = net
        net = slim.max_pool2d(net, [2,2], scope="pool4")

        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        end_points['block5'] = net
        net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5')

        ## add extra layers for SSD
        net = slim.conv2d(net,1024, [3,3], rate=6, scope='conv6')
        net = slim.batch_norm(net,is_training=_batch_norm_is_training)
        end_points['block6'] = net
        # net = tf.layers.dropout(net,rate=_dropout_rate)

        net = slim.conv2d(net,1024, [1,1], scope="conv7")
        net = slim.batch_norm(net,is_training=_batch_norm_is_training)
        end_points["block7"] = net
        # net = tf.layers.dropout(net,rate=_dropout_rate)

        with tf.variable_scope("block8"):
            net = slim.conv2d(net,256,[1,1],scope="conv1x1")
            net = slim.batch_norm(net, is_training=_batch_norm_is_training)
            net = pad2d_layer(net,pad=(1,1))
            net = slim.conv2d(net,512,[3,3],stride = 2, scope="conv3x3",padding='VALID')
            net = slim.batch_norm(net, is_training=_batch_norm_is_training)
        end_points["block8"] = net

        with tf.variable_scope("block9"):
            net = slim.conv2d(net,128,[1,1],scope="conv1x1")
            net = slim.batch_norm(net, is_training=_batch_norm_is_training)
            net = pad2d_layer(net,pad=(1,1))
            net = slim.conv2d(net,256,[3,3],stride = 2, scope="conv3x3",padding='VALID')
            net = slim.batch_norm(net, is_training=_batch_norm_is_training)
        end_points["block9"] = net

        with tf.variable_scope("block10"):
            net = slim.conv2d(net,128,[1,1], scope="conv1x1")
            net = slim.batch_norm(net, is_training=_batch_norm_is_training)
            net = slim.conv2d(net,256,[3,3], scope="conv3x3",padding='VALID')
            net = slim.batch_norm(net, is_training=_batch_norm_is_training)
        end_points["block10"] = net

        with tf.variable_scope("block11"):
            net = slim.conv2d(net,128,[1,1], scope="conv1x1")
            net = slim.batch_norm(net, is_training=_batch_norm_is_training)
            net = slim.conv2d(net,256,[3,3], scope="conv3x3",padding='VALID')
            net = slim.batch_norm(net, is_training=_batch_norm_is_training)
        end_points["block11"] = net

        localizations, classifications, logits = multibox_layer(end_points,size)
        return localizations, classifications, logits, end_points

def ssd_loss(logits, localisations,
               gclasses, glocalisations, gscores,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.):
    with tf.name_scope('ssd_losses'):
        lshape = get_shape(logits[0], 5)
        num_classes = lshape[-1]
        batch_size = lshape[0]

        # Flatten out all vectors!
        flogits = []
        fgclasses = []
        fgscores = []
        flocalisations = []
        fglocalisations = []
        for i in range(len(logits)):
            flogits.append(tf.reshape(logits[i], [-1, num_classes]))
            fgclasses.append(tf.reshape(gclasses[i], [-1]))
            fgscores.append(tf.reshape(gscores[i], [-1]))
            flocalisations.append(tf.reshape(localisations[i], [-1, 4]))
            fglocalisations.append(tf.reshape(glocalisations[i], [-1, 4]))
        # And concat the crap!
        logits = tf.concat(flogits, axis=0)
        gclasses = tf.concat(fgclasses, axis=0)
        gscores = tf.concat(fgscores, axis=0)
        localisations = tf.concat(flocalisations, axis=0)
        glocalisations = tf.concat(fglocalisations, axis=0)
        dtype = logits.dtype

        # Compute positive matching mask...
        pmask = gscores > match_threshold
        fpmask = tf.cast(pmask, dtype)
        n_positives = tf.reduce_sum(fpmask)

        # Hard negative mining... select 3:1 negtive:positive anchors
        no_classes = tf.cast(pmask, tf.int32)
        predictions = slim.softmax(logits)
        nmask = tf.logical_and(tf.logical_not(pmask),
                               gscores > -0.5)
        fnmask = tf.cast(nmask, dtype)
        nvalues = tf.where(nmask,
                           predictions[:, 0],
                           1. - fnmask)
        nvalues_flat = tf.reshape(nvalues, [-1])
        # Number of negative entries to select.
        max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
        n_neg = tf.cast(negative_ratio * n_positives, tf.int32) + batch_size
        n_neg = tf.minimum(n_neg, max_neg_entries)

        val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
        max_hard_pred = -val[-1]
        # Final negative mask.
        nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
        fnmask = tf.cast(nmask, dtype)

        # Add cross-entropy loss.
        with tf.name_scope('cross_entropy_pos'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=gclasses)
            loss = tf.div(tf.reduce_sum(loss * fpmask), batch_size, name='value')
            tf.losses.add_loss(loss)

        with tf.name_scope('cross_entropy_neg'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=no_classes)
            loss = tf.div(tf.reduce_sum(loss * fnmask), batch_size, name='value')
            tf.losses.add_loss(loss)

        # Add localization loss: smooth L1, L2, ...
        with tf.name_scope('localization'):
            # Weights Tensor: positive mask + random negative.
            weights = tf.expand_dims(alpha * fpmask, axis=-1)
            ## from API
            loss = weighted_smooth_l1(localisations,glocalisations,weights)
            loss = tf.div(loss,batch_size,name='value')
            ###normalization by num_matched
            ##original loss
            # loss = abs_smooth(localisations - glocalisations)
            # loss = tf.div(tf.reduce_sum(loss * weights), batch_size, name='value')
            tf.losses.add_loss(loss)
        with tf.name_scope("normalization"):
            loss = tf.div(tf.losses.get_total_loss(),n_positives+1)
            tf.losses.add_loss(loss)