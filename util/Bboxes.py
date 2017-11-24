
import numpy as np
import tensorflow as tf
from tf_util import get_shape,pad_axis,safe_divide
from tensorflow.python  import debug as tf_debug



#############
#tensorflow implementation
#########################

def tf_bboxes_encode(classes,bboxes,anchors,num_classes,no_annotation_label,ignore_threshold=0.5,prior_scaling=[0.1,0.1,0.2,0.2]):


    def tf_bboxes_encode_layer(labels,
                               bboxes,
                               anchors_layer,
                               num_classes,
                               no_annotation_label,
                               ignore_threshold=0.5,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2],
                               dtype=tf.float32):
        """Encode groundtruth labels and bounding boxes using SSD anchors from
            one layer.

            Arguments:
              labels: 1D Tensor(int64) containing groundtruth labels;
              bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
              anchors_layer: Numpy array with layer anchors;
              matching_threshold: Threshold for positive match with groundtruth bboxes;
              prior_scaling: Scaling of encoded coordinates.

            Return:
              (target_labels, target_localizations, target_scores): Target Tensors.
            """
        # Anchors coordinates and volume.
        yref, xref, href, wref = anchors_layer
        ymin = yref - href / 2.
        xmin = xref - wref / 2.
        ymax = yref + href / 2.
        xmax = xref + wref / 2.
        vol_anchors = (xmax - xmin) * (ymax - ymin)

        # Initialize tensors...
        shape = (yref.shape[0], yref.shape[1], href.size)
        feat_labels = tf.zeros(shape, dtype=tf.int64)
        feat_scores = tf.zeros(shape, dtype=dtype)

        feat_ymin = tf.zeros(shape, dtype=dtype)
        feat_xmin = tf.zeros(shape, dtype=dtype)
        feat_ymax = tf.ones(shape, dtype=dtype)
        feat_xmax = tf.ones(shape, dtype=dtype)

        def jaccard_with_anchors(bbox):
            """Compute jaccard score between a box and the anchors.
            (intersection/union)
            """
            int_ymin = tf.maximum(ymin, bbox[0])
            int_xmin = tf.maximum(xmin, bbox[1])
            int_ymax = tf.minimum(ymax, bbox[2])
            int_xmax = tf.minimum(xmax, bbox[3])
            h = tf.maximum(int_ymax - int_ymin, 0.)
            w = tf.maximum(int_xmax - int_xmin, 0.)
            # Volumes.
            inter_vol = h * w
            union_vol = vol_anchors - inter_vol \
                        + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            jaccard = tf.div(inter_vol, union_vol)
            return jaccard

        def intersection_with_anchors(bbox):
            """Compute intersection between score a box and the anchors.
            """
            int_ymin = tf.maximum(ymin, bbox[0])
            int_xmin = tf.maximum(xmin, bbox[1])
            int_ymax = tf.minimum(ymax, bbox[2])
            int_xmax = tf.minimum(xmax, bbox[3])
            h = tf.maximum(int_ymax - int_ymin, 0.)
            w = tf.maximum(int_xmax - int_xmin, 0.)
            inter_vol = h * w
            scores = tf.div(inter_vol, vol_anchors)
            return scores

        def condition(i, feat_labels, feat_scores,
                      feat_ymin, feat_xmin, feat_ymax, feat_xmax):
            """Condition: check label index.
            """
            r = tf.less(i, tf.shape(labels))
            return r[0]

        def body(i, feat_labels, feat_scores,
                 feat_ymin, feat_xmin, feat_ymax, feat_xmax):
            """Body: update feature labels, scores and bboxes.
            Follow the original SSD paper for that purpose:
              - assign values when jaccard > 0.5;
              - only update if beat the score of other bboxes.
            """
            # Jaccard score.
            label = labels[i]
            bbox = bboxes[i]
            jaccard = jaccard_with_anchors(bbox)
            # Mask: check threshold + scores + no annotations + num_classes.
            mask = tf.greater(jaccard, feat_scores)
            # mask = tf.logical_and(mask, tf.greater(jaccard, matching_threshold))
            mask = tf.logical_and(mask, feat_scores > -0.5)
            mask = tf.logical_and(mask, label < num_classes)
            imask = tf.cast(mask, tf.int64)
            fmask = tf.cast(mask, dtype)
            # Update values using mask.
            feat_labels = imask * label + (1 - imask) * feat_labels
            feat_scores = tf.where(mask, jaccard, feat_scores)

            feat_ymin = fmask * bbox[0] + (1 - fmask) * feat_ymin
            feat_xmin = fmask * bbox[1] + (1 - fmask) * feat_xmin
            feat_ymax = fmask * bbox[2] + (1 - fmask) * feat_ymax
            feat_xmax = fmask * bbox[3] + (1 - fmask) * feat_xmax

            # Check no annotation label: ignore these anchors...
            interscts = intersection_with_anchors(bbox)
            mask = tf.logical_and(interscts > ignore_threshold,
                                  label == no_annotation_label)
            # Replace scores by -1.
            feat_scores = tf.where(mask, -tf.cast(mask, dtype), feat_scores)

            return [i + 1, feat_labels, feat_scores,
                    feat_ymin, feat_xmin, feat_ymax, feat_xmax]

        # Main loop definition.
        i = 0
        [i, feat_labels, feat_scores,
         feat_ymin, feat_xmin,
         feat_ymax, feat_xmax] = tf.while_loop(condition, body,
                                               [i, feat_labels, feat_scores,
                                                feat_ymin, feat_xmin,
                                                feat_ymax, feat_xmax])
        # Transform to center / size.
        feat_cy = (feat_ymax + feat_ymin) / 2.
        feat_cx = (feat_xmax + feat_xmin) / 2.
        feat_h = feat_ymax - feat_ymin
        feat_w = feat_xmax - feat_xmin
        # Encode features.
        feat_cy = (feat_cy - yref) / href / prior_scaling[0]
        feat_cx = (feat_cx - xref) / wref / prior_scaling[1]
        feat_h = tf.log(feat_h / href) / prior_scaling[2]
        feat_w = tf.log(feat_w / wref) / prior_scaling[3]
        # Use SSD ordering: x / y / w / h instead of ours.
        feat_localizations = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)
        return feat_labels, feat_localizations, feat_scores

    with tf.name_scope("bboxes_encode"):
        target_classes = []
        target_scores = []
        target_bboxes = []
        for i, anchors_layer in enumerate(anchors):
            with tf.name_scope('bboxes_encode_block_%i' % i):
                t_labels, t_loc, t_scores = \
                    tf_bboxes_encode_layer(classes, bboxes, anchors_layer,
                                               num_classes, no_annotation_label,
                                               ignore_threshold,
                                               prior_scaling, tf.float32)
                target_classes.append(t_labels)
                target_bboxes.append(t_loc)
                target_scores.append(t_scores)
        return target_classes, target_scores, target_bboxes

def tf_bboxes_decode(localizations,
                         anchors,
                         prior_scaling=[0.1, 0.1, 0.2, 0.2],
                         scope='bboxes_decode'):
    """Compute the relative bounding boxes from the SSD net features and
    reference anchors bounding boxes.

    Arguments:
      feat_localizations: List of Tensors containing localization features.
      anchors: List of numpy array containing anchor boxes.

    Return:
      List of Tensors Nx4: ymin, xmin, ymax, xmax
    """

    def tf_bboxes_decode_layer(localizations_layer,
                                   anchors_layer,
                                   prior_scaling=[0.1, 0.1, 0.2, 0.2]):
        """Compute the relative bounding boxes from the layer features and
        reference anchor bounding boxes.

        Arguments:
          feat_localizations: Tensor containing localization features.
          anchors: List of numpy array containing anchor boxes.

        Return:
          Tensor Nx4: ymin, xmin, ymax, xmax
        """
        yref, xref, href, wref = anchors_layer

        # Compute center, height and width
        cx = localizations_layer[:, :, :, :, 0] * wref * prior_scaling[0] + xref
        cy = localizations_layer[:, :, :, :, 1] * href * prior_scaling[1] + yref
        w = wref * tf.exp(localizations_layer[:, :, :, :, 2] * prior_scaling[2])
        h = href * tf.exp(localizations_layer[:, :, :, :, 3] * prior_scaling[3])
        # Boxes coordinates.
        ymin = cy - h / 2.
        xmin = cx - w / 2.
        ymax = cy + h / 2.
        xmax = cx + w / 2.
        bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
        return bboxes

    with tf.name_scope(scope):
        bboxes = []
        for i, anchors_layer in enumerate(anchors):
            bboxes.append(
                tf_bboxes_decode_layer(localizations[i],
                                           anchors_layer,
                                           prior_scaling))
        return bboxes


def tf_bboxes_select(classifications,localizations,threshold,ignore_class=0):
    """Extract classes, scores and bounding boxes from network output layers.
      Batch-compatible: inputs are supposed to have batch-type shapes.

      Args:
        classifications: List of SSD prediction layers;
        localizations: List of localization layers;
        select_threshold: Classification threshold for selecting a box. All boxes
          under the threshold are set to 'zero'. If None, no threshold applied.
      Return:
        d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
          size Batches X N x 1 | 4. Each key corresponding to a class.
      """

    def tf_bboxes_select_layer(classifications_layer, localizations_layer,
                                   select_threshold=None,
                                   ignore_class=0,
                                   scope=None):
        """Extract classes, scores and bounding boxes from features in one layer.
        Batch-compatible: inputs are supposed to have batch-type shapes.

        Args:
          predictions_layer: A SSD prediction layer;
          localizations_layer: A SSD localization layer;
          select_threshold: Classification threshold for selecting a box. All boxes
            under the threshold are set to 'zero'. If None, no threshold applied.
        Return:
          d_scores, d_bboxes: Dictionary of scores and bboxes Tensors of
            size Batches X N x 1 | 4. Each key corresponding to a class.
        """
        select_threshold = 0.0 if select_threshold is None else select_threshold
        with tf.name_scope(scope, 'bboxes_select_layer',
                           [classifications_layer, localizations_layer]):
            # Reshape features: Batches x N x N_labels | 4
            c_shape = get_shape(classifications_layer)
            classifications_layer = tf.reshape(classifications_layer,
                                           tf.stack([c_shape[0], -1, c_shape[-1]]))
            l_shape = get_shape(localizations_layer)
            localizations_layer = tf.reshape(localizations_layer,
                                             tf.stack([l_shape[0], -1, l_shape[-1]]))

            d_scores = {}
            d_bboxes = {}
            for c in range(c_shape[-1]):
                if c != ignore_class:
                    # Remove boxes under the threshold.
                    scores = classifications_layer[:, :, c]
                    fmask = tf.cast(tf.greater_equal(scores, select_threshold), scores.dtype)
                    scores = scores * fmask
                    bboxes = localizations_layer * tf.expand_dims(fmask, axis=-1)
                    # Append to dictionary.
                    d_scores[c] = scores
                    d_bboxes[c] = bboxes

            return d_scores, d_bboxes

    with tf.name_scope('bboxes_select', 'bboxes_select',
                       [classifications, localizations]):
        l_scores = []
        l_bboxes = []
        for i in range(len(classifications)):
            scores, bboxes = tf_bboxes_select_layer(classifications[i],
                                                    localizations[i],
                                                    threshold,
                                                    ignore_class)
            l_scores.append(scores)
            l_bboxes.append(bboxes)
        # Concat results.
        d_scores = {}
        d_bboxes = {}
        for c in l_scores[0].keys():
            ls = [s[c] for s in l_scores]
            lb = [b[c] for b in l_bboxes]
            d_scores[c] = tf.concat(ls, axis=1)
            d_bboxes[c] = tf.concat(lb, axis=1)
        return d_scores, d_bboxes



def tf_bboxes_sort(classifications,localizations,top_k,scope=None):

    """Sort bounding boxes by decreasing order and keep only the top_k.
    If inputs are dictionnaries, assume every key is a different class.
    Assume a batch-type input.

    Args:
      scores: Batch x N Tensor/Dictionary containing float scores.
      bboxes: Batch x N x 4 Tensor/Dictionary containing boxes coordinates.
      top_k: Top_k boxes to keep.
    Return:
      scores, bboxes: Sorted Tensors/Dictionaries of shape Batch x Top_k x 1|4.
    """
    # Dictionaries as inputs.
    if isinstance(classifications, dict) or isinstance(localizations, dict):
        with tf.name_scope(scope, 'bboxes_sort_dict'):
            d_scores = {}
            d_bboxes = {}
            for c in classifications.keys():
                s, b = tf_bboxes_sort(classifications[c], localizations[c], top_k= top_k )
                d_scores[c] = s
                d_bboxes[c] = b
            return d_scores, d_bboxes

    # Tensors inputs.
    with tf.name_scope(scope, 'bboxes_sort', [classifications, localizations]):
        # Sort scores...
        scores, idxes = tf.nn.top_k(classifications, k=top_k, sorted=True)

        # Trick to be able to use tf.gather: map for each element in the first dim.
        def fn_gather(bboxes, idxes):
            bb = tf.gather(bboxes, idxes)
            return [bb]
        r = tf.map_fn(lambda x: fn_gather(x[0], x[1]),
                      [localizations, idxes],
                      dtype=[localizations.dtype],
                      parallel_iterations=10,
                      back_prop=False,
                      swap_memory=False,
                      infer_shape=True)
        bboxes = r[0]
        return scores, bboxes


def tf_bboxes_nms(scores, bboxes, nms_threshold=0.5, keep_top_k=200, scope=None):
    """Apply non-maximum selection to bounding boxes. In comparison to TF
    implementation, use classes information for matching.
    Should only be used on single-entries. Use batch version otherwise.

    Args:
      scores: N Tensor containing float scores.
      bboxes: N x 4 Tensor containing boxes coordinates.
      nms_threshold: Matching threshold in NMS algorithm;
      keep_top_k: Number of total object to keep after NMS.
    Return:
      classes, scores, bboxes Tensors, sorted by score.
        Padded with zero if necessary.
    """
    with tf.name_scope(scope, 'bboxes_nms_single', [scores, bboxes]):
        # Apply NMS algorithm.
        idxes = tf.image.non_max_suppression(bboxes, scores,
                                             keep_top_k, nms_threshold)
        scores = tf.gather(scores, idxes)
        bboxes = tf.gather(bboxes, idxes)
        # Pad results.
        scores = pad_axis(scores, 0, keep_top_k, axis=0)
        bboxes = pad_axis(bboxes, 0, keep_top_k, axis=0)
        return scores, bboxes

def tf_bboxes_nms_batch(classifications,localizations,nms_threshold,keep_top_k,scope=None):

    if isinstance(classifications, dict) or isinstance(localizations, dict):
        with tf.name_scope(scope, 'bboxes_nms_batch_dict'):
            d_scores = {}
            d_bboxes = {}
            for c in classifications.keys():
                s, b = tf_bboxes_nms_batch(classifications[c], localizations[c],
                                        nms_threshold=nms_threshold,
                                        keep_top_k=keep_top_k)
                d_scores[c] = s
                d_bboxes[c] = b
            return d_scores, d_bboxes

    # Tensors inputs.
    with tf.name_scope(scope, 'bboxes_nms_batch'):
        r = tf.map_fn(lambda x: tf_bboxes_nms(x[0], x[1],
                                           nms_threshold, keep_top_k),
                      (classifications, localizations),
                      dtype=(classifications.dtype, localizations.dtype),
                      parallel_iterations=10,
                      back_prop=False,
                      swap_memory=False,
                      infer_shape=True)
        scores, bboxes = r
        return scores, bboxes


def tf_bboxes_filter(classifications, localizations,
                        select_threshold=None, nms_threshold=0.5,
                        clipping_bbox=None, top_k=400, keep_top_k=200):

    rscores, rbboxes = tf_bboxes_select(classifications,localizations,select_threshold,ignore_class=0)

    rscores, rbboxes = tf_bboxes_sort(rscores,rbboxes,top_k)

    rscores, rbboxes = tf_bboxes_nms_batch(rscores,rbboxes,nms_threshold,keep_top_k=keep_top_k)

    return rscores,rbboxes


def tf_bboxes_jaccard(bbox_ref, bboxes, name=None):
    """Compute jaccard score between a reference box and a collection
    of bounding boxes.

    Args:
      bbox_ref: (N, 4) or (4,) Tensor with reference bounding box(es).
      bboxes: (N, 4) Tensor, collection of bounding boxes.
    Return:
      (N,) Tensor with Jaccard scores.
    """
    with tf.name_scope(name, 'bboxes_jaccard'):
        # Should be more efficient to first transpose.
        bboxes = tf.transpose(bboxes)
        bbox_ref = tf.transpose(bbox_ref)
        # Intersection bbox and volume.
        int_ymin = tf.maximum(bboxes[0], bbox_ref[0])
        int_xmin = tf.maximum(bboxes[1], bbox_ref[1])
        int_ymax = tf.minimum(bboxes[2], bbox_ref[2])
        int_xmax = tf.minimum(bboxes[3], bbox_ref[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        union_vol = -inter_vol \
            + (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1]) \
            + (bbox_ref[2] - bbox_ref[0]) * (bbox_ref[3] - bbox_ref[1])
        jaccard = safe_divide(inter_vol, union_vol, 'jaccard')
        return jaccard



def tf_bboxes_matching(label, scores, bboxes,
                    glabels, gbboxes, gdifficults,
                    matching_threshold=0.5, scope=None):
    """Matching a collection of detected boxes with groundtruth values.
    Does not accept batched-inputs.
    The algorithm goes as follows: for every detected box, check
    if one grountruth box is matching. If none, then considered as False Positive.
    If the grountruth box is already matched with another one, it also counts
    as a False Positive. We refer the Pascal VOC documentation for the details.

    Args:
      rclasses, rscores, rbboxes: N(x4) Tensors. Detected objects, sorted by score;
      glabels, gbboxes: Groundtruth bounding boxes. May be zero padded, hence
        zero-class objects are ignored.
      matching_threshold: Threshold for a positive match.
    Return: Tuple of:
       n_gbboxes: Scalar Tensor with number of groundtruth boxes (may difer from
         size because of zero padding).
       tp_match: (N,)-shaped boolean Tensor containing with True Positives.
       fp_match: (N,)-shaped boolean Tensor containing with False Positives.
    """
    with tf.name_scope(scope, 'bboxes_matching_single',
                       [scores, bboxes, glabels, gbboxes]):
        rsize = tf.size(scores)
        rshape = tf.shape(scores)
        rlabel = tf.cast(label, glabels.dtype)
        # Number of groundtruth boxes.
        gdifficults = tf.cast(gdifficults, tf.bool)
        n_gbboxes = tf.count_nonzero(tf.logical_and(tf.equal(glabels, label),
                                                    tf.logical_not(gdifficults)))
        # Grountruth matching arrays.
        gmatch = tf.zeros(tf.shape(glabels), dtype=tf.bool)
        grange = tf.range(tf.size(glabels), dtype=tf.int32)
        # True/False positive matching TensorArrays.
        sdtype = tf.bool
        ta_tp_bool = tf.TensorArray(sdtype, size=rsize, dynamic_size=False, infer_shape=True)
        ta_fp_bool = tf.TensorArray(sdtype, size=rsize, dynamic_size=False, infer_shape=True)

        # Loop over returned objects.
        def m_condition(i, ta_tp, ta_fp, gmatch):
            r = tf.less(i, rsize)
            return r

        def m_body(i, ta_tp, ta_fp, gmatch):
            # Jaccard score with groundtruth bboxes.
            rbbox = bboxes[i]
            jaccard = tf_bboxes_jaccard(rbbox, gbboxes)
            jaccard = jaccard * tf.cast(tf.equal(glabels, rlabel), dtype=jaccard.dtype)

            # Best fit, checking it's above threshold.
            idxmax = tf.cast(tf.argmax(jaccard, axis=0), tf.int32)
            jcdmax = jaccard[idxmax]
            match = jcdmax > matching_threshold
            existing_match = gmatch[idxmax]
            not_difficult = tf.logical_not(gdifficults[idxmax])

            # TP: match & no previous match and FP: previous match | no match.
            # If difficult: no record, i.e FP=False and TP=False.
            tp = tf.logical_and(not_difficult,
                                tf.logical_and(match, tf.logical_not(existing_match)))
            ta_tp = ta_tp.write(i, tp)
            fp = tf.logical_and(not_difficult,
                                tf.logical_or(existing_match, tf.logical_not(match)))
            ta_fp = ta_fp.write(i, fp)
            # Update grountruth match.
            mask = tf.logical_and(tf.equal(grange, idxmax),
                                  tf.logical_and(not_difficult, match))
            gmatch = tf.logical_or(gmatch, mask)

            return [i+1, ta_tp, ta_fp, gmatch]
        # Main loop definition.
        i = 0
        [i, ta_tp_bool, ta_fp_bool, gmatch] = \
            tf.while_loop(m_condition, m_body,
                          [i, ta_tp_bool, ta_fp_bool, gmatch],
                          parallel_iterations=1,
                          back_prop=False)
        # TensorArrays to Tensors and reshape.
        tp_match = tf.reshape(ta_tp_bool.stack(), rshape)
        fp_match = tf.reshape(ta_fp_bool.stack(), rshape)

        # Some debugging information...
        # tp_match = tf.Print(tp_match,
        #                     [n_gbboxes,
        #                      tf.reduce_sum(tf.cast(tp_match, tf.int64)),
        #                      tf.reduce_sum(tf.cast(fp_match, tf.int64)),
        #                      tf.reduce_sum(tf.cast(gmatch, tf.int64))],
        #                     'Matching (NG, TP, FP, GM): ')
        return n_gbboxes, tp_match, fp_match



def tf_bboxes_matching_batch(labels, scores, bboxes,
                          glabels, gbboxes, gdifficults,
                          matching_threshold=0.5, scope=None):
    """Matching a collection of detected boxes with groundtruth values.
    Batched-inputs version.

    Args:
      rclasses, rscores, rbboxes: BxN(x4) Tensors. Detected objects, sorted by score;
      glabels, gbboxes: Groundtruth bounding boxes. May be zero padded, hence
        zero-class objects are ignored.
      matching_threshold: Threshold for a positive match.
    Return: Tuple or Dictionaries with:
       n_gbboxes: Scalar Tensor with number of groundtruth boxes (may difer from
         size because of zero padding).
       tp: (B, N)-shaped boolean Tensor containing with True Positives.
       fp: (B, N)-shaped boolean Tensor containing with False Positives.
    """
    # Dictionaries as inputs.
    if isinstance(scores, dict) or isinstance(bboxes, dict):
        with tf.name_scope(scope, 'bboxes_matching_batch_dict'):
            d_n_gbboxes = {}
            d_tp = {}
            d_fp = {}
            for c in labels:
                n, tp, fp, _ = tf_bboxes_matching_batch(c, scores[c], bboxes[c],
                                                     glabels, gbboxes, gdifficults,
                                                     matching_threshold)
                d_n_gbboxes[c] = n
                d_tp[c] = tp
                d_fp[c] = fp
            return d_n_gbboxes, d_tp, d_fp, scores

    with tf.name_scope(scope, 'bboxes_matching_batch',
                       [scores, bboxes, glabels, gbboxes]):

        n_gbboxes, tp_match, fp_match = tf.map_fn(lambda x: tf_bboxes_matching(labels, x[0], x[1],
                                                x[2], x[3], x[4],
                                                matching_threshold),
                      (scores, bboxes, glabels, gbboxes, gdifficults),
                      dtype=(tf.int64, tf.bool, tf.bool),
                      parallel_iterations=10,
                      back_prop=False,
                      swap_memory=True,
                      infer_shape=True)
        return n_gbboxes, tp_match, fp_match, scores

def tf_bboxes_resize(bbox_ref, bboxes, name=None):
    """Resize bounding boxes based on a reference bounding box,
    assuming that the latter is [0, 0, 1, 1] after transform. Useful for
    updating a collection of boxes after cropping an image.
    """
    # Bboxes is dictionary.
    if isinstance(bboxes, dict):
        with tf.name_scope(name, 'bboxes_resize_dict'):
            d_bboxes = {}
            for c in bboxes.keys():
                d_bboxes[c] = tf_bboxes_resize(bbox_ref, bboxes[c])
            return d_bboxes

    # Tensors inputs.
    with tf.name_scope(name, 'bboxes_resize'):
        # Translate.
        v = tf.stack([bbox_ref[0], bbox_ref[1], bbox_ref[0], bbox_ref[1]])
        bboxes = bboxes - v
        # Scale.
        s = tf.stack([bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1],
                      bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1]])
        bboxes = bboxes / s
        return bboxes


def tf_bboxes_intersection(bbox_ref, bboxes, name=None):
    """Compute relative intersection between a reference box and a
    collection of bounding boxes. Namely, compute the quotient between
    intersection area and box area.

    Args:
      bbox_ref: (N, 4) or (4,) Tensor with reference bounding box(es).
      bboxes: (N, 4) Tensor, collection of bounding boxes.
    Return:
      (N,) Tensor with relative intersection.
    """
    with tf.name_scope(name, 'bboxes_intersection'):
        # Should be more efficient to first transpose.
        bboxes = tf.transpose(bboxes)
        bbox_ref = tf.transpose(bbox_ref)
        # Intersection bbox and volume.
        int_ymin = tf.maximum(bboxes[0], bbox_ref[0])
        int_xmin = tf.maximum(bboxes[1], bbox_ref[1])
        int_ymax = tf.minimum(bboxes[2], bbox_ref[2])
        int_xmax = tf.minimum(bboxes[3], bbox_ref[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        bboxes_vol = (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1])
        scores = safe_divide(inter_vol, bboxes_vol, 'intersection')
        return scores

def tf_bboxes_filter_overlap(labels, bboxes,
                          threshold=0.5, assign_negative=False,
                          scope=None):
    """Filter out bounding boxes based on (relative )overlap with reference
    box [0, 0, 1, 1].  Remove completely bounding boxes, or assign negative
    labels to the one outside (useful for latter processing...).

    Return:
      labels, bboxes: Filtered (or newly assigned) elements.
    """
    with tf.name_scope(scope, 'bboxes_filter', [labels, bboxes]):
        scores = tf_bboxes_intersection(tf.constant([0, 0, 1, 1], bboxes.dtype),
                                     bboxes)
        mask = scores > threshold
        if assign_negative:
            labels = tf.where(mask, labels, -labels)
            # bboxes = tf.where(mask, bboxes, bboxes)
        else:
            labels = tf.boolean_mask(labels, mask)
            bboxes = tf.boolean_mask(bboxes, mask)
        return labels, bboxes


############################
### Numpy implementation
############################



def bboxes_select(classifications,
                      localizations,
                      anchors,
                      select_threshold=0.5,
                      img_shape=(300, 300),
                      num_classes=21,
                      decode=True):
    """Extract classes, scores and bounding boxes from network output layers.

    Return:
      classes, scores, bboxes: Numpy arrays...
    """
    l_classes = []
    l_scores = []
    l_bboxes = []

    def bboxes_select_layer(classifications,
                            localizations,
                            anchors,
                            select_threshold=0.5,
                            img_shape=(300, 300),
                            num_classes=21,
                            decode=True):
        """Extract classes, scores and bounding boxes from features in one layer.

        Return:
          classes, scores, bboxes: Numpy arrays...
        """
        # First decode localizations features if necessary.

        def bboxes_decode(feat_localizations,
                          anchor_bboxes,
                          prior_scaling=[0.1, 0.1, 0.2, 0.2]):
            """Compute the relative bounding boxes from the layer features and
            reference anchor bounding boxes.

            Return:
              numpy array Nx4: ymin, xmin, ymax, xmax
            """
            # Reshape for easier broadcasting.
            l_shape = feat_localizations.shape
            feat_localizations = np.reshape(feat_localizations,
                                            (-1, l_shape[-2], l_shape[-1]))
            yref, xref, href, wref = anchor_bboxes
            xref = np.reshape(xref, [-1, 1])
            yref = np.reshape(yref, [-1, 1])

            # Compute center, height and width
            cx = feat_localizations[:, :, 0] * wref * prior_scaling[0] + xref
            cy = feat_localizations[:, :, 1] * href * prior_scaling[1] + yref
            w = wref * np.exp(feat_localizations[:, :, 2] * prior_scaling[2])
            h = href * np.exp(feat_localizations[:, :, 3] * prior_scaling[3])
            # bboxes: ymin, xmin, xmax, ymax.
            bboxes = np.zeros_like(feat_localizations)
            bboxes[:, :, 0] = cy - h / 2.
            bboxes[:, :, 1] = cx - w / 2.
            bboxes[:, :, 2] = cy + h / 2.
            bboxes[:, :, 3] = cx + w / 2.
            # Back to original shape.
            bboxes = np.reshape(bboxes, l_shape)
            return bboxes

        if decode:
            localizations = bboxes_decode(localizations, anchors)

        # Reshape features to: Batches x N x N_labels | 4.
        p_shape = classifications.shape
        batch_size = p_shape[0] if len(p_shape) == 5 else 1
        classifications = np.reshape(classifications,
                                     (batch_size, -1, p_shape[-1]))
        l_shape = localizations.shape
        localizations = np.reshape(localizations,
                                   (batch_size, -1, l_shape[-1]))

        # Boxes selection: use threshold or score > no-label criteria.
        if select_threshold is None or select_threshold == 0:
            # Class prediction and scores: assign 0. to 0-class
            classes = np.argmax(classifications, axis=2)
            scores = np.amax(classifications, axis=2)
            mask = (classes > 0)
            classes = classes[mask]
            scores = scores[mask]
            bboxes = localizations[mask]
        else:
            sub_predictions = classifications[:, :, 1:]
            idxes = np.where(sub_predictions > select_threshold)
            classes = idxes[-1] + 1
            scores = sub_predictions[idxes]
            bboxes = localizations[idxes[:-1]]

        return classes, scores, bboxes

    for i in range(len(classifications)):
        #### get class_scores_bboxes layer by layer ####
        classes, scores, bboxes = bboxes_select_layer(
            classifications[i], localizations[i], anchors[i],
            select_threshold, img_shape, num_classes, decode)
        l_classes.append(classes)
        l_scores.append(scores)
        l_bboxes.append(bboxes)


    classes = np.concatenate(l_classes, 0)
    scores = np.concatenate(l_scores, 0)
    bboxes = np.concatenate(l_bboxes, 0)
    return classes, scores, bboxes

def bboxes_sort(classes, scores, bboxes, top_k=400):
    """Sort bounding boxes by decreasing order and keep only the top_k
    """
    # if priority_inside:
    #     inside = (bboxes[:, 0] > margin) & (bboxes[:, 1] > margin) & \
    #         (bboxes[:, 2] < 1-margin) & (bboxes[:, 3] < 1-margin)
    #     idxes = np.argsort(-scores)
    #     inside = inside[idxes]
    #     idxes = np.concatenate([idxes[inside], idxes[~inside]])
    idxes = np.argsort(-scores)
    classes = classes[idxes][:top_k]
    scores = scores[idxes][:top_k]
    bboxes = bboxes[idxes][:top_k]
    return classes, scores, bboxes


def bboxes_clip(bbox_ref, bboxes):
    """Clip bounding boxes with respect to reference bbox.
    """
    bboxes = np.copy(bboxes)
    bboxes = np.transpose(bboxes)
    bbox_ref = np.transpose(bbox_ref)
    bboxes[0] = np.maximum(bboxes[0], bbox_ref[0])
    bboxes[1] = np.maximum(bboxes[1], bbox_ref[1])
    bboxes[2] = np.minimum(bboxes[2], bbox_ref[2])
    bboxes[3] = np.minimum(bboxes[3], bbox_ref[3])
    bboxes = np.transpose(bboxes)
    return bboxes


def bboxes_nms(classes, scores, bboxes, nms_threshold=0.45):
    """Apply non-maximum selection to bounding boxes.
    """
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size-1):
        if keep_bboxes[i]:
            # Computer overlap with bboxes which are following.
            overlap = bboxes_jaccard(bboxes[i], bboxes[(i+1):])
            # Overlap threshold for keeping + checking part of the same class
            keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i+1):] != classes[i])
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)

    idxes = np.where(keep_bboxes)
    return classes[idxes], scores[idxes], bboxes[idxes]

def bboxes_jaccard(bboxes1, bboxes2):
    """Computing jaccard index between bboxes1 and bboxes2.
    Note: bboxes1 and bboxes2 can be multi-dimensional, but should broacastable.
    """
    bboxes1 = np.transpose(bboxes1)
    bboxes2 = np.transpose(bboxes2)
    # Intersection bbox and volume.
    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w
    # Union volume.
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])
    jaccard = int_vol / (vol1 + vol2 - int_vol)
    return jaccard
