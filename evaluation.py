import tensorflow as tf
import numpy as np
import math
from dataset.tf_record_decoder import get_dataset, NUM_SAMPLES, NUM_CLASSES
from preprocessing.preprocessing_infer import preprocessing_fn
from util.multibox_parameters import ssd_300_multibox_parameters
from util.tf_util import reshape_list
from util.Bboxes import tf_bboxes_encode
from util.Anchors import multibox_anchors
from nets.ssd_net import ssd_net,ssd_arg_scope,ssd_loss
from util.Bboxes import tf_bboxes_decode,tf_bboxes_filter,tf_bboxes_matching_batch
from util.evaluation_metrics import streaming_tp_fp_arrays,precision_recall,average_precision_voc07,average_precision_voc12
slim = tf.contrib.slim
tf.logging.set_verbosity(tf.logging.DEBUG)


def main(dataset_dir = "TF_DATA/test01_valid_skip20.tf",checkpoint_path ="experiments/1/model/model.ckpt-127668",eval_dir= "log/eval",batch_size = 6,has_difficult = False):

    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()
        ##########
        #Prepare Dataset
        ##########
        dataset = get_dataset(dataset_dir)
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers= 8,
            common_queue_capacity=20 *batch_size,
            common_queue_min=10 * batch_size,
            shuffle=True)
        [image, shape, labels, bboxes] = provider.get(['image', 'shape', 'object/label', 'object/bbox'])

        if has_difficult:
            [difficults] = provider.get(['object/difficult'])
        else:
            difficults = tf.zeros(tf.shape(labels), dtype=tf.int64)

        params = ssd_300_multibox_parameters()
        image_shape = params.img_shape
        image = preprocessing_fn(image, image_shape)

        ####### anchors
        anchors = multibox_anchors(image_shape)
        classes, scores, localizations = tf_bboxes_encode(labels, bboxes, anchors, params.num_classes,
                                                    params.no_annotation_label)
        ####### batches
        batch = tf.train.batch(reshape_list([image, labels, bboxes, difficults, classes, scores, localizations]), batch_size=batch_size, num_threads=8,dynamic_pad=True)
        batch_shape = [1]*4 + [len(anchors)] * 3
        b_image, b_labels,b_bboxes,b_difficults,b_classes, b_scores,b_localizations = reshape_list(batch, batch_shape)

        ###########
        # Run Networks
        ###########
        with slim.arg_scope(ssd_arg_scope()):
            p_localizations, p_classifications, logits, end_points = ssd_net(b_image,image_shape)

        p_localizations = tf_bboxes_decode(p_localizations,anchors)
        p_classifications, p_localizations= tf_bboxes_filter(p_classifications,p_localizations,select_threshold=0.5,nms_threshold=0.7)


        #########
        # Construct Metrics
        ##########
        # FP and TP metrics.

        num_gbboxes,tp,fp,r_scores = tf_bboxes_matching_batch(p_classifications.keys(),p_classifications,p_localizations,b_labels,b_bboxes,b_difficults)
        dict_metrics={}
        tp_fp_metric = streaming_tp_fp_arrays(num_gbboxes, tp, fp, r_scores)
        for c in tp_fp_metric[0].keys():
            dict_metrics['tp_fp_%s' % c] = (tp_fp_metric[0][c],
                                            tp_fp_metric[1][c])

        # Add to summaries precision/recall values.
        aps_voc07 = {}
        aps_voc12 = {}
        for c in tp_fp_metric[0].keys():
            # Precison and recall values.
            prec, rec = precision_recall(*tp_fp_metric[0][c])

            # Average precision VOC07.
            v = average_precision_voc07(prec, rec)
            summary_name = 'AP_VOC07/%s' % c
            op = tf.summary.scalar(summary_name, v, collections=[])
            # op = tf.Print(op, [v], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
            aps_voc07[c] = v

            # Average precision VOC12.
            v = average_precision_voc12(prec, rec)
            summary_name = 'AP_VOC12/%s' % c
            op = tf.summary.scalar(summary_name, v, collections=[])
            # op = tf.Print(op, [v], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
            aps_voc12[c] = v

        # Mean average precision VOC07.
        summary_name = 'AP_VOC07/mAP'
        mAP = tf.add_n(list(aps_voc07.values())) / len(aps_voc07)
        op = tf.summary.scalar(summary_name, mAP, collections=[])
        op = tf.Print(op, [mAP], summary_name)
        tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        # Mean average precision VOC12.
        summary_name = 'AP_VOC12/mAP'
        mAP = tf.add_n(list(aps_voc12.values())) / len(aps_voc12)
        op = tf.summary.scalar(summary_name, mAP, collections=[])
        op = tf.Print(op, [mAP], summary_name)
        tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)


        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(dict_metrics)
        ###############
        # basic setting
        ###############
        num_batches = math.ceil(dataset.num_samples / float(batch_size))
        print "num_batches:",num_batches,"num_samples:",dataset.num_samples
        variables_to_restore = slim.get_variables_to_restore()
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)

        ##############
        # kick off the evaluation
        ##############

        slim.evaluation.evaluate_once(
            "",
            checkpoint_path=checkpoint_path,
            logdir=eval_dir,
            num_evals=num_batches,
            eval_op=list(names_to_updates.values()),
            variables_to_restore=variables_to_restore,
            session_config=config)
if __name__ == "__main__":
    main()