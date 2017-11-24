import tensorflow as tf
from dataset.tf_record_decoder import get_dataset, NUM_SAMPLES
from util.multibox_parameters import ssd_300_multibox_parameters
from util.Anchors import multibox_anchors
from preprocessing.preprocessing_infer import preprocessing_fn
from util.Bboxes import tf_bboxes_encode
from util.tf_util import reshape_list
from nets.ssd_net import ssd_net as SN
from nets.ssd_net import ssd_arg_scope,ssd_loss
import numpy as np
slim = tf.contrib.slim
import cv2



def main(dataset_dir, log_dir, batch_size, lr_init=0.001,lr_decay_epoch=5,lr_decay_factor=0.94):
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():

        ### Decode TF data ###
        dataset = get_dataset(dataset_dir)
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers= 8,
            common_queue_capacity=20 *batch_size,
            common_queue_min=10 * batch_size,
            shuffle=True)



        #########for verifying the data provider
        # with tf.Session() as sess:
        #     coord = tf.train.Coordinator()
        #     threads = tf.train.start_queue_runners(coord=coord)
        #     for i in range(20):

        [image, labels, bboxes] = provider.get(['image', 'object/label', 'object/bbox'])
        ### get network parameters ###
        params = ssd_300_multibox_parameters()
        img_shape = params.img_shape

        # ###preprocessing ###
        # ###TODO + preprocessing_train (flip, crop, etc.)
        image = preprocessing_fn(image, img_shape)
        #
        # ###TODO encode labels and bboxes based on anchors
        anchors = multibox_anchors(img_shape)
        classes, scores, bboxes = tf_bboxes_encode(labels, bboxes, anchors, params.num_classes,
                                                    params.no_annotation_label)
        # classes, scores, bboxes = sess.run([classes, scores, bboxes])
        # ###TODO prepare batch queue
        batch = tf.train.batch(reshape_list([image,classes,scores,bboxes]),batch_size=batch_size,num_threads=8)
        batch_shape = [1] + [len(anchors)] * 3
        b_image, b_classes, b_scores,b_bboxes = reshape_list(batch,batch_shape)
        batch_queue = slim.prefetch_queue.prefetch_queue(reshape_list([b_image, b_classes, b_scores,b_bboxes]))
        b_image, b_classes, b_scores, b_bboxes = reshape_list(batch_queue.dequeue(),batch_shape)
        # img = sess.run(b_image)
        # for j in range(img.shape[0]):
        #     cv2.imshow("res",img[j,...])
        #     cv2.waitKey(500)
        # ### TODO import arg_scope, construct net model, and calculate loss
        with slim.arg_scope(ssd_arg_scope()):
            localizations, classifications, logits, end_points = SN(b_image,img_shape,True)
        ssd_loss(logits,localizations,b_classes,b_bboxes,b_scores)
        # sess.run(tf.global_variables_initializer())
        # pred = sess.run(localizations)
        # print pred
        ####  create train_op and then run train #########

        total_loss = tf.losses.get_total_loss()
        tf.summary.scalar("loss",total_loss)


        ###set exponential learning rate######
        global_step = slim.create_global_step()
        decay_steps = int(NUM_SAMPLES/batch_size*lr_decay_epoch)
        learning_rate = tf.train.exponential_decay(lr_init,global_step,decay_steps,lr_decay_factor,staircase=True,name="exponential_decay_lr")

        # optimizer = tf.train.RMSPropOptimizer(0.001,0.9)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.9,beta2=0.999,epsilon=1.0)
        train_op = slim.learning.create_train_op(total_loss,optimizer,summarize_gradients=True)


        ###kick off the train
        gpu_options = tf.GPUOptions(allow_growth = True)
        config = tf.ConfigProto(log_device_placement=False,
                                gpu_options=gpu_options)

        slim.learning.train(train_op,log_dir,save_summaries_secs=60,global_step = global_step,session_config=config,number_of_steps=130000,init_fn=get_init_fn())

                ########test###########
                # width = 640
                # height = 512
                # for j in range(bboxes.shape[0]):
                #     print bboxes[j]
                #     cv2.putText(img,str(labels[j]),(int(bboxes[j][1]*width),int(bboxes[j][0]*height)),1,2,color=(0,255,0))
                #     cv2.rectangle(img,(int(bboxes[j][1]*width),int(bboxes[j][0]*height)),(int(bboxes[j][3]*width),int(bboxes[j][2]*height)),color=(0,255,255))
                # cv2.imshow("res",img)
                # cv2.waitKey(300)
            #     print "-----"
            # coord.request_stop()
            # coord.join(threads)
def get_init_fn(checkpoint_path = "checkpoints/vgg_16.ckpt",train_dir="log",
                checkpoint_exclude_scopes = "ssd_300_vgg/BatchNorm,ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,\
                ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,\
                ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box",
                checkpoint_model_scope="vgg_16",model_name= "ssd_300_vgg",ignore_missing_vars=False):
    """Returns a function run by the chief worker to warm-start the training.
    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Returns:
      An init function run by the supervisor.
    """
    if checkpoint_path is None:
        return None
    # Warn the user if a checkpoint exists in the train_dir. Then ignore.
    if tf.train.latest_checkpoint(train_dir):
        tf.logging.info(
            'Ignoring --checkpoint_path because a checkpoint already exists in %s'
            % train_dir)
        return None

    exclusions = []
    if checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in checkpoint_exclude_scopes.split(',')]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    # Change model scope if necessary.
    if checkpoint_model_scope is not None:
        variables_to_restore = \
            {var.op.name.replace(model_name,
                                 checkpoint_model_scope): var
             for var in variables_to_restore}


    if tf.gfile.IsDirectory(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    else:
        checkpoint_path = checkpoint_path
    tf.logging.info('Fine-tuning from %s. Ignoring missing vars: %s' % (checkpoint_path, ignore_missing_vars))

    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=ignore_missing_vars)
if __name__ == "__main__":
    main("TF_DATA/train01_valid_skip2.tf",'log',6)