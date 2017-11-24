import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
slim = tf.contrib.slim


## Real : NUM_SAMPLES = 11079
#NUM_SAMPLES = 11079 ##for training
NUM_SAMPLES = 1151
NUM_CLASSES = 2
LABEL_TO_NAME = {0:"none",1:"person"}

ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'shape': 'Shape of the image',
    'object/bbox': 'A list of bounding boxes, one per each object.',
    'object/label': 'A list of labels, one per each object.',
    'object/difficult': 'A list of difficult, one per each object.',
}

########for slim #############
def get_dataset(tf_record_dir):

    reader = tf.TFRecordReader
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
        'object/difficult': slim.tfexample_decoder.Tensor('image/object/bbox/difficult'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features,items_to_handlers)
    return slim.dataset.Dataset(
            data_sources=tf_record_dir,
            reader=reader,
            decoder=decoder,
            num_samples=NUM_SAMPLES,
            items_to_descriptions=ITEMS_TO_DESCRIPTIONS,
            num_classes=NUM_CLASSES,
            labels_to_names=LABEL_TO_NAME)

def get_data_without_sess(tf_record_dir):
    record_iterator = tf.python_io.tf_record_iterator(path=tf_record_dir)
    for i, string_record in enumerate(record_iterator):
        example = tf.train.Example()
        example.ParseFromString(string_record)
        height = int(example.features.feature['image/height'].int64_list.value[0])
        width = int(example.features.feature['image/width'].int64_list.value[0])
        channels = int(example.features.feature['image/channels'].int64_list.value[0])
        img_string = example.features.feature['image/encoded'].bytes_list.value
        label_text = example.features.feature['image/object/bbox/label_text'].bytes_list.value
        label =  example.features.feature['image/channels'].int64_list.value[0]


def read_and_decode(tf_record_dir):

    filename_queue = tf.train.string_input_producer([tf_record_dir])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,features={
        'image/height': tf.FixedLenFeature([1],tf.int64),
        'image/width': tf.FixedLenFeature([1],tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3],tf.int64),
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64)
    })
    img = tf.image.decode_jpeg(features['image/encoded'],3)
    img = tf.reshape(img,[512,640,3])
    label = tf.cast(features['image/object/bbox/label'],tf.int32)
    return img,label

def test_batch_decoder(tf_record_dir):

    img,label = read_and_decode(tf_record_dir)

    img_batch, label_batch = tf.train.shuffle_batch([img,label],batch_size=4,capacity=2000,min_after_dequeue=1000,num_threads=8)
    print "image batch", img_batch._shape

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        img_res, label_res = sess.run([img_batch,label_batch])
        coord.request_stop()
        coord.join(threads)
    imgs = np.asarray(img_res)
    labels = np.asarray(label_res)
    cv2.imshow("s",imgs[0,...])
    cv2.waitKey(0)
    print imgs.shape
    print labels.shape


if __name__ == "__main__":
    a = get_dataset("../TF_DATA/train20_skip1.tf")
    print "dd"