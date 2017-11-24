# import tensorflow as tf
# from util import tf_image_processing
#
#
#
# def tf_summary_image(image, bboxes, name='image', unwhitened=False):
#     """Add image with bounding boxes to summary.
#     """
#     # if unwhitened:
#     #     image = tf_image_unwhitened(image)
#     image = tf.expand_dims(image, 0)
#     bboxes = tf.expand_dims(bboxes, 0)
#     image_with_box = tf.image.draw_bounding_boxes(image, bboxes)
#     tf.summary.image(name, image_with_box)
#
#
# def preprocess_for_train(image, labels, bboxes,
#                          out_shape, data_format='NHWC',
#                          scope='ssd_preprocessing_train'):
#     """Preprocesses the given image for training.
#
#     Note that the actual resizing scale is sampled from
#         [`resize_size_min`, `resize_size_max`].
#
#     Args:
#         image: A `Tensor` representing an image of arbitrary size.
#         output_height: The height of the image after preprocessing.
#         output_width: The width of the image after preprocessing.
#         resize_side_min: The lower bound for the smallest side of the image for
#             aspect-preserving resizing.
#         resize_side_max: The upper bound for the smallest side of the image for
#             aspect-preserving resizing.
#
#     Returns:
#         A preprocessed image.
#     """
#     fast_mode = False
#     with tf.name_scope(scope, 'ssd_preprocessing_train', [image, labels, bboxes]):
#         if image.get_shape().ndims != 3:
#             raise ValueError('Input must be of size [height, width, C>0]')
#         # Convert to float scaled [0, 1].
#         if image.dtype != tf.float32:
#             image = tf.image.convert_image_dtype(image, dtype=tf.float32)
#         tf_summary_image(image, bboxes, 'image_with_bboxes')
#
#         # # Remove DontCare labels.
#         # labels, bboxes = ssd_common.tf_bboxes_filter_labels(out_label,
#         #                                                     labels,
#         #                                                     bboxes)
#
#         # Distort image and bounding boxes.
#         dst_image = image
#         dst_image, labels, bboxes, distort_bbox = \
#             distorted_bounding_box_crop(image, labels, bboxes,
#                                         min_object_covered=MIN_OBJECT_COVERED,
#                                         aspect_ratio_range=CROP_RATIO_RANGE)
#         # Resize image to output size.
#         dst_image = tf_image.resize_image(dst_image, out_shape,
#                                           method=tf.image.ResizeMethod.BILINEAR,
#                                           align_corners=False)
#         tf_summary_image(dst_image, bboxes, 'image_shape_distorted')
#
#         # Randomly flip the image horizontally.
#         dst_image, bboxes = tf_image.random_flip_left_right(dst_image, bboxes)
#
#         # Randomly distort the colors. There are 4 ways to do it.
#         dst_image = apply_with_random_selector(
#                 dst_image,
#                 lambda x, ordering: distort_color(x, ordering, fast_mode),
#                 num_cases=4)
#         tf_summary_image(dst_image, bboxes, 'image_color_distorted')
#
#         # Rescale to VGG input scale.
#         image = dst_image * 255.
#         image = tf_image_whitened(image, [_R_MEAN, _G_MEAN, _B_MEAN])
#         # Image data format.
#         if data_format == 'NCHW':
#             image = tf.transpose(image, perm=(2, 0, 1))
#         return image, labels, bboxes
