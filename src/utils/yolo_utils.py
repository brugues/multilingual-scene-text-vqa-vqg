import cv2
import numpy as np
<<<<<<< HEAD


def yolo_image_preprocess(image, target_size, gt_boxes=None):
    """
        This function resizes a given image to a target size. The image is padded
        if it is not square. If bounding boxes are provided, those are also adjusted
        to the new image size.

        :param image: image to resize
        :param target_size: new image size
        :param gt_boxes: gt bounding boxes for that image

        :return:
            - either the resized image
                or
            - the resized image and the adjusted bounding boxes

    """

=======
import tensorflow as tf


def yolo_image_preporcess(image, target_size, gt_boxes=None):
>>>>>>> parent of 6fb6c3e (add yolov4 tensorflow code)
    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(float(iw) / w, float(ih) / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

<<<<<<< HEAD
    image_padded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_padded[dh:nh + dh, dw:nw + dw, :] = image_resized
    image_padded = image_padded / 255.

    if gt_boxes is None:
        return image_padded
=======
    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded
>>>>>>> parent of 6fb6c3e (add yolov4 tensorflow code)

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
<<<<<<< HEAD

        return image_padded, gt_boxes
=======
        return image_paded, gt_boxes


def yolo_read_pb_return_tensors(pb_file, return_elements):
    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    return_elements = tf.import_graph_def(frozen_graph_def,
                                          return_elements=return_elements)
    # print [op.name for op in tf.get_default_graph().get_operations()]
    return return_elements
>>>>>>> parent of 6fb6c3e (add yolov4 tensorflow code)
