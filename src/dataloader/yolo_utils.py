import cv2
import numpy as np


def vqa_image_preprocess(image, target_size, gt_boxes=None):
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

    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(float(iw) / w, float(ih) / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_padded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_padded[dh:nh + dh, dw:nw + dw, :] = image_resized
    image_padded = image_padded / 255.

    if gt_boxes is None:
        return image_padded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh

        return image_padded, gt_boxes


def olra_image_preprocess(image, target_size, gt_boxes=None):
    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(float(iw) / w, float(ih) / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_padded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_padded[dh:nh + dh, dw:nw + dw, :] = image_resized

    if gt_boxes is None:
        return image_padded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh

        return image_padded, gt_boxes