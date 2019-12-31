#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
import numpy as np
import torch


def random_short_side_scale_jitter(images, min_size, max_size, boxes=None):
    """
    Perform a spatial short scale jittering on the given images and
    corresponding boxes.
    Args:
        images (tensor): images to perform scale jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
        min_size (int): the minimal size to scale the frames.
        max_size (int): the maximal size to scale the frames.
        boxes (ndarray): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
    Returns:
        (tensor): the scaled images with dimension of
            `num frames` x `channel` x `new height` x `new width`.
        (ndarray or None): the scaled boxes with dimension of
            `num boxes` x 4.
    """
    size = int(round(np.random.uniform(min_size, max_size)))

    height = images.shape[2]
    width = images.shape[3]
    if (width <= height and width == size) or (
        height <= width and height == size
    ):
        return images, boxes
    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
        if boxes is not None:
            boxes = boxes * float(new_height) / height
    else:
        new_width = int(math.floor((float(width) / height) * size))
        if boxes is not None:
            boxes = boxes * float(new_width) / width

    return (
        torch.nn.functional.interpolate(
            images,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        ),
        boxes,
    )


def crop_boxes(boxes, x_offset, y_offset):
    """
    Peform crop on the bounding boxes given the offsets.
    Args:
        boxes (ndarray or None): bounding boxes to peform crop. The dimension
            is `num boxes` x 4.
        x_offset (int): cropping offset in the x axis.
        y_offset (int): cropping offset in the y axis.
    Returns:
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    cropped_boxes = boxes.copy()
    cropped_boxes[:, [0, 2]] = boxes[:, [0, 2]] - x_offset
    cropped_boxes[:, [1, 3]] = boxes[:, [1, 3]] - y_offset

    return cropped_boxes


def random_crop(images, size, boxes=None):
    """
    Perform random spatial crop on the given images and corresponding boxes.
    Args:
        images (tensor): images to perform random crop. The dimension is
            `num frames` x `channel` x `height` x `width`.
        size (int): the size of height and width to crop on the image.
        boxes (ndarray or None): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
    Returns:
        cropped (tensor): cropped images with dimension of
            `num frames` x `channel` x `size` x `size`.
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    if images.shape[2] == size and images.shape[3] == size:
        return images
    height = images.shape[2]
    width = images.shape[3]
    y_offset = 0
    if height > size:
        y_offset = int(np.random.randint(0, height - size))
    x_offset = 0
    if width > size:
        x_offset = int(np.random.randint(0, width - size))
    cropped = images[
        :, :, y_offset : y_offset + size, x_offset : x_offset + size
    ]

    cropped_boxes = (
        crop_boxes(boxes, x_offset, y_offset) if boxes is not None else None
    )

    return cropped, cropped_boxes


def horizontal_flip(prob, images, boxes=None):
    """
    Perform horizontal flip on the given images and corresponding boxes.
    Args:
        prob (float): probility to flip the images.
        images (tensor): images to perform horizontal flip, the dimension is
            `num frames` x `channel` x `height` x `width`.
        boxes (ndarray or None): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
    Returns:
        images (tensor): images with dimension of
            `num frames` x `channel` x `height` x `width`.
        flipped_boxes (ndarray or None): the flipped boxes with dimension of
            `num boxes` x 4.
    """
    if boxes is None:
        flipped_boxes = None
    else:
        flipped_boxes = boxes.copy()

    if np.random.uniform() < prob:
        images = images.flip((-1))

        width = images.shape[3]
        if boxes is not None:
            flipped_boxes[:, [0, 2]] = width - boxes[:, [2, 0]] - 1

    return images, flipped_boxes


def uniform_crop(images, size, spatial_idx, boxes=None):
    """
    Perform uniform spatial sampling on the images and corresponding boxes.
    Args:
        images (tensor): images to perform uniform crop. The dimension is
            `num frames` x `channel` x `height` x `width`.
        size (int): size of height and weight to crop the images.
        spatial_idx (int): 0, 1, or 2 for left, center, and right crop if width
            is larger than height. Or 0, 1, or 2 for top, center, and bottom
            crop if height is larger than width.
        boxes (ndarray or None): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
    Returns:
        cropped (tensor): images with dimension of
            `num frames` x `channel` x `size` x `size`.
        cropped_boxes (ndarray or None): the cropped boxes with dimension of
            `num boxes` x 4.
    """
    assert spatial_idx in [0, 1, 2]
    height = images.shape[2]
    width = images.shape[3]

    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_idx == 0:
            y_offset = 0
        elif spatial_idx == 2:
            y_offset = height - size
    else:
        if spatial_idx == 0:
            x_offset = 0
        elif spatial_idx == 2:
            x_offset = width - size
    cropped = images[
        :, :, y_offset : y_offset + size, x_offset : x_offset + size
    ]

    cropped_boxes = (
        crop_boxes(boxes, x_offset, y_offset) if boxes is not None else None
    )

    return cropped, cropped_boxes


def clip_boxes_to_image(boxes, height, width):
    """
    Clip an array of boxes to an image with the given height and width.
    Args:
        boxes (ndarray): bounding boxes to perform clipping.
            Dimension is `num boxes` x 4.
        height (int): given image height.
        width (int): given image width.
    Returns:
        clipped_boxes (ndarray): the clipped boxes with dimension of
            `num boxes` x 4.
    """
    clipped_boxes = boxes.copy()
    clipped_boxes[:, [0, 2]] = np.minimum(
        width - 1.0, np.maximum(0.0, boxes[:, [0, 2]])
    )
    clipped_boxes[:, [1, 3]] = np.minimum(
        height - 1.0, np.maximum(0.0, boxes[:, [1, 3]])
    )
    return clipped_boxes


def blend(images1, images2, alpha):
    """
    Blend two images with a given weight alpha.
    Args:
        images1 (tensor): the first images to be blended, the dimension is
            `num frames` x `channel` x `height` x `width`.
        images2 (tensor): the second images to be blended, the dimension is
            `num frames` x `channel` x `height` x `width`.
        alpha (float): the blending weight.
    Returns:
        (tensor): blended images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    return images1 * alpha + images2 * (1 - alpha)


def grayscale(images):
    """
    Get the grayscale for the input images. The channels of images should be
    in order BGR.
    Args:
        images (tensor): the input images for getting grayscale. Dimension is
            `num frames` x `channel` x `height` x `width`.
    Returns:
        img_gray (tensor): blended images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    # R -> 0.299, G -> 0.587, B -> 0.114.
    img_gray = torch.tensor(images)
    gray_channel = (
        0.299 * images[:, 2] + 0.587 * images[:, 1] + 0.114 * images[:, 0]
    )
    img_gray[:, 0] = gray_channel
    img_gray[:, 1] = gray_channel
    img_gray[:, 2] = gray_channel
    return img_gray


def color_jitter(images, img_brightness=0, img_contrast=0, img_saturation=0):
    """
    Perfrom a color jittering on the input images. The channels of images
    should be in order BGR.
    Args:
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
        img_brightness (float): jitter ratio for brightness.
        img_contrast (float): jitter ratio for contrast.
        img_saturation (float): jitter ratio for saturation.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """

    jitter = []
    if img_brightness != 0:
        jitter.append("brightness")
    if img_contrast != 0:
        jitter.append("contrast")
    if img_saturation != 0:
        jitter.append("saturation")

    if len(jitter) > 0:
        order = np.random.permutation(np.arange(len(jitter)))
        for idx in range(0, len(jitter)):
            if jitter[order[idx]] == "brightness":
                images = brightness_jitter(img_brightness, images)
            elif jitter[order[idx]] == "contrast":
                images = contrast_jitter(img_contrast, images)
            elif jitter[order[idx]] == "saturation":
                images = saturation_jitter(img_saturation, images)
    return images


def brightness_jitter(var, images):
    """
    Perfrom brightness jittering on the input images. The channels of images
    should be in order BGR.
    Args:
        var (float): jitter ratio for brightness.
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    alpha = 1.0 + np.random.uniform(-var, var)

    img_bright = torch.zeros(images.shape)
    images = blend(images, img_bright, alpha)
    return images


def contrast_jitter(var, images):
    """
    Perfrom contrast jittering on the input images. The channels of images
    should be in order BGR.
    Args:
        var (float): jitter ratio for contrast.
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    alpha = 1.0 + np.random.uniform(-var, var)

    img_gray = grayscale(images)
    img_gray[:] = torch.mean(img_gray, dim=(1, 2, 3), keepdim=True)
    images = blend(images, img_gray, alpha)
    return images


def saturation_jitter(var, images):
    """
    Perfrom saturation jittering on the input images. The channels of images
    should be in order BGR.
    Args:
        var (float): jitter ratio for saturation.
        images (tensor): images to perform color jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
    Returns:
        images (tensor): the jittered images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    alpha = 1.0 + np.random.uniform(-var, var)
    img_gray = grayscale(images)
    images = blend(images, img_gray, alpha)

    return images


def lighting_jitter(images, alphastd, eigval, eigvec):
    """
    Perform AlexNet-style PCA jitter on the given images.
    Args:
        images (tensor): images to perform lighting jitter. Dimension is
            `num frames` x `channel` x `height` x `width`.
        alphastd (float): jitter ratio for PCA jitter.
        eigval (list): eigenvalues for PCA jitter.
        eigvec (list[list]): eigenvectors for PCA jitter.
    Returns:
        out_images (tensor): the jittered images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    if alphastd == 0:
        return images
    # generate alpha1, alpha2, alpha3.
    alpha = np.random.normal(0, alphastd, size=(1, 3))
    eig_vec = np.array(eigvec)
    eig_val = np.reshape(eigval, (1, 3))
    rgb = np.sum(
        eig_vec * np.repeat(alpha, 3, axis=0) * np.repeat(eig_val, 3, axis=0),
        axis=1,
    )
    out_images = torch.zeros_like(images)
    for idx in range(images.shape[1]):
        out_images[:, idx] = images[:, idx] + rgb[2 - idx]

    return out_images


def color_normalization(images, mean, stddev):
    """
    Perform color nomration on the given images.
    Args:
        images (tensor): images to perform color normalization. Dimension is
            `num frames` x `channel` x `height` x `width`.
        mean (list): mean values for normalization.
        stddev (list): standard deviations for normalization.

    Returns:
        out_images (tensor): the noramlized images, the dimension is
            `num frames` x `channel` x `height` x `width`.
    """
    assert len(mean) == images.shape[1], "channel mean not computed properly"
    assert (
        len(stddev) == images.shape[1]
    ), "channel stddev not computed properly"

    out_images = torch.zeros_like(images)
    for idx in range(len(mean)):
        out_images[:, idx] = (images[:, idx] - mean[idx]) / stddev[idx]

    return out_images
