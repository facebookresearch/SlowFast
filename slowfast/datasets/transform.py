#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
This script will be responsible for implementing various data transformation
techniques on images.
Assumption: The image format is assumed to be CHW.
Implemented transformations:
    1. Random crop an image
    2. Color normalize an image
    3. Horizontal flip
    4. Pad image
    5. Center crop
    6. Ten crop
    7. Scale
    8. Random sized crop
    9. Lighting
    10. Grayscale
    11. Brightness
    12. Contrast
    13. Saturation
    14. Color-jittering

More fancier pre-processing options at:
https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
"""

import math

import cv2
import numpy as np


def clip_boxes_to_image(boxes, height, width):
    """Clip an array of boxes to an image with the given height and width."""
    boxes[:, [0, 2]] = np.minimum(
        width - 1.0, np.maximum(0.0, boxes[:, [0, 2]])
    )
    boxes[:, [1, 3]] = np.minimum(
        height - 1.0, np.maximum(0.0, boxes[:, [1, 3]])
    )
    return boxes


# Non-local & STRG-style scaling.
# Image should be in format HWC. Scale the smaller edge of image to
# a scale from [1/320, 1/256] for example.
def random_short_side_scale_jitter_list(
    images, min_size, max_size, proposals=None
):

    size = int(round(1.0 / np.random.uniform(1.0 / max_size, 1.0 / min_size)))

    height = images[0].shape[0]
    width = images[0].shape[1]
    if (width <= height and width == size) or (
        height <= width and height == size
    ):
        return images, proposals
    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
        if proposals is not None:
            proposals = [
                proposal * float(new_height) / height for proposal in proposals
            ]
    else:
        new_width = int(math.floor((float(width) / height) * size))
        if proposals is not None:
            proposals = [
                proposal * float(new_width) / width for proposal in proposals
            ]
    return (
        [
            cv2.resize(
                image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
            ).astype(np.float32)
            for image in images
        ],
        proposals,
    )


# Image should be in format HWC. Scale the smaller edge of image to size.
# TODO: considering remove dependency on cv2
def scale(size, image):
    height = image.shape[0]
    width = image.shape[1]
    if (width <= height and width == size) or (
        height <= width and height == size
    ):
        return image
    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
    else:
        new_width = int(math.floor((float(width) / height) * size))
    img = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )
    return img.astype(np.float32)


# Scale the smaller edge of image to size.
def scale_boxes(size, boxes, height, width):
    if (width <= height and width == size) or (
        height <= width and height == size
    ):
        return boxes

    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
        boxes *= float(new_height) / height
    else:
        new_width = int(math.floor((float(width) / height) * size))
        boxes *= float(new_width) / width
    return boxes


def horizontal_flip_list(prob, images, order="CHW", proposals=None):
    _, width, _ = images[0].shape
    if np.random.uniform() < prob:
        if proposals is not None:
            proposals = [flip_boxes(proposal, width) for proposal in proposals]
        if order == "CHW":
            out_images = []
            for image in images:
                image = np.asarray(image).swapaxes(2, 0)
                image = image[::-1]
                out_images.append(image.swapaxes(0, 2))
            return out_images, proposals
        elif order == "HWC":
            # use opencv for flipping image
            return [cv2.flip(image, 1) for image in images], proposals
    return images, proposals


def spatial_shift_crop_list(size, images, spatial_shift_pos, proposals=None):
    assert spatial_shift_pos in [0, 1, 2]

    height = images[0].shape[0]
    width = images[0].shape[1]
    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))

    if height > width:
        if spatial_shift_pos == 0:
            y_offset = 0
        elif spatial_shift_pos == 2:
            y_offset = height - size
    else:
        if spatial_shift_pos == 0:
            x_offset = 0
        elif spatial_shift_pos == 2:
            x_offset = width - size

    cropped = [
        image[y_offset : y_offset + size, x_offset : x_offset + size, :]
        for image in images
    ]
    assert cropped[0].shape[0] == size, "Image height not cropped properly"
    assert cropped[0].shape[1] == size, "Image width not cropped properly"

    if proposals is not None:
        for i in range(len(proposals)):
            proposals[i][:, [0, 2]] -= x_offset
            proposals[i][:, [1, 3]] -= y_offset

    return cropped, proposals


def CHW2HWC(image):
    return image.transpose([1, 2, 0])


def HWC2CHW(image):
    return image.transpose([2, 0, 1])


def color_jitter_list(
    images, img_brightness=0, img_contrast=0, img_saturation=0
):

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
                images = brightness_list(img_brightness, images)
            elif jitter[order[idx]] == "contrast":
                images = contrast_list(img_contrast, images)
            elif jitter[order[idx]] == "saturation":
                images = saturation_list(img_saturation, images)
    return images


# Image should have channel order BGR and CHW format
def lighting_list(imgs, alphastd, eigval, eigvec, alpha=None):
    if alphastd == 0:
        return imgs
    # generate alpha1, alpha2, alpha3
    alpha = np.random.normal(0, alphastd, size=(1, 3))
    eig_vec = np.array(eigvec)
    eig_val = np.reshape(eigval, (1, 3))
    rgb = np.sum(
        eig_vec * np.repeat(alpha, 3, axis=0) * np.repeat(eig_val, 3, axis=0),
        axis=1,
    )
    out_images = []
    for img in imgs:
        for idx in range(img.shape[0]):
            img[idx] = img[idx] + rgb[2 - idx]
        out_images.append(img)
    return out_images


def color_normalization(image, mean, stddev):
    # Input image should in format of CHW
    assert len(mean) == image.shape[0], "channel mean not computed properly"
    assert len(stddev) == image.shape[0], "channel stddev not computed properly"
    for idx in range(image.shape[0]):
        image[idx] = image[idx] - mean[idx]
        image[idx] = image[idx] / stddev[idx]
    return image


def pad_image(image, pad_size, order="CHW"):
    # TODO {haoqi} support non-0 pad
    if order == "CHW":
        img = np.pad(
            image,
            ((0, 0), (pad_size, pad_size), (pad_size, pad_size)),
            mode=str("constant"),
        )
    elif order == "HWC":
        img = np.pad(
            image,
            ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
            mode=str("constant"),
        )
    return img


def horizontal_flip(prob, image, order="CHW"):
    assert order in ["CHW", "HWC"], "order {} is not supported".format(order)
    if np.random.uniform() < prob:
        if order == "CHW":
            image = image[:, :, ::-1]
        elif order == "HWC":
            image = image[:, ::-1, :]
        else:
            raise NotImplementedError("Unknown order {}".format(order))
    return image


def flip_boxes(boxes, im_width):
    boxes_flipped = boxes.copy()
    boxes_flipped[:, 0::4] = im_width - boxes[:, 2::4] - 1
    boxes_flipped[:, 2::4] = im_width - boxes[:, 0::4] - 1
    return boxes_flipped


# random crop from larger image with optional zero padding.
# Image can be in CHW or HWC format. Specify the image order
def random_crop(image, size, pad_size=0, order="CHW"):
    assert order in ["CHW", "HWC"]
    # explicitly dealing processing per image order to avoid flipping images
    if pad_size > 0:
        image = pad_image(image=image, pad_size=pad_size, order=order)
    # image format should be CHW
    if order == "CHW":
        if image.shape[1] == size and image.shape[2] == size:
            return image
        height = image.shape[1]
        width = image.shape[2]
        y_offset = 0
        if height > size:
            y_offset = int(np.random.randint(0, height - size))
        x_offset = 0
        if width > size:
            x_offset = int(np.random.randint(0, width - size))
        cropped = image[
            :, y_offset : y_offset + size, x_offset : x_offset + size
        ]
        assert cropped.shape[1] == size, "Image not cropped properly"
        assert cropped.shape[2] == size, "Image not cropped properly"
    else:
        if image.shape[0] == size and image.shape[1] == size:
            return image
        height = image.shape[0]
        width = image.shape[1]
        y_offset = 0
        if height > size:
            y_offset = int(np.random.randint(0, height - size))
        x_offset = 0
        if width > size:
            x_offset = int(np.random.randint(0, width - size))
        cropped = image[
            y_offset : y_offset + size, x_offset : x_offset + size, :
        ]
        assert cropped.shape[0] == size, "Image not cropped properly"
        assert cropped.shape[1] == size, "Image not cropped properly"
    return cropped


def center_crop(size, image):
    # TODO {haoqi} borc
    height = image.shape[0]
    width = image.shape[1]
    y_offset = int(math.ceil((height - size) / 2))
    x_offset = int(math.ceil((width - size) / 2))
    cropped = image[y_offset : y_offset + size, x_offset : x_offset + size, :]
    assert cropped.shape[0] == size, "Image height not cropped properly"
    assert cropped.shape[1] == size, "Image width not cropped properly"
    return cropped


# ResNet style scale jittering: randomly select the scale from
# [1/max_size, 1/min_size]
def random_scale_jitter(image, min_size, max_size):
    # randomly select a scale from [1/480, 1/256] for example
    img_scale = int(
        round(1.0 / np.random.uniform(1.0 / max_size, 1.0 / min_size))
    )
    image = scale(img_scale, image)
    return image


def random_scale_jitter_list(images, min_size, max_size):
    # randomly select a scale from [1/480, 1/256] for example
    img_scale = int(
        round(1.0 / np.random.uniform(1.0 / max_size, 1.0 / min_size))
    )
    return [scale(img_scale, image) for image in images]


def random_sized_crop(image, size, area_frac=0.08):
    # TODO {haoqi} borc
    for _ in range(0, 10):
        height = image.shape[0]
        width = image.shape[1]
        area = height * width
        target_area = np.random.uniform(area_frac, 1.0) * area
        aspect_ratio = np.random.uniform(3.0 / 4.0, 4.0 / 3.0)
        w = int(round(math.sqrt(float(target_area) * aspect_ratio)))
        h = int(round(math.sqrt(float(target_area) / aspect_ratio)))
        if np.random.uniform() < 0.5:
            w, h = h, w
        if h <= height and w <= width:
            if height == h:
                y_offset = 0
            else:
                y_offset = np.random.randint(0, height - h)
            if width == w:
                x_offset = 0
            else:
                x_offset = np.random.randint(0, width - w)
            y_offset = int(y_offset)
            x_offset = int(x_offset)
            cropped = image[y_offset : y_offset + h, x_offset : x_offset + w, :]
            assert (
                cropped.shape[0] == h and cropped.shape[1] == w
            ), "Wrong crop size"
            cropped = cv2.resize(
                cropped, (size, size), interpolation=cv2.INTER_LINEAR
            )
            return cropped.astype(np.float32)
    return center_crop(size, scale(size, image))


# image should be in format HWC
def five_crop(size, image, crop_images):
    height = image.shape[0]
    width = image.shape[1]
    # given an image, crop the 4 corners and center of given size
    center_cropped = center_crop(size, image)
    crop_images.extend([center_cropped])
    # crop the top left corner:
    crop_images.extend([image[0:size, 0:size, :]])
    # crop the top right corner
    crop_images.extend([image[0:size, width - size : width, :]])
    # crop bottom left corner
    crop_images.extend([image[height - size : height, 0:size, :]])
    # crop bottom right corner
    crop_images.extend([image[height - size : height, width - size : width, :]])
    return crop_images


# image should be in format HWC
def ten_crop(size, image):
    ten_crop_images = []

    # For the original image, crop the center and 4 corners
    ten_crop_images = five_crop(size, image, ten_crop_images)

    # Flip the image horizontally
    flipped = horizontal_flip(1.0, image, order="HWC")
    ten_crop_images = five_crop(size, flipped, ten_crop_images)

    # convert this into 10 x H x W x C
    return np.concatenate([arr[np.newaxis] for arr in ten_crop_images]).astype(
        np.float32
    )


def lighting(img, alphastd, eigval, eigvec):
    # TODO(haoqifan): Refactor
    if alphastd == 0:
        return img
    # generate alpha1, alpha2, alpha3
    alpha = np.random.normal(0, alphastd, size=(1, 3))
    eig_vec = np.array(eigvec)
    eig_val = np.reshape(eigval, (1, 3))
    rgb = np.sum(
        eig_vec * np.repeat(alpha, 3, axis=0) * np.repeat(eig_val, 3, axis=0),
        axis=1,
    )
    for idx in range(img.shape[0]):
        img[idx] = img[idx] + rgb[2 - idx]
    return img


# Random crop with size 8% - 100% image area and aspect ratio in [3/4, 4/3]
# Reference: GoogleNet paper
# Image should be in format HWC
def random_sized_crop_list(images, size, crop_area_fraction=0.08):
    for _ in range(0, 10):
        height = images[0].shape[0]
        width = images[0].shape[1]
        area = height * width
        target_area = np.random.uniform(crop_area_fraction, 1.0) * area
        aspect_ratio = np.random.uniform(3.0 / 4.0, 4.0 / 3.0)
        w = int(round(math.sqrt(float(target_area) * aspect_ratio)))
        h = int(round(math.sqrt(float(target_area) / aspect_ratio)))
        if np.random.uniform() < 0.5:
            w, h = h, w
        if h <= height and w <= width:
            if height == h:
                y_offset = 0
            else:
                y_offset = np.random.randint(0, height - h)
            if width == w:
                x_offset = 0
            else:
                x_offset = np.random.randint(0, width - w)
            y_offset = int(y_offset)
            x_offset = int(x_offset)

            croppsed_images = []
            for image in images:
                cropped = image[
                    y_offset : y_offset + h, x_offset : x_offset + w, :
                ]
                assert (
                    cropped.shape[0] == h and cropped.shape[1] == w
                ), "Wrong crop size"
                cropped = cv2.resize(
                    cropped, (size, size), interpolation=cv2.INTER_LINEAR
                )
                croppsed_images.append(cropped.astype(np.float32))
            return croppsed_images

    return [center_crop(size, scale(size, image)) for image in images]


def blend(image1, image2, alpha):
    return image1 * alpha + image2 * (1 - alpha)


# image should be in format CHW and the channels in order BGR
def grayscale(image):
    # R -> 0.299, G -> 0.587, B -> 0.114
    img_gray = np.copy(image)
    gray_channel = 0.299 * image[2] + 0.587 * image[1] + 0.114 * image[0]
    img_gray[0] = gray_channel
    img_gray[1] = gray_channel
    img_gray[2] = gray_channel
    return img_gray


def saturation(var, image):
    img_gray = grayscale(image)
    alpha = 1.0 + np.random.uniform(-var, var)
    return blend(image, img_gray, alpha)


def brightness(var, image):
    img_bright = np.zeros(image.shape)
    alpha = 1.0 + np.random.uniform(-var, var)
    return blend(image, img_bright, alpha)


def contrast(var, image):
    img_gray = grayscale(image)
    img_gray.fill(np.mean(img_gray[0]))
    alpha = 1.0 + np.random.uniform(-var, var)
    return blend(image, img_gray, alpha)


def saturation_list(var, images):
    alpha = 1.0 + np.random.uniform(-var, var)

    out_images = []
    for image in images:
        img_gray = grayscale(image)
        out_images.append(blend(image, img_gray, alpha))
    return out_images


def brightness_list(var, images):
    alpha = 1.0 + np.random.uniform(-var, var)

    out_images = []
    for image in images:
        img_bright = np.zeros(image.shape)
        out_images.append(blend(image, img_bright, alpha))
    return out_images


def contrast_list(var, images):
    alpha = 1.0 + np.random.uniform(-var, var)

    out_images = []
    for image in images:
        img_gray = grayscale(image)
        img_gray.fill(np.mean(img_gray[0]))
        out_images.append(blend(image, img_gray, alpha))
    return out_images


def color_jitter(image, img_brightness=0, img_contrast=0, img_saturation=0):
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
                image = brightness(img_brightness, image)
            elif jitter[order[idx]] == "contrast":
                image = contrast(img_contrast, image)
            elif jitter[order[idx]] == "saturation":
                image = saturation(img_saturation, image)
    return image
