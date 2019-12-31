#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
import numpy as np

import cv2


def clip_boxes_to_image(boxes, height, width):
    """
    Clip the boxes with the height and width of the image size.
    Args:
        boxes (ndarray): bounding boxes to peform crop. The dimension is
        `num boxes` x 4.
        height (int): the height of the image.
        width (int): the width of the image.
    Returns:
        boxes (ndarray): cropped bounding boxes.
    """
    boxes[:, [0, 2]] = np.minimum(
        width - 1.0, np.maximum(0.0, boxes[:, [0, 2]])
    )
    boxes[:, [1, 3]] = np.minimum(
        height - 1.0, np.maximum(0.0, boxes[:, [1, 3]])
    )
    return boxes


def random_short_side_scale_jitter_list(images, min_size, max_size, boxes=None):
    """
    Perform a spatial short scale jittering on the given images and
    corresponding boxes.
    Args:
        images (list): list of images to perform scale jitter. Dimension is
            `height` x `width` x `channel`.
        min_size (int): the minimal size to scale the frames.
        max_size (int): the maximal size to scale the frames.
        boxes (list): optional. Corresponding boxes to images. Dimension is
            `num boxes` x 4.
    Returns:
        (list): the list of scaled images with dimension of
            `new height` x `new width` x `channel`.
        (ndarray or None): the scaled boxes with dimension of
            `num boxes` x 4.
    """
    size = int(round(1.0 / np.random.uniform(1.0 / max_size, 1.0 / min_size)))

    height = images[0].shape[0]
    width = images[0].shape[1]
    if (width <= height and width == size) or (
        height <= width and height == size
    ):
        return images, boxes
    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
        if boxes is not None:
            boxes = [
                proposal * float(new_height) / height for proposal in boxes
            ]
    else:
        new_width = int(math.floor((float(width) / height) * size))
        if boxes is not None:
            boxes = [proposal * float(new_width) / width for proposal in boxes]
    return (
        [
            cv2.resize(
                image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
            ).astype(np.float32)
            for image in images
        ],
        boxes,
    )


def scale(size, image):
    """
    Scale the short side of the image to size.
    Args:
        size (int): size to scale the image.
        image (array): image to perform short side scale. Dimension is
            `height` x `width` x `channel`.
    Returns:
        (ndarray): the scaled image with dimension of
            `height` x `width` x `channel`.
    """
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


def scale_boxes(size, boxes, height, width):
    """
    Scale the short side of the box to size.
    Args:
        size (int): size to scale the image.
        boxes (ndarray): bounding boxes to peform scale. The dimension is
        `num boxes` x 4.
        height (int): the height of the image.
        width (int): the width of the image.
    Returns:
        boxes (ndarray): scaled bounding boxes.
    """
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


def horizontal_flip_list(prob, images, order="CHW", boxes=None):
    """
    Horizontally flip the list of image and optional boxes.
    Args:
        prob (float): probability to flip.
        image (list): ilist of images to perform short side scale. Dimension is
            `height` x `width` x `channel` or `channel` x `height` x `width`.
        order (str): order of the `height`, `channel` and `width`.
        boxes (list): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
    Returns:
        (ndarray): the scaled image with dimension of
            `height` x `width` x `channel`.
        (list): optional. Corresponding boxes to images. Dimension is
            `num boxes` x 4.
    """
    _, width, _ = images[0].shape
    if np.random.uniform() < prob:
        if boxes is not None:
            boxes = [flip_boxes(proposal, width) for proposal in boxes]
        if order == "CHW":
            out_images = []
            for image in images:
                image = np.asarray(image).swapaxes(2, 0)
                image = image[::-1]
                out_images.append(image.swapaxes(0, 2))
            return out_images, boxes
        elif order == "HWC":
            return [cv2.flip(image, 1) for image in images], boxes
    return images, boxes


def spatial_shift_crop_list(size, images, spatial_shift_pos, boxes=None):
    """
    Perform left, center, or right crop of the given list of images.
    Args:
        size (int): size to crop.
        image (list): ilist of images to perform short side scale. Dimension is
            `height` x `width` x `channel` or `channel` x `height` x `width`.
        spatial_shift_pos (int): option includes 0 (left), 1 (middle), and
            2 (right) crop.
        boxes (list): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
    Returns:
        cropped (ndarray): the cropped list of images with dimension of
            `height` x `width` x `channel`.
        boxes (list): optional. Corresponding boxes to images. Dimension is
            `num boxes` x 4.
    """

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

    if boxes is not None:
        for i in range(len(boxes)):
            boxes[i][:, [0, 2]] -= x_offset
            boxes[i][:, [1, 3]] -= y_offset
    return cropped, boxes


def CHW2HWC(image):
    """
    Transpose the dimension from `channel` x `height` x `width` to
        `height` x `width` x `channel`.
    Args:
        image (array): image to transpose.
    Returns
        (array): transposed image.
    """
    return image.transpose([1, 2, 0])


def HWC2CHW(image):
    """
    Transpose the dimension from `height` x `width` x `channel` to
        `channel` x `height` x `width`.
    Args:
        image (array): image to transpose.
    Returns
        (array): transposed image.
    """
    return image.transpose([2, 0, 1])


def color_jitter_list(
    images, img_brightness=0, img_contrast=0, img_saturation=0
):
    """
    Perform color jitter on the list of images.
    Args:
        images (list): list of images to perform color jitter.
        img_brightness (float): jitter ratio for brightness.
        img_contrast (float): jitter ratio for contrast.
        img_saturation (float): jitter ratio for saturation.
    Returns:
        images (list): the jittered list of images.
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
                images = brightness_list(img_brightness, images)
            elif jitter[order[idx]] == "contrast":
                images = contrast_list(img_contrast, images)
            elif jitter[order[idx]] == "saturation":
                images = saturation_list(img_saturation, images)
    return images


def lighting_list(imgs, alphastd, eigval, eigvec, alpha=None):
    """
    Perform AlexNet-style PCA jitter on the given list of images.
    Args:
        images (list): list of images to perform lighting jitter.
        alphastd (float): jitter ratio for PCA jitter.
        eigval (list): eigenvalues for PCA jitter.
        eigvec (list[list]): eigenvectors for PCA jitter.
    Returns:
        out_images (list): the list of jittered images.
    """
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
    """
    Perform color normalization on the image with the given mean and stddev.
    Args:
        image (array): image to perform color normalization.
        mean (float): mean value to subtract.
        stddev (float): stddev to devide.
    """
    # Input image should in format of CHW
    assert len(mean) == image.shape[0], "channel mean not computed properly"
    assert len(stddev) == image.shape[0], "channel stddev not computed properly"
    for idx in range(image.shape[0]):
        image[idx] = image[idx] - mean[idx]
        image[idx] = image[idx] / stddev[idx]
    return image


def pad_image(image, pad_size, order="CHW"):
    """
    Pad the given image with the size of pad_size.
    Args:
        image (array): image to pad.
        pad_size (int): size to pad.
        order (str): order of the `height`, `channel` and `width`.
    Returns:
        img (array): padded image.
    """
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
    """
    Horizontally flip the image.
    Args:
        prob (float): probability to flip.
        image (array): image to pad.
        order (str): order of the `height`, `channel` and `width`.
    Returns:
        img (array): flipped image.
    """
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
    """
    Horizontally flip the boxes.
    Args:
        boxes (array): box to flip.
        im_width (int): width of the image.
    Returns:
        boxes_flipped (array): flipped box.
    """

    boxes_flipped = boxes.copy()
    boxes_flipped[:, 0::4] = im_width - boxes[:, 2::4] - 1
    boxes_flipped[:, 2::4] = im_width - boxes[:, 0::4] - 1
    return boxes_flipped


def crop_boxes(boxes, x_offset, y_offset):
    """
    Crop the boxes given the offsets.
    Args:
        boxes (array): boxes to crop.
        x_offset (int): offset on x.
        y_offset (int): offset on y.
    """
    boxes[:, [0, 2]] = boxes[:, [0, 2]] - x_offset
    boxes[:, [1, 3]] = boxes[:, [1, 3]] - y_offset
    return boxes


def random_crop_list(images, size, pad_size=0, order="CHW", boxes=None):
    """
    Perform random crop on a list of images.
    Args:
        images (list): list of images to perform random crop.
        size (int): size to crop.
        pad_size (int): padding size.
        order (str): order of the `height`, `channel` and `width`.
        boxes (list): optional. Corresponding boxes to images.
            Dimension is `num boxes` x 4.
    Returns:
        cropped (ndarray): the cropped list of images with dimension of
            `height` x `width` x `channel`.
        boxes (list): optional. Corresponding boxes to images. Dimension is
            `num boxes` x 4.
    """
    # explicitly dealing processing per image order to avoid flipping images.
    if pad_size > 0:
        images = [
            pad_image(pad_size=pad_size, image=image, order=order)
            for image in images
        ]

    # image format should be CHW.
    if order == "CHW":
        if images[0].shape[1] == size and images[0].shape[2] == size:
            return images, boxes
        height = images[0].shape[1]
        width = images[0].shape[2]
        y_offset = 0
        if height > size:
            y_offset = int(np.random.randint(0, height - size))
        x_offset = 0
        if width > size:
            x_offset = int(np.random.randint(0, width - size))
        cropped = [
            image[:, y_offset : y_offset + size, x_offset : x_offset + size]
            for image in images
        ]
        assert cropped[0].shape[1] == size, "Image not cropped properly"
        assert cropped[0].shape[2] == size, "Image not cropped properly"
    elif order == "HWC":
        if images[0].shape[0] == size and images[0].shape[1] == size:
            return images, boxes
        height = images[0].shape[0]
        width = images[0].shape[1]
        y_offset = 0
        if height > size:
            y_offset = int(np.random.randint(0, height - size))
        x_offset = 0
        if width > size:
            x_offset = int(np.random.randint(0, width - size))
        cropped = [
            image[y_offset : y_offset + size, x_offset : x_offset + size, :]
            for image in images
        ]
        assert cropped[0].shape[0] == size, "Image not cropped properly"
        assert cropped[0].shape[1] == size, "Image not cropped properly"

    if boxes is not None:
        boxes = [crop_boxes(proposal, x_offset, y_offset) for proposal in boxes]
    return cropped, boxes


def center_crop(size, image):
    """
    Perform center crop on input images.
    Args:
        size (int): size of the cropped height and width.
        image (array): the image to perform center crop.
    """
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
    """
    Perform ResNet style random scale jittering: randomly select the scale from
        [1/max_size, 1/min_size].
    Args:
        image (array): image to perform random scale.
        min_size (int): min size to scale.
        max_size (int) max size to scale.
    Returns:
        image (array): scaled image.
    """
    img_scale = int(
        round(1.0 / np.random.uniform(1.0 / max_size, 1.0 / min_size))
    )
    image = scale(img_scale, image)
    return image


def random_scale_jitter_list(images, min_size, max_size):
    """
    Perform ResNet style random scale jittering on a list of image: randomly
        select the scale from [1/max_size, 1/min_size]. Note that all the image
        will share the same scale.
    Args:
        images (list): list of images to perform random scale.
        min_size (int): min size to scale.
        max_size (int) max size to scale.
    Returns:
        images (list): list of scaled image.
    """
    img_scale = int(
        round(1.0 / np.random.uniform(1.0 / max_size, 1.0 / min_size))
    )
    return [scale(img_scale, image) for image in images]


def random_sized_crop(image, size, area_frac=0.08):
    """
    Perform random sized cropping on the given image. Random crop with size
        8% - 100% image area and aspect ratio in [3/4, 4/3].
    Args:
        image (array): image to crop.
        size (int): size to crop.
        area_frac (float): area of fraction.
    Returns:
        (array): cropped image.
    """
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


def lighting(img, alphastd, eigval, eigvec):
    """
    Perform AlexNet-style PCA jitter on the given image.
    Args:
        image (array): list of images to perform lighting jitter.
        alphastd (float): jitter ratio for PCA jitter.
        eigval (array): eigenvalues for PCA jitter.
        eigvec (list): eigenvectors for PCA jitter.
    Returns:
        img (tensor): the jittered image.
    """
    if alphastd == 0:
        return img
    # generate alpha1, alpha2, alpha3.
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


def random_sized_crop_list(images, size, crop_area_fraction=0.08):
    """
    Perform random sized cropping on the given list of images. Random crop with
        size 8% - 100% image area and aspect ratio in [3/4, 4/3].
    Args:
        images (list): image to crop.
        size (int): size to crop.
        area_frac (float): area of fraction.
    Returns:
        (list): list of cropped image.
    """
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


def grayscale(image):
    """
    Convert the image to gray scale.
    Args:
        image (tensor): image to convert to gray scale. Dimension is
            `channel` x `height` x `width`.
    Returns:
        img_gray (tensor): image in gray scale.
    """
    # R -> 0.299, G -> 0.587, B -> 0.114.
    img_gray = np.copy(image)
    gray_channel = 0.299 * image[2] + 0.587 * image[1] + 0.114 * image[0]
    img_gray[0] = gray_channel
    img_gray[1] = gray_channel
    img_gray[2] = gray_channel
    return img_gray


def saturation(var, image):
    """
    Perform color saturation on the given image.
    Args:
        var (float): variance.
        image (array): image to perform color saturation.
    Returns:
        (array): image that performed color saturation.
    """
    img_gray = grayscale(image)
    alpha = 1.0 + np.random.uniform(-var, var)
    return blend(image, img_gray, alpha)


def brightness(var, image):
    """
    Perform color brightness on the given image.
    Args:
        var (float): variance.
        image (array): image to perform color brightness.
    Returns:
        (array): image that performed color brightness.
    """
    img_bright = np.zeros(image.shape).astype(image.dtype)
    alpha = 1.0 + np.random.uniform(-var, var)
    return blend(image, img_bright, alpha)


def contrast(var, image):
    """
    Perform color contrast on the given image.
    Args:
        var (float): variance.
        image (array): image to perform color contrast.
    Returns:
        (array): image that performed color contrast.
    """
    img_gray = grayscale(image)
    img_gray.fill(np.mean(img_gray[0]))
    alpha = 1.0 + np.random.uniform(-var, var)
    return blend(image, img_gray, alpha)


def saturation_list(var, images):
    """
    Perform color saturation on the list of given images.
    Args:
        var (float): variance.
        images (list): list of images to perform color saturation.
    Returns:
        (list): list of images that performed color saturation.
    """
    alpha = 1.0 + np.random.uniform(-var, var)

    out_images = []
    for image in images:
        img_gray = grayscale(image)
        out_images.append(blend(image, img_gray, alpha))
    return out_images


def brightness_list(var, images):
    """
    Perform color brightness on the given list of images.
    Args:
        var (float): variance.
        images (list): list of images to perform color brightness.
    Returns:
        (array): list of images that performed color brightness.
    """
    alpha = 1.0 + np.random.uniform(-var, var)

    out_images = []
    for image in images:
        img_bright = np.zeros(image.shape).astype(image.dtype)
        out_images.append(blend(image, img_bright, alpha))
    return out_images


def contrast_list(var, images):
    """
    Perform color contrast on the given list of images.
    Args:
        var (float): variance.
        images (list): list of images to perform color contrast.
    Returns:
        (array): image that performed color contrast.
    """
    alpha = 1.0 + np.random.uniform(-var, var)

    out_images = []
    for image in images:
        img_gray = grayscale(image)
        img_gray.fill(np.mean(img_gray[0]))
        out_images.append(blend(image, img_gray, alpha))
    return out_images


def color_jitter(image, img_brightness=0, img_contrast=0, img_saturation=0):
    """
    Perform color jitter on the given image.
    Args:
        image (array): image to perform color jitter.
        img_brightness (float): jitter ratio for brightness.
        img_contrast (float): jitter ratio for contrast.
        img_saturation (float): jitter ratio for saturation.
    Returns:
        image (array): the jittered image.
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
                image = brightness(img_brightness, image)
            elif jitter[order[idx]] == "contrast":
                image = contrast(img_contrast, image)
            elif jitter[order[idx]] == "saturation":
                image = saturation(img_saturation, image)
    return image
