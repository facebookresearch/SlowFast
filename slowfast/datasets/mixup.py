# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
This implementation is based on
https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/mixup.py,
published under an Apache License 2.0.

COMMENT FROM ORIGINAL:
Mixup and Cutmix
Papers:
mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)
CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features (https://arxiv.org/abs/1905.04899) # NOQA
Code Reference:
CutMix: https://github.com/clovaai/CutMix-PyTorch
Hacked together by / Copyright 2020 Ross Wightman
"""

import numpy as np
import torch


def convert_to_one_hot(targets, num_classes, on_value=1.0, off_value=0.0):
    """
    This function converts target class indices to one-hot vectors, given the
    number of classes.
    Args:
        targets (loader): Class labels.
        num_classes (int): Total number of classes.
        on_value (float): Target Value for ground truth class.
        off_value (float): Target Value for other classes.This value is used for
            label smoothing.
    """

    targets = targets.long().view(-1, 1)
    return torch.full(
        (targets.size()[0], num_classes), off_value, device=targets.device
    ).scatter_(1, targets, on_value)


def mixup_target(target, num_classes, lam=1.0, smoothing=0.0):
    """
    This function converts target class indices to one-hot vectors, given the
    number of classes.
    Args:
        targets (loader): Class labels.
        num_classes (int): Total number of classes.
        lam (float): lamba value for mixup/cutmix.
        smoothing (float): Label smoothing value.
    """
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    target1 = convert_to_one_hot(
        target,
        num_classes,
        on_value=on_value,
        off_value=off_value,
    )
    target2 = convert_to_one_hot(
        target.flip(0),
        num_classes,
        on_value=on_value,
        off_value=off_value,
    )
    return target1 * lam + target2 * (1.0 - lam)


def rand_bbox(img_shape, lam, margin=0.0, count=None):
    """
    Generates a random square bbox based on lambda value.

    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
        count (int): Number of bbox to generate
    """
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)
    return yl, yh, xl, xh


def get_cutmix_bbox(img_shape, lam, correct_lam=True, count=None):
    """
    Generates the box coordinates for cutmix.

    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        correct_lam (bool): Apply lambda correction when cutmix bbox clipped by
            image borders.
        count (int): Number of bbox to generate
    """

    yl, yu, xl, xu = rand_bbox(img_shape, lam, count=count)
    if correct_lam:
        bbox_area = (yu - yl) * (xu - xl)
        lam = 1.0 - bbox_area / float(img_shape[-2] * img_shape[-1])
    return (yl, yu, xl, xu), lam


class MixUp:
    """
    Apply mixup and/or cutmix for videos at batch level.
    mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)
    CutMix: Regularization Strategy to Train Strong Classifiers with Localizable
        Features (https://arxiv.org/abs/1905.04899)
    """

    def __init__(
        self,
        mixup_alpha=1.0,
        cutmix_alpha=0.0,
        mix_prob=1.0,
        switch_prob=0.5,
        correct_lam=True,
        label_smoothing=0.1,
        num_classes=1000,
    ):
        """
        Args:
            mixup_alpha (float): Mixup alpha value.
            cutmix_alpha (float): Cutmix alpha value.
            mix_prob (float): Probability of applying mixup or cutmix.
            switch_prob (float): Probability of switching to cutmix instead of
                mixup when both are active.
            correct_lam (bool): Apply lambda correction when cutmix bbox
                clipped by image borders.
            label_smoothing (float): Apply label smoothing to the mixed target
                tensor. If label_smoothing is not used, set it to 0.
            num_classes (int): Number of classes for target.
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mix_prob = mix_prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.correct_lam = correct_lam

    def _get_mixup_params(self):
        lam = 1.0
        use_cutmix = False
        if np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0.0 and self.cutmix_alpha > 0.0:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mix = (
                    np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
                    if use_cutmix
                    else np.random.beta(self.mixup_alpha, self.mixup_alpha)
                )
            elif self.mixup_alpha > 0.0:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.cutmix_alpha > 0.0:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            lam = float(lam_mix)
        return lam, use_cutmix

    def _mix_batch(self, x):
        lam, use_cutmix = self._get_mixup_params()
        if lam == 1.0:
            return 1.0
        if use_cutmix:
            (yl, yh, xl, xh), lam = get_cutmix_bbox(
                x.shape,
                lam,
                correct_lam=self.correct_lam,
            )
            x[..., yl:yh, xl:xh] = x.flip(0)[..., yl:yh, xl:xh]
        else:
            x_flipped = x.flip(0).mul_(1.0 - lam)
            x.mul_(lam).add_(x_flipped)
        return lam

    def __call__(self, x, target):
        if self.mix_prob > 0.0:
            assert len(x) > 1, "Batch size should be greater than 1 for mixup."
            lam = self._mix_batch(x)
        elif self.mix_prob == 0.0:
            lam = 1.0
        else:
            raise NotImplementedError
        target = mixup_target(
            target, self.num_classes, lam, self.label_smoothing
        )
        return x, target
