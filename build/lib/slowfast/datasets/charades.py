#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
from itertools import chain as chain
import torch
import torch.utils.data

import slowfast.utils.logging as logging
from slowfast.utils.env import pathmgr

from . import utils as utils
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Charades(torch.utils.data.Dataset):
    """
    Charades video loader. Construct the Charades video loader, then sample
    clips from the videos. For training and validation, a single clip is randomly
    sampled from every video with random cropping, scaling, and flipping. For
    testing, multiple clips are uniformaly sampled from every video with uniform
    cropping. For uniform cropping, we take the left, center, and right crop if
    the width is larger than height, or take top, center, and bottom crop if the
    height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Load Charades data (frame paths, labels, etc. ) to a given Dataset object.
        The dataset could be downloaded from Chrades official website
        (https://allenai.org/plato/charades/).
        Please see datasets/DATASET.md for more information about the data format.
        Args:
            dataset (Dataset): a Dataset object to load Charades data to.
            mode (string): 'train', 'val', or 'test'.
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Charades ".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing Charades {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR,
            "{}.csv".format("train" if self.mode == "train" else "val"),
        )
        assert pathmgr.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )
        (self._path_to_videos, self._labels) = utils.load_image_lists(
            path_to_file, self.cfg.DATA.PATH_PREFIX, return_list=True
        )

        if self.mode != "train":
            # Form video-level labels from frame level annotations.
            self._labels = utils.convert_to_video_level_labels(self._labels)

        self._path_to_videos = list(
            chain.from_iterable(
                [[x] * self._num_clips for x in self._path_to_videos]
            )
        )
        self._labels = list(
            chain.from_iterable([[x] * self._num_clips for x in self._labels])
        )
        self._spatial_temporal_idx = list(
            chain.from_iterable(
                [range(self._num_clips) for _ in range(len(self._labels))]
            )
        )

        logger.info(
            "Charades dataloader constructed (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    def get_seq_frames(self, index):
        """
        Given the video index, return the list of indexs of sampled frames.
        Args:
            index (int): the video index.
        Returns:
            seq (list): the indexes of sampled frames from the video.
        """
        temporal_sample_index = (
            -1
            if self.mode in ["train", "val"]
            else self._spatial_temporal_idx[index]
            // self.cfg.TEST.NUM_SPATIAL_CROPS
        )
        num_frames = self.cfg.DATA.NUM_FRAMES
        sampling_rate = utils.get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )
        video_length = len(self._path_to_videos[index])
        assert video_length == len(self._labels[index])

        clip_length = (num_frames - 1) * sampling_rate + 1
        if temporal_sample_index == -1:
            if clip_length > video_length:
                start = random.randint(video_length - clip_length, 0)
            else:
                start = random.randint(0, video_length - clip_length)
        else:
            gap = float(max(video_length - clip_length, 0)) / (
                self.cfg.TEST.NUM_ENSEMBLE_VIEWS - 1
            )
            start = int(round(gap * temporal_sample_index))

        seq = [
            max(min(start + i * sampling_rate, video_length - 1), 0)
            for i in range(num_frames)
        ]

        return seq

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video frames can be fetched.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): the index of the video.
            time index (zero): The time index is currently not supported.
            {} extra data, currently not supported
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, self._num_yielded = index
            if self.cfg.MULTIGRID.SHORT_CYCLE:
                index, short_cycle_idx = index

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["test"]:
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                self._spatial_temporal_idx[index]
                % self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        seq = self.get_seq_frames(index)
        frames = torch.as_tensor(
            utils.retry_load_images(
                [self._path_to_videos[index][frame] for frame in seq],
                self._num_retries,
            )
        )

        label = utils.aggregate_labels(
            [self._labels[index][i] for i in range(seq[0], seq[-1] + 1)]
        )
        label = torch.as_tensor(
            utils.as_binary_vector(label, self.cfg.MODEL.NUM_CLASSES)
        )

        # Perform color normalization.
        frames = utils.tensor_normalize(
            frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
        )
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)
        # Perform data augmentation.
        frames = utils.spatial_sampling(
            frames,
            spatial_idx=spatial_sample_index,
            min_scale=min_scale,
            max_scale=max_scale,
            crop_size=crop_size,
            random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
            inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
        )
        frames = utils.pack_pathway_output(self.cfg, frames)
        return frames, label, index, 0, {}

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return self.num_videos

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
