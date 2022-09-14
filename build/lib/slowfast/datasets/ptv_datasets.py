# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import functools
import os
from typing import Dict
import torch
from torch.utils.data import (
    DistributedSampler,
    RandomSampler,
    SequentialSampler,
)
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    NormalizeVideo,
    RandomCropVideo,
    RandomHorizontalFlipVideo,
)

import slowfast.utils.logging as logging

from pytorchvideo.data import (
    Charades,
    LabeledVideoDataset,
    SSv2,
    make_clip_sampler,
)
from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    RandomShortSideScale,
    ShortSideScale,
    UniformCropVideo,
    UniformTemporalSubsample,
)

from . import utils as utils
from .build import DATASET_REGISTRY

logger = logging.get_logger(__name__)


class PTVDatasetWrapper(torch.utils.data.IterableDataset):
    """
    Wrapper for PyTorchVideo datasets.
    """

    def __init__(self, num_videos, clips_per_video, crops_per_clip, dataset):
        """
        Construct the dataset.

        Args:
            num_vidoes (int): number of videos in the dataset.
            clips_per_video (int): number of clips per video in the dataset.
            dataset (torch.utils.data.IterableDataset): a PyTorchVideo dataset.
        """
        self._clips_per_video = clips_per_video
        self._crops_per_clip = crops_per_clip
        self._num_videos = num_videos
        self.dataset = dataset

    def __next__(self):
        """
        Retrieves the next clip from the dataset.
        """
        return self.dataset.__next__()

    @property
    def sampler(self):
        """
        Returns:
            (torch.utils.data.Sampler): video sampler for the dataset.
        """
        return self.dataset.video_sampler

    def __len__(self):
        """
        Returns:
            (int): the number of clips per replica in the IterableDataset.
        """
        return len(self.sampler) * self._clips_per_video * self._crops_per_clip

    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of clips in total in the dataset.
        """
        return self._num_videos * self._clips_per_video * self._crops_per_clip

    def __iter__(self):
        return self


class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors. Each tensor
    corresponding to a unique pathway.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, x: torch.Tensor):
        return utils.pack_pathway_output(self.cfg, x)


class DictToTuple(torch.nn.Module):
    """
    Transform for converting output from dict to a tuple following PySlowFast
    dataset output format.
    """

    def __init__(self, num_clips, num_crops):
        super().__init__()
        self._num_clips = num_clips
        self._num_crops = num_crops

    def forward(self, x: Dict[str, torch.Tensor]):
        index = (
            x["video_index"] * self._num_clips * self._num_crops
            + x["clip_index"] * self._num_crops
            + x["aug_index"]
        )

        return x["video"], x["label"], index, {}


def div255(x):
    """
    Scale clip frames from [0, 255] to [0, 1].
    Args:
        x (Tensor): A tensor of the clip's RGB frames with shape:
            (channel, time, height, width).

    Returns:
        x (Tensor): Scaled tensor by divide 255.
    """
    return x / 255.0


@DATASET_REGISTRY.register()
def Ptvkinetics(cfg, mode):
    """
    Construct the Kinetics video loader with a given csv file. The format of
    the csv file is:
    ```
    path_to_video_1 label_1
    path_to_video_2 label_2
    ...
    path_to_video_N label_N
    ```
    For `train` and `val` mode, a single clip is randomly sampled from every video
    with random cropping, scaling, and flipping. For `test` mode, multiple clips are
    uniformaly sampled from every video with center cropping.
    Args:
        cfg (CfgNode): configs.
        mode (string): Options includes `train`, `val`, or `test` mode.
            For the train and val mode, the data loader will take data
            from the train or val set, and sample one clip per video.
            For the test mode, the data loader will take data from test set,
            and sample multiple clips per video.
    """
    # Only support train, val, and test mode.
    assert mode in [
        "train",
        "val",
        "test",
    ], "Split '{}' not supported".format(mode)

    logger.info("Constructing Ptvkinetics {}...".format(mode))

    clip_duration = (
        cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE / cfg.DATA.TARGET_FPS
    )
    path_to_file = os.path.join(
        cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(mode)
    )
    labeled_video_paths = LabeledVideoPaths.from_path(path_to_file)
    num_videos = len(labeled_video_paths)
    labeled_video_paths.path_prefix = cfg.DATA.PATH_PREFIX
    logger.info(
        "Constructing kinetics dataloader (size: {}) from {}".format(
            num_videos, path_to_file
        )
    )

    if mode in ["train", "val"]:
        num_clips = 1
        num_crops = 1

        transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(cfg.DATA.NUM_FRAMES),
                            Lambda(div255),
                            NormalizeVideo(cfg.DATA.MEAN, cfg.DATA.STD),
                            RandomShortSideScale(
                                min_size=cfg.DATA.TRAIN_JITTER_SCALES[0],
                                max_size=cfg.DATA.TRAIN_JITTER_SCALES[1],
                            ),
                            RandomCropVideo(cfg.DATA.TRAIN_CROP_SIZE),
                        ]
                        + (
                            [RandomHorizontalFlipVideo(p=0.5)]
                            if cfg.DATA.RANDOM_FLIP
                            else []
                        )
                        + [PackPathway(cfg)]
                    ),
                ),
                DictToTuple(num_clips, num_crops),
            ]
        )

        clip_sampler = make_clip_sampler("random", clip_duration)
        if cfg.NUM_GPUS > 1:
            video_sampler = DistributedSampler
        else:
            video_sampler = (
                RandomSampler if mode == "train" else SequentialSampler
            )
    else:
        num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS
        num_crops = cfg.TEST.NUM_SPATIAL_CROPS

        transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(cfg.DATA.NUM_FRAMES),
                            Lambda(div255),
                            NormalizeVideo(cfg.DATA.MEAN, cfg.DATA.STD),
                            ShortSideScale(
                                size=cfg.DATA.TRAIN_JITTER_SCALES[0]
                            ),
                        ]
                    ),
                ),
                UniformCropVideo(size=cfg.DATA.TEST_CROP_SIZE),
                ApplyTransformToKey(key="video", transform=PackPathway(cfg)),
                DictToTuple(num_clips, num_crops),
            ]
        )
        clip_sampler = make_clip_sampler(
            "constant_clips_per_video",
            clip_duration,
            num_clips,
            num_crops,
        )
        video_sampler = (
            DistributedSampler if cfg.NUM_GPUS > 1 else SequentialSampler
        )

    return PTVDatasetWrapper(
        num_videos=num_videos,
        clips_per_video=num_clips,
        crops_per_clip=num_crops,
        dataset=LabeledVideoDataset(
            labeled_video_paths=labeled_video_paths,
            clip_sampler=clip_sampler,
            video_sampler=video_sampler,
            transform=transform,
            decode_audio=False,
            decoder=cfg.DATA.DECODING_BACKEND,
        ),
    )


def process_charades_label(x, mode, num_classes):
    """
    Process the video label for Charades dataset. Use video-level label for
    training mode, otherwise use clip-level label. Then convert the label into
    a binary vector.
    Args:
        x (dict): a video clip including label index.
        mode (string): Options includes `train`, `val`, or `test` mode.
        num_classes (int): Number of classes in the dataset.

    Returns:
        x (dict): video clip with updated label information.
    """
    label = (
        utils.aggregate_labels(x["label"])
        if mode == "train"
        else x["video_label"]
    )
    x["label"] = torch.as_tensor(utils.as_binary_vector(label, num_classes))

    return x


def rgb2bgr(x):
    """
    Convert clip frames from RGB mode to BRG mode.
    Args:
        x (Tensor): A tensor of the clip's RGB frames with shape:
            (channel, time, height, width).

    Returns:
        x (Tensor): Converted tensor
    """
    return x[[2, 1, 0], ...]


@DATASET_REGISTRY.register()
def Ptvcharades(cfg, mode):
    """
    Construct PyTorchVideo Charades video loader.
    Load Charades data (frame paths, labels, etc. ) to Charades Dataset object.
    The dataset could be downloaded from Chrades official website
    (https://allenai.org/plato/charades/).
    Please see datasets/DATASET.md for more information about the data format.
    For `train` and `val` mode, a single clip is randomly sampled from every video
    with random cropping, scaling, and flipping. For `test` mode, multiple clips are
    uniformaly sampled from every video with center cropping.
    Args:
        cfg (CfgNode): configs.
        mode (string): Options includes `train`, `val`, or `test` mode.
            For the train and val mode, the data loader will take data
            from the train or val set, and sample one clip per video.
            For the test mode, the data loader will take data from test set,
            and sample multiple clips per video.
    """
    # Only support train, val, and test mode.
    assert mode in [
        "train",
        "val",
        "test",
    ], "Split '{}' not supported".format(mode)

    logger.info("Constructing Ptvcharades {}...".format(mode))

    clip_duration = (
        (cfg.DATA.NUM_FRAMES - 1) * cfg.DATA.SAMPLING_RATE + 1
    ) / cfg.DATA.TARGET_FPS

    if mode in ["train", "val"]:
        num_clips = 1
        num_crops = 1

        transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            Lambda(div255),
                            NormalizeVideo(cfg.DATA.MEAN, cfg.DATA.STD),
                            RandomShortSideScale(
                                min_size=cfg.DATA.TRAIN_JITTER_SCALES[0],
                                max_size=cfg.DATA.TRAIN_JITTER_SCALES[1],
                            ),
                            RandomCropVideo(cfg.DATA.TRAIN_CROP_SIZE),
                            Lambda(rgb2bgr),
                        ]
                        + (
                            [RandomHorizontalFlipVideo(p=0.5)]
                            if cfg.DATA.RANDOM_FLIP
                            else []
                        )
                        + [PackPathway(cfg)]
                    ),
                ),
                Lambda(
                    functools.partial(
                        process_charades_label,
                        mode=mode,
                        num_classes=cfg.MODEL.NUM_CLASSES,
                    )
                ),
                DictToTuple(num_clips, num_crops),
            ]
        )
        clip_sampler = make_clip_sampler("random", clip_duration)
        if cfg.NUM_GPUS > 1:
            video_sampler = DistributedSampler
        else:
            video_sampler = (
                RandomSampler if mode == "train" else SequentialSampler
            )
    else:
        num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS
        num_crops = cfg.TEST.NUM_SPATIAL_CROPS

        transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            Lambda(div255),
                            NormalizeVideo(cfg.DATA.MEAN, cfg.DATA.STD),
                            ShortSideScale(size=cfg.DATA.TEST_CROP_SIZE),
                        ]
                    ),
                ),
                UniformCropVideo(size=cfg.DATA.TEST_CROP_SIZE),
                Lambda(
                    functools.partial(
                        process_charades_label,
                        mode=mode,
                        num_classes=cfg.MODEL.NUM_CLASSES,
                    )
                ),
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [Lambda(rgb2bgr), PackPathway(cfg)],
                    ),
                ),
                DictToTuple(num_clips, num_crops),
            ]
        )
        clip_sampler = make_clip_sampler(
            "constant_clips_per_video",
            clip_duration,
            num_clips,
            num_crops,
        )
        video_sampler = (
            DistributedSampler if cfg.NUM_GPUS > 1 else SequentialSampler
        )

    data_path = os.path.join(cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(mode))
    dataset = Charades(
        data_path=data_path,
        clip_sampler=clip_sampler,
        video_sampler=video_sampler,
        transform=transform,
        video_path_prefix=cfg.DATA.PATH_PREFIX,
        frames_per_clip=cfg.DATA.NUM_FRAMES,
    )

    logger.info(
        "Constructing charades dataloader (size: {}) from {}".format(
            len(dataset._path_to_videos), data_path
        )
    )

    return PTVDatasetWrapper(
        num_videos=len(dataset._path_to_videos),
        clips_per_video=num_clips,
        crops_per_clip=num_crops,
        dataset=dataset,
    )


@DATASET_REGISTRY.register()
def Ptvssv2(cfg, mode):
    """
    Construct PyTorchVideo Something-Something v2 SSv2 video loader.
    Load SSv2 data (frame paths, labels, etc. ) to SSv2 Dataset object.
    The dataset could be downloaded from Chrades official website
    (https://20bn.com/datasets/something-something).
    Please see datasets/DATASET.md for more information about the data format.
    For training and validation, a single  clip is randomly sampled from every
    video with random cropping and scaling. For testing, multiple clips are
    uniformaly sampled from every video with uniform cropping. For uniform cropping,
    we take the left, center, and right crop if the width is larger than height,
    or take top, center, and bottom crop if the height is larger than the width.
    Args:
        cfg (CfgNode): configs.
        mode (string): Options includes `train`, `val`, or `test` mode.
    """
    # Only support train, val, and test mode.
    assert mode in [
        "train",
        "val",
        "test",
    ], "Split '{}' not supported".format(mode)

    logger.info("Constructing Ptvcharades {}...".format(mode))

    if mode in ["train", "val"]:
        num_clips = 1
        num_crops = 1

        transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            Lambda(div255),
                            NormalizeVideo(cfg.DATA.MEAN, cfg.DATA.STD),
                            RandomShortSideScale(
                                min_size=cfg.DATA.TRAIN_JITTER_SCALES[0],
                                max_size=cfg.DATA.TRAIN_JITTER_SCALES[1],
                            ),
                            RandomCropVideo(cfg.DATA.TRAIN_CROP_SIZE),
                            Lambda(rgb2bgr),
                        ]
                        + (
                            [RandomHorizontalFlipVideo(p=0.5)]
                            if cfg.DATA.RANDOM_FLIP
                            else []
                        )
                        + [PackPathway(cfg)]
                    ),
                ),
                DictToTuple(num_clips, num_crops),
            ]
        )
        clip_sampler = make_clip_sampler(
            "constant_clips_per_video",
            1,  # Put arbitrary duration as ssv2 always needs full video clip.
            num_clips,
            num_crops,
        )
        if cfg.NUM_GPUS > 1:
            video_sampler = DistributedSampler
        else:
            video_sampler = (
                RandomSampler if mode == "train" else SequentialSampler
            )
    else:
        assert cfg.TEST.NUM_ENSEMBLE_VIEWS == 1
        num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS
        num_crops = cfg.TEST.NUM_SPATIAL_CROPS

        transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            Lambda(div255),
                            NormalizeVideo(cfg.DATA.MEAN, cfg.DATA.STD),
                            ShortSideScale(size=cfg.DATA.TEST_CROP_SIZE),
                        ]
                    ),
                ),
                UniformCropVideo(size=cfg.DATA.TEST_CROP_SIZE),
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [Lambda(rgb2bgr), PackPathway(cfg)],
                    ),
                ),
                DictToTuple(num_clips, num_crops),
            ]
        )
        clip_sampler = make_clip_sampler(
            "constant_clips_per_video",
            1,  # Put arbitrary duration as ssv2 always needs full video clip.
            num_clips,
            num_crops,
        )
        video_sampler = (
            DistributedSampler if cfg.NUM_GPUS > 1 else SequentialSampler
        )

    label_name_file = os.path.join(
        cfg.DATA.PATH_TO_DATA_DIR, "something-something-v2-labels.json"
    )
    video_label_file = os.path.join(
        cfg.DATA.PATH_TO_DATA_DIR,
        "something-something-v2-{}.json".format(
            "train" if mode == "train" else "validation"
        ),
    )
    data_path = os.path.join(
        cfg.DATA.PATH_TO_DATA_DIR,
        "{}.csv".format("train" if mode == "train" else "val"),
    )
    dataset = SSv2(
        label_name_file=label_name_file,
        video_label_file=video_label_file,
        video_path_label_file=data_path,
        clip_sampler=clip_sampler,
        video_sampler=video_sampler,
        transform=transform,
        video_path_prefix=cfg.DATA.PATH_PREFIX,
        frames_per_clip=cfg.DATA.NUM_FRAMES,
        rand_sample_frames=mode == "train",
    )

    logger.info(
        "Constructing ssv2 dataloader (size: {}) from {}".format(
            len(dataset._path_to_videos), data_path
        )
    )

    return PTVDatasetWrapper(
        num_videos=len(dataset._path_to_videos),
        clips_per_video=num_clips,
        crops_per_clip=num_crops,
        dataset=dataset,
    )
