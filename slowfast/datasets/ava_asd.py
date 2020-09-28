import os
import torch
import random

from common.utils.pathUtils import createFullPathTree, loadPickle
from slowfast.datasets import utils as utils
from slowfast.datasets import transform as transform
from slowfast.datasets.build import DATASET_REGISTRY
import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)

@DATASET_REGISTRY.register()
class Ava_asd(torch.utils.data.Dataset):
    """
    Ava - active speaker data loader. This loader samples sequences (aka windows) 
    of from a sequence of a tracklet. Each sequence is a single tracklet
    from a sequence of frames. A sequence is randomly sampled from teh collection of tracklets
    Random cropping is performed and flopping during training / validation
    No pre-processing is done for testing
    This loader can handle two types of input
    1) tensors - THe face ptches are already extracted into tensors. This requires minimal
    processing at run time as no video decoding is necessary but is expensive in size of data stored and read
    2) Videos. Preprocessed videos of teh face patches. Advantae is data files are small - but requires decoding
    at run time 
    """

    def __init__(self, cfg, mode, num_retries=10, labelSuffix=None):
        """
        Construct the Ava video loader with a given pandas table as pickle file. The table columns are:
        ```
        clip fid trId abel min max

        clip - video clip name 
        fid - Fram d (offset) starts at 0
        trId - Tracklet Id - integer from 0 - total number of tracklets
        min - First Frame Id with reference to the original video from which clips extracted
        max - ending frame Id with reference to the original video from which clips extracted

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
        ], "Split '{}' not supported for Ava-asd".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._num_retries = num_retries
        self.labelSuffix = labelSuffix
        if self.labelSuffix is None:
            labelName = "{}_LABEL_FILE_SUFFIX".format(mode.upper())
            self.labelSuffix = getattr(cfg.DATA, labelName) if hasattr(cfg.DATA, labelName) else cfg.DATA.LABEL_FILE_SUFFIX

        self.dataDesc = None
        self._construct_loader()
        
    def _construct_loader(self):
        path_to_file = createFullPathTree(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}{}".format(self.mode,  self.labelSuffix)
        )
        assert os.path.exists(path_to_file), "{} not found. Looking for data description pandas table".format(
            path_to_file
        )

        self.dataDesc = loadPickle(path_to_file)
        print("AvaAsd Load {} from {}".format(len(self.dataDesc), path_to_file))
        assert os.path.exists(self.cfg.DATA.PATH_PREFIX), "{} not found - path to data files"
        if self.cfg.DATA.DATA_TYPE == 'tensors':
            self.dataAccessFn = self.loadFromTensorFile
        else:
            raise Exception("Unknown data access type {} Expect one of [tensors] ".format(self.cfg.DATA.DATA_TYPE))

    def loadFromTensorFile(self, row):
        fid = row[self.cfg.DATA.FID_COL] - row[self.cfg.DATA.START_FRAME_COL]
        fid_start = fid-self.cfg.DATA.NUM_FRAMES
        if fid_start < 0:
            return None
        tensorPath = createFullPathTree(self.cfg.DATA.PATH_PREFIX,  '{}_tr_{}_st_{}_end_{}.pt'.format(
            row[self.cfg.DATA.CLIP_NAME_COL], row[self.cfg.DATA.TRACK_COL],
            row[self.cfg.DATA.START_FRAME_COL], row[self.cfg.DATA.END_FRAME_COL]))
        if not os.path.exists(tensorPath):
            return None
        
        dat = torch.load(tensorPath)
        # Sequence too short?
        if dat.shape[0] < fid:
            return None
        
        if dat[fid_start:fid,:,:,:].shape[0] != self.cfg.DATA.NUM_FRAMES:
            print("PATH {} start {} fid {} shape {}".format(tensorPath, fid_start, fid, dat[fid_start:fid,:,:,:].shape))
        return dat[fid_start:fid,:,:,:]

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
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
            temporal_sample_index = (
                self._spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
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

        # extract a sequnce of cfg.DATA_NUM_FRAMES. 
        # If given index is invalid will pick another random index
        for _ in range(self._num_retries):
            row = self.dataDesc.iloc[index]
            frames = self.dataAccessFn(row)
            # Access failed - try another 
            if frames is None:
                index = random.randint(0, len(self.dataDesc) - 1)
                continue

            if frames.shape[0] != self.cfg.DATA.NUM_FRAMES:
                print("FRAMES SHape {} fid {} start {}".format(frames.shape, row[self.cfg.DATA.FID_COL],  row[self.cfg.DATA.START_FRAME_COL]))
            # logger.info("Found Frame {}".format(frames.shape))
            # Perform color normalization.
            frames = utils.tensor_normalize(
                frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
            )
            # T H W C -> C T H W.
            frames = frames.permute(3, 0, 1, 2)
            # Perform data augmentation.
            # logger.info("S_DX {} MIN {} MAX {} SZE {} HFLIP {} SAMP {}".format(spatial_sample_index, min_scale, max_scale, crop_size, 
            #                                                      self.cfg.DATA.RANDOM_FLIP, self.cfg.DATA.INV_UNIFORM_SAMPLE ))
            frames = utils.spatial_sampling(
                frames,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            )

            label = row[self.cfg.DATA.LABEL_COL]
            frames = utils.pack_pathway_output(self.cfg, frames)
            return frames, label, index, {}
        else:
            raise RuntimeError(
                "Failed to fetch tensor after {} retries.".format(
                    self._num_retries
                )
            )

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self.dataDesc)
