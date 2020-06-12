#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Video models."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import slowfast.utils.weight_init_helper as init_helper
from slowfast.models.batchnorm_helper import get_norm
from slowfast.utils import misc

from . import head_helper, resnet_helper, stem_helper
from .build import MODEL_REGISTRY

# Number of blocks for different stages given the model depth.
_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "c2d": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "c2d_nopool": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[1]],  # res4 temporal kernel.
        [[1]],  # res5 temporal kernel.
    ],
    "i3d": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "i3d_nopool": [
        [[5]],  # conv1 temporal kernel.
        [[3]],  # res2 temporal kernel.
        [[3, 1]],  # res3 temporal kernel.
        [[3, 1]],  # res4 temporal kernel.
        [[1, 3]],  # res5 temporal kernel.
    ],
    "slow": [
        [[1]],  # conv1 temporal kernel.
        [[1]],  # res2 temporal kernel.
        [[1]],  # res3 temporal kernel.
        [[3]],  # res4 temporal kernel.
        [[3]],  # res5 temporal kernel.
    ],
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ],
    "avslowfast": [
        [[1], [5], [1]],  # conv1 temp kernel for slow, fast and audio pathway.
        [[1], [3], [1]],  # res2 temp kernel for slow, fast and audio pathway.
        [[1], [3], [1]],  # res3 temp kernel for slow, fast and audio pathway.
        [[3], [3], [1]],  # res4 temp kernel for slow, fast and audio pathway.
        [[3], [3], [1]],  # res5 temp kernel for slow, fast and audio pathway.
    ],
}

_POOL1 = {
    "c2d": [[2, 1, 1]],
    "c2d_nopool": [[1, 1, 1]],
    "i3d": [[2, 1, 1]],
    "i3d_nopool": [[1, 1, 1]],
    "slow": [[1, 1, 1]],
    "slowfast": [[1, 1, 1], [1, 1, 1]],
    "avslowfast": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
}


class AVS(nn.Module):
    """
    Compute Audio-Visual synchronization loss
    """
    
    def __init__(self, ref_dim, query_dim, proj_dim, num_gpus, num_shards):
        super(AVS, self).__init__()
        
        # initialize fc projection layers
        self.proj_dim = proj_dim
        self.ref_fc = nn.Linear(ref_dim, proj_dim, bias=True)
        self.query_fc = nn.Linear(query_dim, proj_dim, bias=True)
        self.num_gpus = num_gpus
        self.num_shards = num_shards
    
    
    def contrastive_loss(self, ref, pos, neg, audio_mask, margin):
        """
        https://arxiv.org/abs/1807.00230
        """
        # assert pos.shape[0] == neg.shape[0]
        N = torch.sum(audio_mask)
        
        pos_dist = ref - pos
        neg_dist = ref - neg
        pos_dist = pos_dist[audio_mask]
        neg_dist = neg_dist[audio_mask]
        
        pos_loss = torch.norm(pos_dist)**2
        neg_dist = torch.norm(neg_dist, dim=1)
        neg_loss = torch.sum(torch.clamp(margin - neg_dist, min=0)**2)
        loss = (pos_loss + neg_loss) / (2*N + 1e-8)
        return loss
        
        
    def forward(self, ref, pos, neg, audio_mask, norm='L2', margin=0.99):
        # reduce T, H, W dims
        ref = torch.mean(ref, (2, 3, 4))
        pos = torch.mean(pos, (2, 3, 4))
        neg = torch.mean(neg, (2, 3, 4))
        
        # projection
        ref = self.ref_fc(ref)
        pos = self.query_fc(pos)
        neg = self.query_fc(neg)
        
        # normalize
        if norm == 'L2':
            ref = torch.nn.functional.normalize(ref, p=2, dim=1)
            pos = torch.nn.functional.normalize(pos, p=2, dim=1)
            neg = torch.nn.functional.normalize(neg, p=2, dim=1)
            # scale data so that ||x-y||^2 fall in [0, 1]
            ref = ref * 0.5
            pos = pos * 0.5
            neg = neg * 0.5
        elif norm == 'Tanh':
            scale = 1.0 / self.proj_dim
            ref = torch.nn.functional.tanh(ref) * scale
            pos = torch.nn.functional.tanh(pos) * scale
            neg = torch.nn.functional.tanh(neg) * scale
        
        # contrstive loss
        loss = self.contrastive_loss(ref, pos, neg, audio_mask, margin)
        
        # scale the loss with nGPUs and shards
        # loss = loss / float(self.num_gpus * self.num_shards)
        loss = loss / float(self.num_shards)
        
        return loss


class FuseAV(nn.Module):
    """
    Fuses the information from audio to visual pathways.
    """
    
    def __init__(
        self,
        # slow pathway
        dim_in_s,
        # fast pathway
        dim_in_f,
        fusion_conv_channel_ratio_f,
        fusion_kernel_f,
        alpha_f,
        # audio pathway
        dim_in_a,
        fusion_conv_channel_mode_a,
        fusion_conv_channel_dim_a,
        fusion_conv_channel_ratio_a,
        fusion_kernel_a,
        alpha_a,
        conv_num_a,
        # fusion connections
        use_fs_fusion,
        use_afs_fusion,
        # AVS
        use_avs,
        avs_proj_dim,
        # general params
        num_gpus=1,
        num_shards=1,
        eps=1e-5,
        bn_mmt=0.1,
        inplace_relu=True,
    ):
        """
        Perform A2TS fusion described in AVSlowFast paper.
        """
        super(FuseAV, self).__init__()
        self.conv_num_a = conv_num_a
        self.use_fs_fusion = use_fs_fusion
        self.use_afs_fusion = use_afs_fusion
                
        # perform F->S fusion
        if use_fs_fusion:
            self.conv_f2s = nn.Conv3d(
                dim_in_f,
                dim_in_f * fusion_conv_channel_ratio_f,
                kernel_size=[fusion_kernel_f, 1, 1],
                stride=[alpha_f, 1, 1],
                padding=[fusion_kernel_f // 2, 0, 0],
                bias=False,
            )
            self.bn_f2s = nn.BatchNorm3d(
                dim_in_f * fusion_conv_channel_ratio_f, eps=eps, momentum=bn_mmt
            )
            self.relu_f2s = nn.ReLU(inplace_relu)
        
        # perform A->FS fusion
        if fusion_conv_channel_mode_a == 'ByDim':
            afs_fusion_interm_dim = int(fusion_conv_channel_dim_a)
        elif fusion_conv_channel_mode_a == 'ByRatio':
            afs_fusion_interm_dim = int(dim_in_a * fusion_conv_channel_ratio_a)
        else:
            raise RuntimeError
        if use_afs_fusion:
            cur_dim_in = dim_in_a
            for idx in range(conv_num_a):
                if idx == conv_num_a - 1:
                    cur_stride = alpha_a
                    cur_dim_out = int(dim_in_f * fusion_conv_channel_ratio_f \
                                      + dim_in_s)
                else:
                    cur_stride = 1
                    cur_dim_out = afs_fusion_interm_dim
                conv_a2fs = nn.Conv3d(
                    cur_dim_in,
                    cur_dim_out,
                    kernel_size=[1, fusion_kernel_a, 1],
                    stride=[1, cur_stride, 1],
                    padding=[0, fusion_kernel_a // 2, 0],
                    bias=False,
                )
                bn_a2fs = nn.BatchNorm3d(
                    cur_dim_out, eps=eps, momentum=bn_mmt
                )
                relu_a2fs = nn.ReLU(inplace_relu)
                self.add_module('conv_a2fs_%d' % idx, conv_a2fs)
                self.add_module('bn_a2fs_%d' % idx, bn_a2fs)
                self.add_module('relu_a2fs_%d' % idx, relu_a2fs)
                cur_dim_in = cur_dim_out
        
        dim_in_a = int(dim_in_f * fusion_conv_channel_ratio_f + dim_in_s)
        
        # optionally compute audiovisual synchronization loss
        if use_avs:
            self.avs = AVS(
                dim_in_f * fusion_conv_channel_ratio_f + dim_in_s, 
                dim_in_a, 
                avs_proj_dim,
                num_gpus,
                num_shards,
            )
            
            
    def forward(self, x, get_misaligned_audio=False, mode='AFS'):
        """
        Forward function for audiovisual fusion, note that it currently only 
        supports A->FS fusion mode (which is the default used in AVSlowFast paper)
        Args:
            x (list): contains slow, fast and audio features
            get_misaligned_audio (bool): whether misaligned audio is carried in x
            mode:
                AFS  -- fuse audio, fast and slow
                AS   -- fuse audio and slow 
                FS   -- fuse fast and slow 
                NONE -- do not fuse at all
        """
        x_s = x[0]
        x_f = x[1]
        x_a = x[2]
        fuse = x_s
        cache = {}
        
        if mode != 'NONE':
            fs_proc, afs_proc = None, None
            
            # F->S
            if self.use_fs_fusion:
                if 'F' in mode:
                    fs_proc = self.conv_f2s(x_f)
                    fs_proc = self.bn_f2s(fs_proc)
                    fs_proc = self.relu_f2s(fs_proc)
                    fs_proc = torch.cat([fuse, fs_proc], 1)
                    fuse = fs_proc
                    cache['fs'] = fs_proc
                    
            # A->FS
            if self.use_afs_fusion:
                # [N C 1 T F] -> [N C 1 T 1]
                afs_proc = torch.mean(x_a, dim=-1, keepdim=True)
                for idx in range(self.conv_num_a):
                    conv = getattr(self, 'conv_a2fs_%d' % idx)
                    bn = getattr(self, 'bn_a2fs_%d' % idx)
                    relu = getattr(self, 'relu_a2fs_%d' % idx)
                    afs_proc = conv(afs_proc)
                    afs_proc = bn(afs_proc)
                    afs_proc = relu(afs_proc)
                if get_misaligned_audio:
                    afs_proc_pos, afs_proc_neg = torch.chunk(afs_proc, 2, dim=0)
                    cache['a_pos'] = afs_proc_pos
                    cache['a_neg'] = afs_proc_neg
                else:
                    afs_proc_pos = afs_proc
                if 'A' in mode:
                    # [N C 1 T 1] -> [N C T 1 1]
                    afs_proc_pos = afs_proc_pos.permute(0, 1, 3, 2, 4) 
                    afs_proc_pos = afs_proc_pos + fuse
                    fuse = afs_proc_pos
                
        return [fuse, x_f, x_a], cache


class FuseFastToSlow(nn.Module):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(
        self,
        dim_in,
        fusion_conv_channel_ratio,
        fusion_kernel,
        alpha,
        eps=1e-5,
        bn_mmt=0.1,
        inplace_relu=True,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
        """
        super(FuseFastToSlow, self).__init__()
        self.conv_f2s = nn.Conv3d(
            dim_in,
            dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            bias=False,
        )
        self.bn = nn.BatchNorm3d(
            dim_in * fusion_conv_channel_ratio, eps=eps, momentum=bn_mmt
        )
        self.relu = nn.ReLU(inplace_relu)


    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]


@MODEL_REGISTRY.register()
class AVSlowFast(nn.Module):
    """
    Model builder for AVSlowFast network.
    Fanyi Xiao, Yong Jae Lee, Kristen Grauman, Jitendra Malik, Christoph Feichtenhofer.
    "Audiovisual Slowfast Networks for Video Recognition."
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(AVSlowFast, self).__init__()
        self.norm_module = get_norm(cfg)
        self.num_pathways = 3
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )


    def _construct_network(self, cfg):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()
        
        self.DROPPATHWAY_RATE = cfg.SLOWFAST.DROPPATHWAY_RATE
        self.FS_FUSION = cfg.SLOWFAST.FS_FUSION
        self.AFS_FUSION = cfg.SLOWFAST.AFS_FUSION
        self.GET_MISALIGNED_AUDIO = cfg.DATA.GET_MISALIGNED_AUDIO
        self.AVS_FLAG = cfg.SLOWFAST.AVS_FLAG
        self.AVS_PROJ_DIM = cfg.SLOWFAST.AVS_PROJ_DIM
        self.AVS_VAR_THRESH = cfg.SLOWFAST.AVS_VAR_THRESH
        self.AVS_DUPLICATE_THRESH = cfg.SLOWFAST.AVS_DUPLICATE_THRESH
        
        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]
        tf_trans_func = [cfg.RESNET.TRANS_FUNC] * 2 + \
                        [cfg.RESNET.AUDIO_TRANS_FUNC]
        trans_func = [tf_trans_func] * cfg.RESNET.AUDIO_TRANS_NUM + \
            [cfg.RESNET.TRANS_FUNC] * (4 - cfg.RESNET.AUDIO_TRANS_NUM)

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group
        out_dim_ratio = (
            cfg.SLOWFAST.BETA_INV // cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO
        )

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]
        
        if cfg.SLOWFAST.AU_REDUCE_TF_DIM:
            tf_stride = 2
        else:
            tf_stride = 1
        tf_dim_reduction = 1

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[
                width_per_group, 
                width_per_group // cfg.SLOWFAST.BETA_INV, 
                width_per_group // cfg.SLOWFAST.AU_BETA_INV
            ],
            kernel=[
                temp_kernel[0][0] + [7, 7], 
                temp_kernel[0][1] + [7, 7], 
                [temp_kernel[0][2] + [9, 1], temp_kernel[0][2] + [1, 9]],
            ],
            stride=[[1, 2, 2], [1, 2, 2], [[1, 1, 1], [1, 1, 1]]],
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
                [[temp_kernel[0][2][0] // 2, 4, 0], [temp_kernel[0][2][0] // 2, 0, 4]],
            ],
            stride_pool=[True, True, False],
        )
        
        if self.FS_FUSION[0] or self.AFS_FUSION[0]:
            self.s1_fuse = FuseAV(
                # Slow
                width_per_group,
                # Fast
                width_per_group // cfg.SLOWFAST.BETA_INV,
                cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
                cfg.SLOWFAST.FUSION_KERNEL_SZ,
                cfg.SLOWFAST.ALPHA,
                # Audio
                width_per_group // cfg.SLOWFAST.AU_BETA_INV,
                cfg.SLOWFAST.AU_FUSION_CONV_CHANNEL_MODE,
                cfg.SLOWFAST.AU_FUSION_CONV_CHANNEL_DIM,
                cfg.SLOWFAST.AU_FUSION_CONV_CHANNEL_RATIO,
                cfg.SLOWFAST.AU_FUSION_KERNEL_SZ,
                cfg.SLOWFAST.AU_ALPHA // tf_dim_reduction,
                cfg.SLOWFAST.AU_FUSION_CONV_NUM,
                # Fusion connections
                self.FS_FUSION[0],
                self.AFS_FUSION[0],
                # AVS
                self.AVS_FLAG[0],
                self.AVS_PROJ_DIM,
                # nGPUs and shards
                num_gpus=cfg.NUM_GPUS,
                num_shards=cfg.NUM_SHARDS,
            )
        
        slow_dim = width_per_group + \
            (width_per_group // out_dim_ratio if self.FS_FUSION[0] else 0)
        self.s2 = resnet_helper.ResStage(
            dim_in=[
                slow_dim,
                width_per_group // cfg.SLOWFAST.BETA_INV,
                width_per_group // cfg.SLOWFAST.AU_BETA_INV,
            ],
            dim_out=[
                width_per_group * 4,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
                width_per_group * 4 // cfg.SLOWFAST.AU_BETA_INV,
            ],
            dim_inner=[
                dim_inner, 
                dim_inner // cfg.SLOWFAST.BETA_INV, 
                dim_inner // cfg.SLOWFAST.AU_BETA_INV
            ],
            temp_kernel_sizes=temp_kernel[1],
            stride=[1] * 3,
            num_blocks=[d2] * 3,
            num_groups=[num_groups] * 3,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=trans_func[0],
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )
        if self.FS_FUSION[1] or self.AFS_FUSION[1]:
            self.s2_fuse = FuseAV(
                # Slow
                width_per_group * 4,
                # Fast
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
                cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
                cfg.SLOWFAST.FUSION_KERNEL_SZ,
                cfg.SLOWFAST.ALPHA,
                # Audio
                width_per_group * 4 // cfg.SLOWFAST.AU_BETA_INV,
                cfg.SLOWFAST.AU_FUSION_CONV_CHANNEL_MODE,
                cfg.SLOWFAST.AU_FUSION_CONV_CHANNEL_DIM,
                cfg.SLOWFAST.AU_FUSION_CONV_CHANNEL_RATIO,
                cfg.SLOWFAST.AU_FUSION_KERNEL_SZ,
                cfg.SLOWFAST.AU_ALPHA // tf_dim_reduction,
                cfg.SLOWFAST.AU_FUSION_CONV_NUM,
                # Fusion connections
                self.FS_FUSION[1],
                self.AFS_FUSION[1],
                # AVS
                self.AVS_FLAG[1],
                self.AVS_PROJ_DIM,
                # nGPUs and shards
                num_gpus=cfg.NUM_GPUS,
                num_shards=cfg.NUM_SHARDS,
            )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)
            
        slow_dim = width_per_group * 4 + \
            (width_per_group * 4 // out_dim_ratio if self.FS_FUSION[1] else 0)
        self.s3 = resnet_helper.ResStage(
            dim_in=[
                slow_dim,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
                width_per_group * 4 // cfg.SLOWFAST.AU_BETA_INV,
            ],
            dim_out=[
                width_per_group * 8,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
                width_per_group * 8 // cfg.SLOWFAST.AU_BETA_INV,
            ],
            dim_inner=[
                dim_inner * 2, 
                dim_inner * 2 // cfg.SLOWFAST.BETA_INV,
                dim_inner * 2 // cfg.SLOWFAST.AU_BETA_INV
            ],
            temp_kernel_sizes=temp_kernel[2],
            stride=[2, 2, tf_stride],
            num_blocks=[d3] * 3,
            num_groups=[num_groups] * 3,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=trans_func[1],
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )
        tf_dim_reduction *= tf_stride
        
        if self.FS_FUSION[2] or self.AFS_FUSION[2]:
            self.s3_fuse = FuseAV(
                # Slow
                width_per_group * 8,
                # Fast
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
                cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
                cfg.SLOWFAST.FUSION_KERNEL_SZ,
                cfg.SLOWFAST.ALPHA,
                # Audio
                width_per_group * 8 // cfg.SLOWFAST.AU_BETA_INV,
                cfg.SLOWFAST.AU_FUSION_CONV_CHANNEL_MODE,
                cfg.SLOWFAST.AU_FUSION_CONV_CHANNEL_DIM,
                cfg.SLOWFAST.AU_FUSION_CONV_CHANNEL_RATIO,
                cfg.SLOWFAST.AU_FUSION_KERNEL_SZ,
                cfg.SLOWFAST.AU_ALPHA // tf_dim_reduction,
                cfg.SLOWFAST.AU_FUSION_CONV_NUM,
                # Fusion connections
                self.FS_FUSION[2],
                self.AFS_FUSION[2],
                # AVS
                self.AVS_FLAG[2],
                self.AVS_PROJ_DIM,
                # nGPUs and shards
                num_gpus=cfg.NUM_GPUS,
                num_shards=cfg.NUM_SHARDS,
            )

        slow_dim = width_per_group * 8 + \
            (width_per_group * 8 // out_dim_ratio if self.FS_FUSION[2] else 0)
        self.s4 = resnet_helper.ResStage(
            dim_in=[
                slow_dim,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
                width_per_group * 8 // cfg.SLOWFAST.AU_BETA_INV,
            ],
            dim_out=[
                width_per_group * 16,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
                width_per_group * 16 // cfg.SLOWFAST.AU_BETA_INV,
            ],
            dim_inner=[
                dim_inner * 4, 
                dim_inner * 4 // cfg.SLOWFAST.BETA_INV,
                dim_inner * 4 // cfg.SLOWFAST.AU_BETA_INV
            ],
            temp_kernel_sizes=temp_kernel[3],
            stride=[2, 2, tf_stride],
            num_blocks=[d4] * 3,
            num_groups=[num_groups] * 3,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=trans_func[2],
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )
        tf_dim_reduction *= tf_stride
        
        if self.FS_FUSION[3] or self.AFS_FUSION[3]:
            self.s4_fuse = FuseAV(
                # Slow
                width_per_group * 16,
                # Fast
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
                cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
                cfg.SLOWFAST.FUSION_KERNEL_SZ,
                cfg.SLOWFAST.ALPHA,
                # Audio
                width_per_group * 16 // cfg.SLOWFAST.AU_BETA_INV,
                cfg.SLOWFAST.AU_FUSION_CONV_CHANNEL_MODE,
                cfg.SLOWFAST.AU_FUSION_CONV_CHANNEL_DIM,
                cfg.SLOWFAST.AU_FUSION_CONV_CHANNEL_RATIO,
                cfg.SLOWFAST.AU_FUSION_KERNEL_SZ,
                cfg.SLOWFAST.AU_ALPHA // tf_dim_reduction,
                cfg.SLOWFAST.AU_FUSION_CONV_NUM,
                # Fusion connections
                self.FS_FUSION[3],
                self.AFS_FUSION[3],
                # AVS
                self.AVS_FLAG[3],
                self.AVS_PROJ_DIM,
                # nGPUs and shards
                num_gpus=cfg.NUM_GPUS,
                num_shards=cfg.NUM_SHARDS,
            )
        
        slow_dim = width_per_group * 16 + \
            (width_per_group * 16 // out_dim_ratio if self.FS_FUSION[3] else 0)
        self.s5 = resnet_helper.ResStage(
            dim_in=[
                slow_dim,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
                width_per_group * 16 // cfg.SLOWFAST.AU_BETA_INV,
            ],
            dim_out=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                width_per_group * 32 // cfg.SLOWFAST.AU_BETA_INV,
            ],
            dim_inner=[
                dim_inner * 8, 
                dim_inner * 8 // cfg.SLOWFAST.BETA_INV,
                dim_inner * 8 // cfg.SLOWFAST.AU_BETA_INV,
            ],
            temp_kernel_sizes=temp_kernel[4],
            stride=[2, 2, tf_stride],
            num_blocks=[d5] * 3,
            num_groups=[num_groups] * 3,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=trans_func[3],
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )
        tf_dim_reduction *= tf_stride
        
        # setup AVS for pool5 output
        if self.AVS_FLAG[4]:
            # this FuseAV object is used for compute AVS loss only
            self.s5_fuse = FuseAV(
                # Slow
                width_per_group * 32,
                # Fast
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
                cfg.SLOWFAST.FUSION_KERNEL_SZ,
                cfg.SLOWFAST.ALPHA,
                # Audio
                width_per_group * 32 // cfg.SLOWFAST.AU_BETA_INV,
                cfg.SLOWFAST.AU_FUSION_CONV_CHANNEL_MODE,
                cfg.SLOWFAST.AU_FUSION_CONV_CHANNEL_DIM,
                cfg.SLOWFAST.AU_FUSION_CONV_CHANNEL_RATIO,
                cfg.SLOWFAST.AU_FUSION_KERNEL_SZ,
                cfg.SLOWFAST.AU_ALPHA // tf_dim_reduction,
                cfg.SLOWFAST.AU_FUSION_CONV_NUM,
                # Fusion connections
                True,
                True,
                # AVS
                self.AVS_FLAG[4],
                self.AVS_PROJ_DIM,
                # nGPUs and shards
                num_gpus=cfg.NUM_GPUS,
                num_shards=cfg.NUM_SHARDS,
            )

        self.head = head_helper.ResNetBasicHead(
            dim_in=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                width_per_group * 32 // cfg.SLOWFAST.AU_BETA_INV,
            ],
            num_classes=cfg.MODEL.NUM_CLASSES,
            pool_size=[
                [
                    cfg.DATA.NUM_FRAMES
                    // cfg.SLOWFAST.ALPHA
                    // pool_size[0][0],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                ],
                [
                    cfg.DATA.NUM_FRAMES // pool_size[1][0],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[1][1],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[1][2],
                ],
                [
                    1,
                    cfg.DATA.AUDIO_FRAME_NUM // tf_dim_reduction,
                    cfg.DATA.AUDIO_MEL_NUM // tf_dim_reduction,
                ],
            ],
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
        )
    
    
    def freeze_bn(self, freeze_bn_affine):
        """
        Freeze the BN parameters
        """
        print("Freezing Mean/Var of BatchNorm.")
        if freeze_bn_affine:
            print("Freezing Weight/Bias of BatchNorm.")
        for name, m in self.named_modules():
            if isinstance(m, nn.BatchNorm1d) or \
                isinstance(m, nn.BatchNorm2d) or \
                isinstance(m, nn.BatchNorm3d):
                # if 'pathway2' in name or 'a2fs' in name:
                #     continue
                m.eval()
                if freeze_bn_affine:
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
    
    
    def gen_fusion_avs_pattern(self):
        """
        This function generates a fusion pattern and a avs loss compute pattern.
        Specifically, fusion pattern is determined by both pre-defined fusion 
        connections between Slow/Fast/Audio, and the flag of whether to drop the 
        audio pathway, which is generated on the fly. 
        For AVS pattern, it is determined by fusion pattern. For example, if we 
        decided to have AFS fusion pattern like [False, False, True, True], 
        which means to have fusion between audio and visual after res3 and res4,
        and let's say our AFS_FUSION is [False, False, False, True], then we will 
        not compute AVS loss anywhere. This is because since we have fused audio
        into visual at res3, any visual features after this has already "seen" 
        audio features and the problem of telling whether audio and visual is in-sync
        will be trivial.
        """
        is_drop = self.training and random.random() < self.DROPPATHWAY_RATE
        fs_fusion = self.FS_FUSION
        afs_fusion = self.AFS_FUSION
        runtime_afs_fusion = []
        fusion_pattern, avs_pattern = [], []
        
        for idx in range(4):
            # If a junction has both audiovisual fusion and slowfast fusion,
            # we call it 'AFS'. If it only has slowfast fusion, we call it 'FS'.
            # If it only has audio and slow fusion, we call it 'AS'
            cur_fs_fusion = fs_fusion[idx]
            cur_afs_fusion = afs_fusion[idx] and not is_drop
            if cur_fs_fusion and cur_afs_fusion:
                fusion_pattern.append('AFS')
            elif cur_fs_fusion and not cur_afs_fusion:
                fusion_pattern.append('FS')
            elif not cur_fs_fusion and cur_afs_fusion:
                fusion_pattern.append('AS')
            else:
                fusion_pattern.append('NONE')
            runtime_afs_fusion.append(cur_afs_fusion)
        
        # compute the earliest audiovisual fusion, so that we don't do AVS
        # for any stage later than that
        earliest_afs = 4
        for idx in range(3, -1, -1):
            if runtime_afs_fusion[idx]:
                earliest_afs = idx
        
        for idx in range(5):
            if idx <= earliest_afs and self.AVS_FLAG[idx]:
                avs_pattern.append(True)
            else:
                avs_pattern.append(False)
        
        return fusion_pattern, avs_pattern
    
    
    def move_C_to_N(self, x):
        """
        Assume x is with shape [N C T H W], this function merges C into N which 
        results in shape [N*C 1 T H W]
        """
        N, C, T, H, W = x[2].size()
        x[2] = x[2].reshape(N*C, 1, T, H, W)
        return x
    
    
    def filter_duplicates(self, x):
        """
        Compute a valid mask for near-duplicates and near-zero audios
        """
        mask = None
        if self.GET_MISALIGNED_AUDIO:
            with torch.no_grad():
                audio = x[2]
                N, C, T, H, W = audio.size()
                audio = audio.reshape(N//2, C*2, -1)
                # remove pairs that are near-zero
                audio_std = torch.std(audio, dim=2) ** 2
                mask = audio_std > self.AVS_VAR_THRESH
                mask = mask[:, 0] * mask[:, 1]
                # remove pairs that are near-duplicate
                audio = F.normalize(audio, dim=2)
                similarity = audio[:, 0, :] * audio[:, 1, :]
                similarity = torch.sum(similarity, dim=1)
                similarity = similarity < self.AVS_DUPLICATE_THRESH
                # integrate var and dup mask
                mask = mask * similarity
                # mask = mask.float()
        return mask
    
    
    def get_pos_audio(self, x):
        """
        Slice the data and only take the first half 
        along the first dim for positive data
        """
        x[2], _ = torch.chunk(x[2], 2, dim=0)
        return x
    
    
    def avs_forward(self, features, audio_mask):
        """
        Forward for AVS loss
        """
        loss_list = {}
        avs_pattern = features['avs_pattern']
        for idx in range(5):
            if avs_pattern[idx]:
                a_pos = features['s{}_a_pos'.format(idx + 1)]
                a_neg = features['s{}_a_neg'.format(idx + 1)]
                fs = features['s{}_fs'.format(idx + 1)]
                fuse = getattr(self, 's{}_fuse'.format(idx + 1))
                avs = getattr(fuse, 'avs')
                loss = avs(fs, a_pos, a_neg, audio_mask)
                loss_list['s{}_avs'.format(idx + 1)] = loss
        return loss_list
        
        
    def forward(self, x):
        # generate fusion pattern
        fusion_pattern, avs_pattern = self.gen_fusion_avs_pattern()
        
        # tackle misaligned logmel
        if self.GET_MISALIGNED_AUDIO:
            x = self.move_C_to_N(x)
        
        # generate mask for audio
        audio_mask = self.filter_duplicates(x)
        
        # initialize feature list
        features = {'avs_pattern': avs_pattern}
        
        # execute forward
        x = self.s1(x)
        if self.FS_FUSION[0] or self.AFS_FUSION[0]:
            x, interm_feat = self.s1_fuse(
                x, 
                get_misaligned_audio=self.GET_MISALIGNED_AUDIO, 
                mode=fusion_pattern[0],
            )
            features = misc.update_dict_with_prefix(
                features, 
                interm_feat, 
                prefix='s1_'
            )
        x = self.s2(x)
        if self.FS_FUSION[1] or self.AFS_FUSION[1]:
            x, interm_feat = self.s2_fuse(
                x, 
                get_misaligned_audio=self.GET_MISALIGNED_AUDIO, 
                mode=fusion_pattern[1],
            )
            features = misc.update_dict_with_prefix(
                features, 
                interm_feat, 
                prefix='s2_'
            )        
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        if self.FS_FUSION[2] or self.AFS_FUSION[2]:
            x, interm_feat = self.s3_fuse(
                x, 
                get_misaligned_audio=self.GET_MISALIGNED_AUDIO, 
                mode=fusion_pattern[2],
            )
            features = misc.update_dict_with_prefix(
                features, 
                interm_feat, 
                prefix='s3_'
            )
        x = self.s4(x)
        if self.FS_FUSION[3] or self.AFS_FUSION[3]:
            x, interm_feat = self.s4_fuse(
                x, 
                get_misaligned_audio=self.GET_MISALIGNED_AUDIO, 
                mode=fusion_pattern[3],
            )
            features = misc.update_dict_with_prefix(
                features, 
                interm_feat, 
                prefix='s4_'
            )
        x = self.s5(x)
        if self.AVS_FLAG[4]:
            _, interm_feat = self.s5_fuse(
                x, 
                get_misaligned_audio=self.GET_MISALIGNED_AUDIO, 
                mode='FS',
            )
            features = misc.update_dict_with_prefix(
                features, 
                interm_feat, 
                prefix='s5_'
            )
        
        # drop the negative samples in audio
        if self.GET_MISALIGNED_AUDIO:
            x = self.get_pos_audio(x)
        
        x = self.head(x)
        
        if self.training and self.GET_MISALIGNED_AUDIO:
            # compute loss if in training
            loss = self.avs_forward(features, audio_mask)
            return x, loss
        else:
            return x


class FuseFastToSlow(nn.Module):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
    """

    def __init__(
        self,
        dim_in,
        fusion_conv_channel_ratio,
        fusion_kernel,
        alpha,
        eps=1e-5,
        bn_mmt=0.1,
        inplace_relu=True,
        norm_module=nn.BatchNorm3d,
    ):
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(FuseFastToSlow, self).__init__()
        self.conv_f2s = nn.Conv3d(
            dim_in,
            dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            bias=False,
        )
        self.bn = norm_module(
            num_features=dim_in * fusion_conv_channel_ratio,
            eps=eps,
            momentum=bn_mmt,
        )
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]


@MODEL_REGISTRY.register()
class SlowFast(nn.Module):
    """
    SlowFast model builder for SlowFast network.

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFast, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 2
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group
        out_dim_ratio = (
            cfg.SLOWFAST.BETA_INV // cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO
        )

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group, width_per_group // cfg.SLOWFAST.BETA_INV],
            kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]],
            stride=[[1, 2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
            ],
            norm_module=self.norm_module,
        )
        self.s1_fuse = FuseFastToSlow(
            width_per_group // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[
                width_per_group + width_per_group // out_dim_ratio,
                width_per_group // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 4,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner, dim_inner // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )
        self.s2_fuse = FuseFastToSlow(
            width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 4 + width_per_group * 4 // out_dim_ratio,
                width_per_group * 4 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 8,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 2, dim_inner * 2 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )
        self.s3_fuse = FuseFastToSlow(
            width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 8 + width_per_group * 8 // out_dim_ratio,
                width_per_group * 8 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 16,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 4, dim_inner * 4 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )
        self.s4_fuse = FuseFastToSlow(
            width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 16 + width_per_group * 16 // out_dim_ratio,
                width_per_group * 16 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_out=[
                width_per_group * 32,
                width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
            ],
            dim_inner=[dim_inner * 8, dim_inner * 8 // cfg.SLOWFAST.BETA_INV],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if cfg.DETECTION.ENABLE:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[
                    width_per_group * 32,
                    width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                ],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[
                    [
                        cfg.DATA.NUM_FRAMES
                        // cfg.SLOWFAST.ALPHA
                        // pool_size[0][0],
                        1,
                        1,
                    ],
                    [cfg.DATA.NUM_FRAMES // pool_size[1][0], 1, 1],
                ],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2] * 2,
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR] * 2,
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[
                    width_per_group * 32,
                    width_per_group * 32 // cfg.SLOWFAST.BETA_INV,
                ],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                else [
                    [
                        cfg.DATA.NUM_FRAMES
                        // cfg.SLOWFAST.ALPHA
                        // pool_size[0][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                    ],
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[1][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[1][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[1][2],
                    ],
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )

    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)
        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x


@MODEL_REGISTRY.register()
class ResNet(nn.Module):
    """
    ResNet model builder. It builds a ResNet like network backbone without
    lateral connection (C2D, I3D, Slow).

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf

    Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    https://arxiv.org/pdf/1711.07971.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(ResNet, self).__init__()
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        """
        Builds a single pathway ResNet model.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1[cfg.MODEL.ARCH]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[cfg.RESNET.DEPTH]

        num_groups = cfg.RESNET.NUM_GROUPS
        width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]

        self.s1 = stem_helper.VideoModelStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[width_per_group],
            kernel=[temp_kernel[0][0] + [7, 7]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 3, 3]],
            norm_module=self.norm_module,
        )

        self.s2 = resnet_helper.ResStage(
            dim_in=[width_per_group],
            dim_out=[width_per_group * 4],
            dim_inner=[dim_inner],
            temp_kernel_sizes=temp_kernel[1],
            stride=cfg.RESNET.SPATIAL_STRIDES[0],
            num_blocks=[d2],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[0],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
            nonlocal_group=cfg.NONLOCAL.GROUP[0],
            nonlocal_pool=cfg.NONLOCAL.POOL[0],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[0],
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[width_per_group * 4],
            dim_out=[width_per_group * 8],
            dim_inner=[dim_inner * 2],
            temp_kernel_sizes=temp_kernel[2],
            stride=cfg.RESNET.SPATIAL_STRIDES[1],
            num_blocks=[d3],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[1],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[1],
            nonlocal_group=cfg.NONLOCAL.GROUP[1],
            nonlocal_pool=cfg.NONLOCAL.POOL[1],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[1],
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[width_per_group * 8],
            dim_out=[width_per_group * 16],
            dim_inner=[dim_inner * 4],
            temp_kernel_sizes=temp_kernel[3],
            stride=cfg.RESNET.SPATIAL_STRIDES[2],
            num_blocks=[d4],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[2],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[2],
            nonlocal_group=cfg.NONLOCAL.GROUP[2],
            nonlocal_pool=cfg.NONLOCAL.POOL[2],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[2],
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[width_per_group * 16],
            dim_out=[width_per_group * 32],
            dim_inner=[dim_inner * 8],
            temp_kernel_sizes=temp_kernel[4],
            stride=cfg.RESNET.SPATIAL_STRIDES[3],
            num_blocks=[d5],
            num_groups=[num_groups],
            num_block_temp_kernel=cfg.RESNET.NUM_BLOCK_TEMP_KERNEL[3],
            nonlocal_inds=cfg.NONLOCAL.LOCATION[3],
            nonlocal_group=cfg.NONLOCAL.GROUP[3],
            nonlocal_pool=cfg.NONLOCAL.POOL[3],
            instantiation=cfg.NONLOCAL.INSTANTIATION,
            trans_func_name=cfg.RESNET.TRANS_FUNC,
            stride_1x1=cfg.RESNET.STRIDE_1X1,
            inplace_relu=cfg.RESNET.INPLACE_RELU,
            dilation=cfg.RESNET.SPATIAL_DILATIONS[3],
            norm_module=self.norm_module,
        )

        if self.enable_detection:
            self.head = head_helper.ResNetRoIHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[[cfg.DATA.NUM_FRAMES // pool_size[0][0], 1, 1]],
                resolution=[[cfg.DETECTION.ROI_XFORM_RESOLUTION] * 2],
                scale_factor=[cfg.DETECTION.SPATIAL_SCALE_FACTOR],
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
                aligned=cfg.DETECTION.ALIGNED,
            )
        else:
            self.head = head_helper.ResNetBasicHead(
                dim_in=[width_per_group * 32],
                num_classes=cfg.MODEL.NUM_CLASSES,
                pool_size=[None, None]
                if cfg.MULTIGRID.SHORT_CYCLE
                else [
                    [
                        cfg.DATA.NUM_FRAMES // pool_size[0][0],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                        cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                    ]
                ],  # None for AdaptiveAvgPool3d((1, 1, 1))
                dropout_rate=cfg.MODEL.DROPOUT_RATE,
                act_func=cfg.MODEL.HEAD_ACT,
            )

    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s2(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        if self.enable_detection:
            x = self.head(x, bboxes)
        else:
            x = self.head(x)
        return x
