import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from slowfast.models.conv2Plus1 import Conv2Plus1
from slowfast.models.build import MODEL_REGISTRY
from slowfast.models.batchnorm_helper import get_norm
# from slowfast.models.video_model_builder import FuseFastToSlow
from slowfast.models.video_model_builder import _TEMPORAL_KERNEL_BASIS
from slowfast.models import head_helper

# Model Complexity  as [ [stageRepeats], [outputChannels]]
# See Table 5 in the ShuffleNet paper


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
    "x3d": [
        [[5]],  # conv1 temporal kernels.
        [[3]],  # res2 temporal kernels.
        [[3]],  # res3 temporal kernels.
        [[3]],  # res4 temporal kernels.
        [[3]],  # res5 temporal kernels.
    ],
}



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
        THW,
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
            THW - expected input tensor shape
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(FuseFastToSlow, self).__init__()
        self.conv_f2s = Conv2Plus1(
            dim_in,
            dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1, 1],
            stride=[alpha, 1, 1],
            padding=[fusion_kernel // 2, 0, 0],
            bias=False,
            THW=THW,
        )        
        # self.conv_f2s = nn.Conv3d(
        #     dim_in,
        #     dim_in * fusion_conv_channel_ratio,
        #     kernel_size=[fusion_kernel, 1, 1],
        #     stride=[alpha, 1, 1],
        #     padding=[fusion_kernel // 2, 0, 0],
        #     bias=False,
        # )
        self.bn = norm_module(
            num_features=dim_in * fusion_conv_channel_ratio,
            eps=eps,
            momentum=bn_mmt,
        )
        self.relu = nn.ReLU(inplace_relu)

    @staticmethod
    def frameSizeDivider():
        # Factor by which frame size diminishes
        # no frame size reduction
        return [[1, 1, 1], [1, 1, 1]]


    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]


class ShuffleNetStem(nn.Module):
    """
    Construct a stem module for shuffleNet
    Args:
        dim_in Input channel dimension.
        dim_out  Output channel dimension.
        kernel: Kernel height, width
        stride Conv stride
        padding: Conv padding
        THW - expected input tensor shape     
    """
    def __init__(self, dim_in, dim_out, kernel, stride, THW):
        super().__init__()
        assert isinstance(kernel, list) and len(kernel) == 3, "Must specify a 3 element list for kernel dimension found {}".format(kernel)
        padding = [ x // 2 for x in kernel]
        self.s =  nn.Sequential(
            Conv2Plus1(dim_in, dim_out, kernel_size=kernel, stride=stride, 
                padding=padding, bias=False, THW=THW),
            # nn.Conv3d(dim_in, dim_out, kernel_size=kernel, stride=stride, 
            #     padding=padding, bias=False,)
            nn.BatchNorm3d(dim_out),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1]),
        )
    
    def frameSizeDivider(self):
        # Factor by which frame size diminishes
        # Conv stride is 2 and MaxPool has stride 2
        return [1, 4, 4]

    def forward(self, x):
        return self.s(x)

class SlowFastShuffleNetStem(nn.Module):
    """
    Slowfast ShuffleNet input stem for both pathways 
    """
    def __init__(self, dim_in, dim_out, kernel, stride, repeat, THW):
        """
        List sizes should be 2
        Args:
            dim_in (list): the list of channel dimensions of the inputs.
            dim_out (list): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernels' size of the convolutions in the stem
                layers. Temporal kernel size, height kernel size, width kernel
                size in order.
            stride (list): the stride sizes of the convolutions in the stem
                layer. Temporal kernel stride, height kernel size, width kernel
                size in order.
            padding (list): Padding for the p stem pathways
            repeat: Number of repeats
            THW (list) - expected input tensor shape       
        """
        super().__init__()
        assert len({
            len(dim_in),
            len(dim_out),
            len(kernel),
            len(stride),
        }) == 1, "Inconsistent inputs for shuffleNet pathways"
        assert repeat == 1, "Expect just a single block in Stem"
        self.num_pathways = len(dim_in)
        
        for pathway in range(len(dim_in)):
            stem = ShuffleNetStem(
                dim_in[pathway],
                dim_out[pathway],
                kernel[pathway],
                stride[pathway],
                THW[pathway],
                )
            self.add_module("pathway{}_stem".format(pathway), stem)

    def frameSizeDivider(self):
        # Factor by which frame size diminishes
        div = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_stem".format(pathway))
            div.append(m.frameSizeDivider())
        return div

    def forward(self, x):
        assert (
            len(x) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        for pathway in range(len(x)):
            m = getattr(self, "pathway{}_stem".format(pathway))
            x[pathway] = m(x[pathway])
        return x

class InvertedResidual(nn.Module):
    """
    Basic building block in ShuffleNet
    See Fig 3 in shuffleNet paper
    """
    def __init__(self, inp, out, temp_kernel, stride, THW, cfg=None):
        """
        The `__init__` method of any subclass should also contain these arguments.
        Args:
            dim_in: channel dimensions of the input.
            dim_out: channel dimensions of the output.
            temp_kernel: size of the temporal kernel
            stride: Convolution stride
            THW - expected input tensor shape
            cfg - configuration options
        """         
        super().__init__()

        assert (1 <= stride <= 3), 'Expect stride in range[1,2] Found {} '.format(stride)
        self.stride = stride

        branch_features = out // 2
        assert (self.stride != 1) or (inp == branch_features * 2), "stride {} inp {} != 2 * {} = {}".format(stride, inp, branch_features, branch_features*2)

        if self.stride > 1:
            br1 = self.makeDepthWiseOrFull(inp, branch_features, temp_kernel, THW, cfg)
            br1.extend( [nn.BatchNorm3d(branch_features), nn.ReLU(inplace=True),])
            self.branch1 = nn.Sequential(*br1)
        else:
            self.branch1 = nn.Sequential()

        br2 = [
            Conv2Plus1(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=[temp_kernel, 1, 1], stride=1, 
                        padding=[temp_kernel // 2, 0, 0], THW=THW,
                        bias=False),            
            # nn.Conv3d(inp if (self.stride > 1) else branch_features,
            #           branch_features, kernel_size=[temp_kernel, 1, 1], stride=1, 
            #             padding=[temp_kernel // 2, 0, 0], 
            #             bias=False),
            nn.BatchNorm3d(branch_features),
            nn.ReLU(inplace=True),            
        ]
        br2.extend(self.makeDepthWiseOrFull(branch_features, branch_features, temp_kernel, THW, cfg))
        br2.extend([nn.BatchNorm3d(branch_features), nn.ReLU(inplace=True),])
        self.branch2 = nn.Sequential(*br2)

    def makeDepthWiseOrFull(self, inp, branch_features, temp_kernel, THW, cfg=None):
        if cfg.depthwise:
            assert cfg.depthwise == False, "Depthwise convolution not supported for R2Plus1 architectures"
            ret = [
            self.depthwise_conv(inp, inp, kernel_size=[temp_kernel, 3, 3], 
                stride=[1, self.stride, self.stride,],
                padding=[temp_kernel // 2, 1, 1], cfg=cfg),
            nn.BatchNorm3d(inp),
            nn.Conv3d(inp, branch_features, kernel_size=[temp_kernel, 1, 1], stride=1, 
                padding=[temp_kernel // 2, 0, 0],
                bias=False),
            ]
        else:
            ret = [
                Conv2Plus1(inp, branch_features, kernel_size=[temp_kernel, 3, 3],
                    stride=[1, self.stride, self.stride,], 
                    padding=[temp_kernel // 2, 1, 1],
                    bias=False,
                    THW=THW)                
                # nn.Conv3d(inp, branch_features, kernel_size=[temp_kernel, 3, 3],
                #     stride=[1, self.stride, self.stride,], 
                #     padding=[temp_kernel // 2, 1, 1],
                #     bias=False)
                ]

        return ret

    @staticmethod
    def updateFrameSize(THW, tHWDivider):
        assert len(THW) == len(tHWDivider), "Cannot update FrameSize Expect len thw {}  == len Divider {}".format(THW, tHWDivider)
        THWUpdate = [x // div for (x, div) in zip(THW, tHWDivider)]
        
        return THWUpdate

    def frameSizeDivider(self,):
        # Returns the frame side divider for this module
        return [1, self.stride, self.stride]

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False, THW=None, cfg=None):
        return nn.Conv3d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    @staticmethod
    def channel_shuffle(x, groups):
        # type: (torch.Tensor, int) -> torch.Tensor
        batchsize, num_channels, num_frame,  height, width = x.data.size()
        channels_per_group = num_channels // groups

        # reshape
        x = x.view(-1, groups,
                channels_per_group, num_frame, height, width)

        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(-1, channels_per_group*groups, num_frame, height, width)

        return x

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            b1 = self.branch1(x)
            b2 = self.branch2(x)
            out = torch.cat((b1, b2), dim=1)

        out = self.channel_shuffle(out, 2)

        return out

class ShuffleNetStage(nn.Module):
    """
    Create a single ShuffleNet stage
    See Table 3 in shuffleNet paper
    """
    def __init__(self, dim_in, dim_out, temp_kernel, repeat, THW, cfg=None):
        """
        The `__init__` method of any subclass should also contain these arguments.
        ShuffleNetStage buildsone stage 
        Args:
            dim_in: channel dimensions of the input.
            dim_out: channel dimensions of the output.
            temp_kernel: size of the temporal kernel
            repeat: Number of repeat blocks in the stage
            THW - expected input tensor shape
            cfg - configuration options
        """ 
        super().__init__()
        seq = [InvertedResidual(dim_in, dim_out, temp_kernel, 2, THW=THW, cfg=cfg)]
        for _ in range(repeat-1):
            THW = InvertedResidual.updateFrameSize(THW, seq[-1].frameSizeDivider())
            seq.append(InvertedResidual(dim_out, dim_out, temp_kernel, 1, THW=THW, cfg=cfg))
        self.stage = nn.Sequential(*seq)

    def frameSizeDivider(self):
        # Factor by which frame size diminishes
        div = self.stage[0].frameSizeDivider()
        for ss in self.stage[1:]:
            div = [x * y for (x,y) in zip(ss.frameSizeDivider(), div)]
        return div

    def forward(self, x):
        return self.stage(x)

class SlowFastShuffleNetStage(nn.Module):
    """
    Slowfast ShuffleNet create a single stage for both pathways
    """
    def __init__(self, dim_in, dim_out, temp_kernel, repeat, THW, cfg=None):
        """
        The `__init__` method of any subclass should also contain these arguments.
        SlowFastShuffleNetStage builds p streams, where p can be greater or equal to one.
        Args:
            dim_in (list): list of p the channel dimensions of the input.
                Different channel dimensions control the input dimension of
                different pathways.
            dim_out (list): list of p the channel dimensions of the output.
                Different channel dimensions control the input dimension of
                different pathways.
            temp_kernel - list of p channel for the tem,poral kernel size
            repeat:  Number of repeats
            THW - expected input tensor shape
            cfg Configuration options
        """
        super().__init__()
        assert  len( {
            len(dim_in),
            len(dim_out),
            len(temp_kernel)
         }) == 1,  "Expect number of pathways inp {} == out {} == temp kernael {}".format(
             len(dim_in), len(dim_out), len(temp_kernel))
    
        self.num_pathways = len(dim_in)

        for pathway in range(self.num_pathways):
            stage = ShuffleNetStage(dim_in[pathway], dim_out[pathway], temp_kernel[pathway][0], repeat, THW[pathway], cfg=cfg) 
            self.add_module(
                "pathway{}".format(pathway), stage
            )

    def frameSizeDivider(self):
        # Factor by which frame size diminishes
        div = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}".format(pathway))
            div.append(m.frameSizeDivider())
        return div

    def forward(self, inputs):
        output = []
        for pathway in range(self.num_pathways):
            x = inputs[pathway]
            m = getattr(self, "pathway{}".format(pathway))
            x = m(x)
            output.append(x)

        return output

class SlowFastShuffleConv5(nn.Module):
    """
    Slowfast ShuffleNet create  final layer for all pathways
    """
    def __init__(self, dim_in, dim_out, temp_kernel, THW):
        """
        The `__init__` method of any subclass should also contain these arguments.
        SlowFastShuffleNetStage builds p streams, where p can be greater or equal to one.
        Args:
            dim_in (list): list of p the channel dimensions of the input.
                Different channel dimensions control the input dimension of
                different pathways.
            dim_out (list): list of p the channel dimensions of the output.
                Different channel dimensions control the input dimension of
                different pathways.
            temp_kernel - list of p channel for the tem,poral kernel size
            THW - expected input tensor shape
        """
        super().__init__()
        assert  len( {
            len(dim_in),
            len(dim_out),
            len(temp_kernel)
         }) == 1,  "Expect number of pathways inp {} == out {} == temp kernael {}".format(
             len(dim_in), len(dim_out), len(temp_kernel))
    
        self.num_pathways = len(dim_in)

        for pathway in range(self.num_pathways):
            conv5 = nn.Sequential(
                Conv2Plus1(dim_in[pathway], dim_out[pathway], kernel_size=[temp_kernel[pathway][0], 1, 1], stride=1, 
                    padding=[temp_kernel[pathway][0] // 2, 0, 0],
                    bias=False,
                    THW=THW[pathway]),
                # nn.Conv3d(dim_in[pathway], dim_out[pathway], kernel_size=[temp_kernel[pathway][0], 1, 1], stride=1, 
                #     padding=[temp_kernel[pathway][0] // 2, 0, 0],
                #     bias=False),
                nn.BatchNorm3d(dim_out[pathway]),
                nn.ReLU(inplace=True),
            )
            self.add_module("conv_5_{}".format(pathway), conv5)

    @staticmethod
    def frameSizeDivider():
        # Factor by which frame size diminishes
        return [[1, 1, 1], [1, 1, 1]]            

    def forward(self, inputs):
        output = []
        for pathway in range(self.num_pathways):
            x = inputs[pathway]
            m = getattr(self, "conv_5_{}".format(pathway))
            x = m(x)
            output.append(x)

        return output
@MODEL_REGISTRY.register()
class SlowFastShuffleNetV2R2Plus1(nn.Module):
    """
    Slowfast model builder using a shufflenet V2
    backbone described in 
    References:
    1. Slowfast networks for video recognition https://arxiv.org/pdf/1812.03982.pdf
    2. ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design
    https://arxiv.org/pdf/1807.11164.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super().__init__()
        self.num_pathways = 2
        self.norm_module = get_norm(cfg)
        self._construct_network(cfg)
        print("Created SlowFastShuffleNetV2R2Plus1 version {}".format(cfg.SHUFFLENET.COMPLEXITY))

    @staticmethod
    def updateFrameSize(THW, tHWDivider):
        assert len(THW) == len(tHWDivider), "Expect len(THW) == len(tHWDivider) pathways. Found {}  {}".format(len(THW), len(tHWDivider))

        THWupdate = []
        for thw, div in zip(THW, tHWDivider):
            assert len(thw) == len(div), "Cannot update FrameSize Expect len thw {}  == len div {}".format(THW, div)
            THWupdate.append([x // dd for (x, dd) in zip(thw, div)])
        
        return THWupdate

    def _construct_network(self, cfg):
        """
        Builds the 2 pathway network The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        pool_size = [[1, 1, 1], [1, 1, 1]]
        assert len({len(pool_size), self.num_pathways}) == 1
        assert cfg.SHUFFLENET.COMPLEXITY in cfg.SHUFFLENET.ARCHITECTURE_TABLE.keys(), "{} unrecognized ShuffleNetComplexity must be {}".format(
            cfg.SHUFFLENET.COMPLEXITY, list(cfg.SHUFFLENET.ARCHITECTURE_TABLE.keys())
        )
        stage_repeats, stage_outs = cfg.SHUFFLENET.ARCHITECTURE_TABLE[cfg.SHUFFLENET.COMPLEXITY]
    
        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]
        out_dim_ratio = (
            cfg.SLOWFAST.BETA_INV // cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO
        )
        
        stageIdx = 0
        layers = nn.ModuleDict()
        # Input tensor (frame) size in TxHxxW for both pathways
        THW = [
            [1, cfg.DATA.FRAME_HEIGHT, cfg.DATA.FRAME_WIDTH],  # Slow pathway
            [cfg.DATA.NUM_FRAMES, cfg.DATA.FRAME_HEIGHT, cfg.DATA.FRAME_WIDTH]  # Fast Pathway
        ]

        layers['s0'] = SlowFastShuffleNetStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[stage_outs[stageIdx], stage_outs[stageIdx] // cfg.SLOWFAST.BETA_INV],
            kernel=[temp_kernel[0][0] + [3, 3], temp_kernel[0][1] + [3, 3]],
            stride=[[1, 2, 2]] * 2,
            repeat=stage_repeats[stageIdx],
            THW=THW
        )
        THW = SlowFastShuffleNetV2R2Plus1.updateFrameSize(THW, layers['s0'].frameSizeDivider())
        layers['s0_fuse'] = FuseFastToSlow(
            stage_outs[stageIdx] // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
            THW=THW[1],     # Conv is run on the fast pathway
        )
        THW = SlowFastShuffleNetV2R2Plus1.updateFrameSize(THW, FuseFastToSlow.frameSizeDivider())

        for stageIdx in range(1, len(stage_repeats)):
            stage = SlowFastShuffleNetStage(
                dim_in=[
                    stage_outs[stageIdx-1] + stage_outs[stageIdx-1] // out_dim_ratio,
                    stage_outs[stageIdx-1] // cfg.SLOWFAST.BETA_INV],
                dim_out=[stage_outs[stageIdx] , stage_outs[stageIdx] // cfg.SLOWFAST.BETA_INV],
                temp_kernel=temp_kernel[stageIdx],
                repeat=stage_repeats[stageIdx],
                cfg=cfg,
                THW=THW,
            )
            THW = SlowFastShuffleNetV2R2Plus1.updateFrameSize(THW, stage.frameSizeDivider())
            fuse = FuseFastToSlow(
                stage_outs[stageIdx] // cfg.SLOWFAST.BETA_INV,
                cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
                cfg.SLOWFAST.FUSION_KERNEL_SZ,
                cfg.SLOWFAST.ALPHA,
                norm_module=self.norm_module,
                THW=THW[1],     # Conv is run on the fast pathway
            )
            THW = SlowFastShuffleNetV2R2Plus1.updateFrameSize(THW, FuseFastToSlow.frameSizeDivider())
            layers["s{}".format(stageIdx)] = stage
            layers["s{}_fuse".format(stageIdx)] = fuse

        stageIdx = len(stage_outs) - 1
        layers['conv5'] = SlowFastShuffleConv5(
            dim_in=[
                stage_outs[stageIdx-1] + stage_outs[stageIdx-1] // out_dim_ratio, 
                stage_outs[stageIdx-1] // cfg.SLOWFAST.BETA_INV],
            dim_out=[stage_outs[stageIdx] , stage_outs[stageIdx] // cfg.SLOWFAST.BETA_INV], 
            temp_kernel=temp_kernel[stageIdx],
            THW =THW,         
        )
        THW = SlowFastShuffleNetV2R2Plus1.updateFrameSize(THW, SlowFastShuffleConv5.frameSizeDivider())

        layers['head'] = head_helper.ResNetBasicHead(
            dim_in=[stage_outs[stageIdx] , stage_outs[stageIdx] // cfg.SLOWFAST.BETA_INV],
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
        self.layers = layers

    def forward(self, x, bboxes=None):
        for k, l in self.layers.items():
            x = l(x)
        return x
