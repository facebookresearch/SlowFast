import torch
import torch.nn as nn
from slowfast.models.build import MODEL_REGISTRY
from slowfast.models.batchnorm_helper import get_norm
from slowfast.models.video_model_builder import FuseFastToSlow
from slowfast.models.video_model_builder import _TEMPORAL_KERNEL_BASIS
from slowfast.models import head_helper

# Model Complexity  as [ [stageRepeats], [outputChannels]]
# See Table 5 in the ShuffleNet paper
_MODEL_COMPLEXITY = {
    "shufflenet_v2_x0_5": [[1, 4, 8, 4], [24, 48, 96, 192, 1024]],
    "shufflenet_v2_x1_0": [[1, 4, 8, 4], [24, 112, 240, 464, 1024]],
    "shufflenet_v2_x1_5": [[1, 4, 8, 4], [24, 176, 352, 704, 1024]],
    "shufflenet_v2_x2_0": [[1, 4, 8, 4], [24, 240, 488, 976, 2048]],
}

class ShuffleNetStem(nn.Module):
    """
    Construct a stem module for shuffleNet
    Args:
        dim_in Input channel dimension.
        dim_out  Output channel dimension.
        kernel: Kernel height, width
        stride Conv stride
        padding: Conv padding       
    """
    def __init__(self, dim_in, dim_out, kernel, stride):
        super().__init__()
        assert isinstance(kernel, list) and len(kernel) == 3, "Must specify a 3 element list for kernel dimension found {}".format(kernel)
        padding = [ x // 2 for x in kernel]
        self.s =  nn.Sequential(
            nn.Conv3d(dim_in, dim_out, kernel_size=kernel, stride=stride, 
                padding=padding, bias=False,),
            nn.BatchNorm3d(dim_out),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=[1, 3, 3], stride=[1, 2, 2], padding=[0, 1, 1]),
        )
    def forward(self, x):
        return self.s(x)

class SlowFastShuffleNetStem(nn.Module):
    """
    Slowfast ShuffleNet input stem for both pathways 
    """
    def __init__(self, dim_in, dim_out, kernel, stride, repeat):
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
                )
            self.add_module("pathway{}_stem".format(pathway), stem)

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
    def __init__(self, inp, out, temp_kernel, stride):
        """
        The `__init__` method of any subclass should also contain these arguments.
        Args:
            dim_in: channel dimensions of the input.
            dim_out: channel dimensions of the output.
            temp_kernel: size of the temporal kernel
            stride: Convolution stride
        """         
        super().__init__()

        assert (1 <= stride <= 3), 'Expect stride in range[1,2] Found {} '.format(stride)
        self.stride = stride

        branch_features = out // 2
        assert (self.stride != 1) or (inp == branch_features * 2), "stride {} inp {} != 2 * {} = {}".format(stride, inp, branch_features, branch_features*2)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=[temp_kernel, 3, 3], 
                    stride=[1, self.stride, self.stride,],
                    padding=[temp_kernel // 2, 1, 1]),
                nn.BatchNorm3d(inp),
                nn.Conv3d(inp, branch_features, kernel_size=[temp_kernel, 1, 1], stride=1, 
                    padding=[temp_kernel // 2, 0, 0],
                    bias=False),
                nn.BatchNorm3d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv3d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=[temp_kernel, 1, 1], stride=1, 
                        padding=[temp_kernel // 2, 0, 0], 
                        bias=False),
            nn.BatchNorm3d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, 
                kernel_size=[temp_kernel, 3, 3], 
                stride=[1, self.stride, self.stride],
                padding=[temp_kernel // 2, 1, 1]),
            nn.BatchNorm3d(branch_features),
            nn.Conv3d(branch_features, branch_features, kernel_size=[temp_kernel, 1, 1], stride=1, 
                padding=[temp_kernel // 2, 0, 0],
                    bias=False),
            nn.BatchNorm3d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv3d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    @staticmethod
    def channel_shuffle(x, groups):
        # type: (torch.Tensor, int) -> torch.Tensor
        batchsize, num_channels, num_frame,  height, width = x.data.size()
        channels_per_group = num_channels // groups

        # reshape
        x = x.view(batchsize, groups,
                channels_per_group, num_frame, height, width)

        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batchsize, -1, num_frame, height, width)

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
    def __init__(self, dim_in, dim_out, temp_kernel, repeat,):
        """
        The `__init__` method of any subclass should also contain these arguments.
        ShuffleNetStage buildsone stage 
        Args:
            dim_in: channel dimensions of the input.
            dim_out: channel dimensions of the output.
            temp_kernel: size of the temporal kernel
            repeat: Number of repeat blocks in the stage
        """ 
        super().__init__()
        seq = [InvertedResidual(dim_in, dim_out, temp_kernel, 2)]
        for _ in range(repeat-1):
            seq.append(InvertedResidual(dim_out, dim_out, temp_kernel, 1))
        self.stage = nn.Sequential(*seq)

    def forward(self, x):
        return self.stage(x)

class SlowFastShuffleNetStage(nn.Module):
    """
    Slowfast ShuffleNet create a single stage for both pathways
    """
    def __init__(self, dim_in, dim_out, temp_kernel, repeat):
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
            stage = ShuffleNetStage(dim_in[pathway], dim_out[pathway], temp_kernel[pathway][0], repeat,) 
            self.add_module(
                "pathway{}".format(pathway), stage
            )

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
    def __init__(self, dim_in, dim_out, temp_kernel):
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
                nn.Conv3d(dim_in[pathway], dim_out[pathway], kernel_size=[temp_kernel[pathway][0], 1, 1], stride=1, 
                    padding=[temp_kernel[pathway][0] // 2, 0, 0],
                    bias=False),
                nn.BatchNorm3d(dim_out[pathway]),
                nn.ReLU(inplace=True),
            )
            self.add_module("conv_5_{}".format(pathway), conv5)

    def forward(self, inputs):
        output = []
        for pathway in range(self.num_pathways):
            x = inputs[pathway]
            m = getattr(self, "conv_5_{}".format(pathway))
            x = m(x)
            output.append(x)

        return output
@MODEL_REGISTRY.register()
class SlowFastShuffleNetV2(nn.Module):
    """
    Slowfast model builder using a shufflenet V2
    backbone described in 
    References:
    1. Slowfast networks for video recogniion https://arxiv.org/pdf/1812.03982.pdf
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
        print("Created SlowFastShuffleNetV2 version {}".format(cfg.SHUFFLENET.COMPLEXITY))

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
        assert cfg.SHUFFLENET.COMPLEXITY in _MODEL_COMPLEXITY.keys(), "{} unrecognized ShffleNetComplexity must be {}".format(
            cfg.SHUFFLENET.COMPLEXITY, list(_MODEL_COMPLEXITY.keys())
        )
        stage_repeats, stage_outs = _MODEL_COMPLEXITY[cfg.SHUFFLENET.COMPLEXITY]

        # out_dim_ratio = (
        #     cfg.SLOWFAST.BETA_INV // cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO
        # )
        temp_kernel = _TEMPORAL_KERNEL_BASIS[cfg.MODEL.ARCH]
        out_dim_ratio = (
            cfg.SLOWFAST.BETA_INV // cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO
        )
        
        stageIdx = 0
        self.s0 = SlowFastShuffleNetStem(
            dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_out=[stage_outs[stageIdx], stage_outs[stageIdx] // cfg.SLOWFAST.BETA_INV],
            kernel=[temp_kernel[0][0] + [3, 3], temp_kernel[0][1] + [3, 3]],
            stride=[[1, 2, 2]] * 2,
            repeat=stage_repeats[stageIdx],
        )
        self.s0_fuse = FuseFastToSlow(
            stage_outs[stageIdx] // cfg.SLOWFAST.BETA_INV,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        for stageIdx in range(1, len(stage_repeats)):
            stage = SlowFastShuffleNetStage(
                dim_in=[
                    stage_outs[stageIdx-1] + stage_outs[stageIdx-1] // out_dim_ratio,
                    stage_outs[stageIdx-1] // cfg.SLOWFAST.BETA_INV],
                dim_out=[stage_outs[stageIdx] , stage_outs[stageIdx] // cfg.SLOWFAST.BETA_INV],
                temp_kernel=temp_kernel[stageIdx],
                repeat=stage_repeats[stageIdx]
            )
            fuse = FuseFastToSlow(
                stage_outs[stageIdx] // cfg.SLOWFAST.BETA_INV,
                cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
                cfg.SLOWFAST.FUSION_KERNEL_SZ,
                cfg.SLOWFAST.ALPHA,
                norm_module=self.norm_module,
            )
            self.add_module("s{}".format(stageIdx),stage)
            self.add_module("s{}_fuse".format(stageIdx), fuse)

        stageIdx = len(stage_outs) - 1
        self.conv5 = SlowFastShuffleConv5(
            dim_in=[
                stage_outs[stageIdx-1] + stage_outs[stageIdx-1] // out_dim_ratio, 
                stage_outs[stageIdx-1] // cfg.SLOWFAST.BETA_INV],
            dim_out=[stage_outs[stageIdx] , stage_outs[stageIdx] // cfg.SLOWFAST.BETA_INV], 
            temp_kernel=temp_kernel[stageIdx]          
        )

        self.head = head_helper.ResNetBasicHead(
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

    def forward(self, x, bboxes=None):
        x = self.s0(x)
        x = self.s0_fuse(x)
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        # for pathway in range(self.num_pathways):
        #     pool = getattr(self, "pathway{}_pool".format(pathway))
        #     x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)

        x = self.conv5(x)

        x = self.head(x)
        return x

