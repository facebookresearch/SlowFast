import torch
import torch.nn as nn
from collections import OrderedDict


from slowfast.models.build import MODEL_REGISTRY

# from slowfast.models.batchnorm_helper import get_norm
# from slowfast.models.video_model_builder import FuseFastToSlow
# from slowfast.models import head_helper

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


def get_norm(cfg):
    """
    Args:
        cfg (CfgNode): model building configs, details are in the comments of
            the config file.
    Returns:
        nn.Module: the normalization layer.
    """
    return nn.BatchNorm2d


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
        norm_module=nn.BatchNorm2d,
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
                default is nn.BatchNorm2d.
        """
        super(FuseFastToSlow, self).__init__()
        self.conv_f2s = nn.Conv2d(
            dim_in,
            int(dim_in * fusion_conv_channel_ratio),
            kernel_size=[1, 1],
            stride=[1, 1],
            padding=[0, 0],
            bias=False,
        )
        self.bn = norm_module(
            num_features=int(dim_in * fusion_conv_channel_ratio),
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


class ResNetBasicHead(nn.Module):
    """
    ResNe(X)t 2D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ResNetBasicHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)

        for pathway in range(self.num_pathways):
            if pool_size[pathway] is None:
                avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            else:
                avg_pool = nn.AvgPool2d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=3)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        # (N, C, H, W) -> (N, H, W, C).
        x = x.permute((0, 2, 3, 1))
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        # Performs fully convlutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2,])

        x = x.view(x.shape[0], -1)
        return x


class ShuffleNetStem(nn.Module):
    """
    Construct a stem module for shuffleNet
    Args:
        dim_in Input channel dimension.
        dim_out  Output channel dimension.
        kernel: Kernel height, width
        stride Conv stride
        groups - Conv Group size      
    """
    def __init__(self, dim_in, dim_out, kernel, stride, groups):
        super().__init__()
        assert isinstance(kernel, list) and len(kernel) == 2, "Must specify a 2 element list for kernel dimension found {}".format(kernel)
        padding = [ x // 2 for x in kernel]
        self.s =  nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel, stride=stride, 
                padding=padding, bias=False,groups=groups),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=[3, 3], stride=[2, 2], padding=[1, 1]),
        )
    def forward(self, x):
        return self.s(x)

class SlowFastShuffleNetStem(nn.Module):
    """
    Slowfast ShuffleNet input stem for both pathways 
    """
    def __init__(self, dim_in, dim_out, kernel, stride, repeat, groups):
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
            groups (list) group size for each pathway
        """
        super().__init__()
        assert len({
            len(dim_in),
            len(dim_out),
            len(kernel),
            len(stride),
            len(groups),
        }) == 1, "Inconsistent inputs for shuffleNet pathways dIn {} dOut {} kern {} str {} gr {}".format(
            len(dim_in), len(dim_out), len(kernel), len(stride), len(groups),
        )
        assert repeat == 1, "Expect just a single block in Stem"
        self.num_pathways = len(dim_in)
        
        for pathway in range(len(dim_in)):
            stem = ShuffleNetStem(
                dim_in[pathway],
                dim_out[pathway],
                kernel[pathway],
                stride[pathway],
                groups[pathway],
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
    def __init__(self, inp, out, temp_kernel, stride, groups, cfg=None):
        """
        The `__init__` method of any subclass should also contain these arguments.
        Args:
            dim_in: channel dimensions of the input.
            dim_out: channel dimensions of the output.
            temp_kernel: size of the temporal kernel
            stride: Convolution stride
            groups - Conv groups
        """         
        super().__init__()

        assert (1 <= stride <= 3), 'Expect stride in range[1,3] Found {} '.format(stride)
        self.stride = stride

        branch_features = out // 2
        assert (self.stride != 1) or (inp == branch_features * 2), "stride {} inp {} != 2 * {} = {}".format(stride, inp, branch_features, branch_features*2)

        if self.stride > 1:
            br1 = self.makeDepthWiseOrFull(inp, branch_features, temp_kernel, groups, cfg)
            br1.extend( [nn.BatchNorm2d(branch_features), nn.ReLU(inplace=True),])
            self.branch1 = nn.Sequential(*br1)
        else:
            self.branch1 = nn.Sequential()

        br2 = [
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=[1, 1], stride=1, 
                        padding=[0, 0], groups = groups,
                        bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),            
        ]
        br2.extend(self.makeDepthWiseOrFull(branch_features, branch_features, temp_kernel, groups, cfg))
        br2.extend([nn.BatchNorm2d(branch_features), nn.ReLU(inplace=True),])
        self.branch2 = nn.Sequential(*br2)

    def makeDepthWiseOrFull(self, inp, branch_features, temp_kernel, groups, cfg=None):
        assert not cfg.depthwise, "Depthwise convolution not supported for R2Plus1 architectures"

        ret = [
            nn.Conv2d(inp, branch_features, kernel_size=[3, 3],
                stride=[self.stride, self.stride,], 
                padding=[1, 1],
                groups=groups,
                bias=False)
            ]

        return ret


    @staticmethod
    def channel_shuffle(x, groups):
        # type: (torch.Tensor, int) -> torch.Tensor
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups

        # reshape
        x = x.view(-1, groups,
                channels_per_group, height, width)

        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(-1, channels_per_group*groups, height, width)

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
    def __init__(self, dim_in, dim_out, temp_kernel, repeat, groups, cfg=None):
        """
        The `__init__` method of any subclass should also contain these arguments.
        ShuffleNetStage buildsone stage 
        Args:
            dim_in: channel dimensions of the input.
            dim_out: channel dimensions of the output.
            temp_kernel: size of the temporal kernel
            repeat: Number of repeat blocks in the stage
            groups - conv grouping
        """ 
        super().__init__()
        seq = [InvertedResidual(dim_in, dim_out, temp_kernel, 2, groups, cfg=cfg)]
        for _ in range(repeat-1):
            seq.append(InvertedResidual(dim_out, dim_out, temp_kernel, 1, groups, cfg=cfg))
        self.stage = nn.Sequential(*seq)

    def forward(self, x):
        return self.stage(x)

class SlowFastShuffleNetStage(nn.Module):
    """
    Slowfast ShuffleNet create a single stage for both pathways
    """
    def __init__(self, dim_in, dim_out, temp_kernel, repeat, groups, cfg=None):
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
            groups - Conv group sizes
        """
        super().__init__()
        assert  len( {
            len(dim_in),
            len(dim_out),
            len(temp_kernel),
            len(groups)
         }) == 1,  "Expect number of pathways inp {} == out {} == temp kernel {} == grps {}".format(
             len(dim_in), len(dim_out), len(temp_kernel), len(groups))
    
        self.num_pathways = len(dim_in)

        for pathway in range(self.num_pathways):
            stage = ShuffleNetStage(dim_in[pathway], dim_out[pathway], temp_kernel[pathway][0], repeat, groups[pathway], cfg=cfg) 
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
    def __init__(self, dim_in, dim_out, temp_kernel, groups):
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
            groups - (list) gConv group sizes
        """
        super().__init__()
        assert  len( {
            len(dim_in),
            len(dim_out),
            len(temp_kernel),
            len(groups),
         }) == 1,  "Expect number of pathways inp {} == out {} == temp kernael {} grps {}".format(
             len(dim_in), len(dim_out), len(temp_kernel), len(groups))
    
        self.num_pathways = len(dim_in)

        for pathway in range(self.num_pathways):
            conv5 = nn.Sequential(
                nn.Conv2d(dim_in[pathway], dim_out[pathway], kernel_size=[1, 1], stride=1, 
                    padding=[0, 0], groups=groups[pathway],
                    bias=False),
                nn.BatchNorm2d(dim_out[pathway]),
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
class SlowFastShuffleNetV2D2(nn.Module):
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
        print("Created SlowFastShuffleNetV2D2 version {}".format(cfg.SHUFFLENET.COMPLEXITY))

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
        grps = [1, cfg.DATA.NUM_FRAMES]

        stageIdx = 0
        layers = nn.ModuleDict()

        layers['s0'] = SlowFastShuffleNetStem(
            dim_in=[cfg.DATA.INPUT_CHANNEL_NUM[0], cfg.DATA.INPUT_CHANNEL_NUM[1] * cfg.DATA.NUM_FRAMES],
            dim_out=[stage_outs[stageIdx], stage_outs[stageIdx] // cfg.SLOWFAST.BETA_INV * cfg.DATA.NUM_FRAMES],
            kernel=[[3, 3], [3, 3]],
            stride=[[2, 2]] * 2,
            repeat=stage_repeats[stageIdx],
            groups=grps,
        )
        layers['s0_fuse'] = FuseFastToSlow(
            stage_outs[stageIdx] // cfg.SLOWFAST.BETA_INV * cfg.DATA.NUM_FRAMES,
            cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO / cfg.DATA.NUM_FRAMES,
            cfg.SLOWFAST.FUSION_KERNEL_SZ,
            cfg.SLOWFAST.ALPHA,
            norm_module=self.norm_module,
        )

        for stageIdx in range(1, len(stage_repeats)):
            stage = SlowFastShuffleNetStage(
                dim_in=[
                    stage_outs[stageIdx-1] + stage_outs[stageIdx-1] // out_dim_ratio,
                    stage_outs[stageIdx-1] // cfg.SLOWFAST.BETA_INV * cfg.DATA.NUM_FRAMES,],
                dim_out=[stage_outs[stageIdx] , stage_outs[stageIdx] // cfg.SLOWFAST.BETA_INV * cfg.DATA.NUM_FRAMES],
                temp_kernel=temp_kernel[stageIdx],
                repeat=stage_repeats[stageIdx],
                groups=grps,
                cfg=cfg
            )
            fuse = FuseFastToSlow(
                stage_outs[stageIdx] // cfg.SLOWFAST.BETA_INV * cfg.DATA.NUM_FRAMES,
                cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO / cfg.DATA.NUM_FRAMES,
                cfg.SLOWFAST.FUSION_KERNEL_SZ,
                cfg.SLOWFAST.ALPHA,
                norm_module=self.norm_module,
            )
            layers["s{}".format(stageIdx)] = stage
            layers["s{}_fuse".format(stageIdx)] = fuse

        stageIdx = len(stage_outs) - 1
        layers['conv5'] = SlowFastShuffleConv5(
            dim_in=[
                stage_outs[stageIdx-1] + stage_outs[stageIdx-1] // out_dim_ratio, 
                stage_outs[stageIdx-1] // cfg.SLOWFAST.BETA_INV * cfg.DATA.NUM_FRAMES],
            dim_out=[stage_outs[stageIdx] , stage_outs[stageIdx] // cfg.SLOWFAST.BETA_INV * cfg.DATA.NUM_FRAMES], 
            temp_kernel=temp_kernel[stageIdx],
            groups=grps,        
        )

        layers['head'] = ResNetBasicHead(
            dim_in=[stage_outs[stageIdx] , stage_outs[stageIdx] // cfg.SLOWFAST.BETA_INV * cfg.DATA.NUM_FRAMES],
            num_classes=cfg.MODEL.NUM_CLASSES,
            pool_size=[None, None]
            if cfg.MULTIGRID.SHORT_CYCLE
            else [
                [
                    cfg.DATA.CROP_SIZE // 32 // pool_size[0][1],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[0][2],
                ],
                [
                    cfg.DATA.CROP_SIZE // 32 // pool_size[1][1],
                    cfg.DATA.CROP_SIZE // 32 // pool_size[1][2],
                ],
            ],  
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
        )
        self.layers = layers

    def forward(self, x, bboxes=None):
        # Tensor shapes BxCxTxHxW
        # Reshape the fast to be BxCTxHxW
        x[0] = torch.squeeze(x[0], dim=2)

        dims = x[1].shape
        x[1] = x[1].permute((0, 2, 1, 3, 4))
        x[1] = x[1].reshape((dims[0], dims[1]*dims[2], dims[3], dims[4]))
        for k, l in self.layers.items():
            # print("Before layer: {}  x {} {}".format(k,  x[0].shape, x[1].shape))
            # self.timer.tic(k)
            x = l(x)
        #     self.timer.toc(k)
        #     if len(x) > 1:
        #         print("After layer: {} Time {} x {} {}".format(k, self.timer.elapsed(k), x[0].shape, x[1].shape))
        #     else:
        #         print("layer: {} Time {} x {} ".format(k, self.timer.elapsed(k), x[0].shape, ))
        # print("Total time {}".format(self.timer.totTimes()))
        return x
