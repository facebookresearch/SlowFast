import torch
import torch.nn as nn
from collections import OrderedDict
from slowfast.models.build import MODEL_REGISTRY
from slowfast.models.batchnorm_helper import get_norm
from slowfast.models.video_model_builder import FuseFastToSlow
from slowfast.models.video_model_builder import _TEMPORAL_KERNEL_BASIS
from slowfast.models import head_helper

def _make_divisible(v, divisor, min_value=None):
  """
  This function is taken from the original tf repo.
  It ensures that all layers have a channel number that is divisible by 8
  It can be seen here:
  https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
  :param v:
  :param divisor:
  :param min_value:
  :return:
  """
  if min_value is None:
      min_value = divisor
  new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * v:
      new_v += divisor
  return new_v


class ConvBNReLU(nn.Sequential):
  def __init__(self, in_planes, out_planes, kernel, stride=1, groups=1):
    if isinstance(kernel, list):
      padding = [ (k - 1) // 2 for k in kernel ]
    else:
      padding = (kernel - 1) // 2
  
    super().__init__(
      nn.Conv3d(in_planes, out_planes, kernel, stride, padding, groups=groups, bias=False),
      nn.BatchNorm3d(out_planes),
      nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
  '''
  MobileNets' BottleNeck - See table 1 in the paper
  '''
  def __init__(self, inp, oup, stride, expand_ratio, kernel, norm_layer=None):
    '''
    inp - Number of input channels
    oup - Numbr of output channels
    expand_ratio - factor to calculate hidden units
    kernel - Kernel size for depth wise conv
    norm_layer - final layer normalization - default Batchnorm
    '''
    super().__init__()
    spacialStride = stride[-1] if isinstance(stride, list) else stride
    assert spacialStride in [1, 2], "spacual stride must < 1 found {}".format(spacialStride)

    if norm_layer is None:
      norm_layer = nn.BatchNorm3d

    hidden_dim = int(round(inp * expand_ratio))
    self.use_res_connect = spacialStride == 1 and inp == oup

    layers = []
    if expand_ratio != 1:
        # pointwise
      layers.append(ConvBNReLU(inp, hidden_dim, kernel=1, ))
    layers.extend([
        # depth wise conv
      ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, kernel=kernel,),
      # point-wise linear
      nn.Conv3d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
      norm_layer(oup),
    ])
    self.conv = nn.Sequential(*layers)

  def forward(self, x):
    if self.use_res_connect:
      return x + self.conv(x)
    else:
      return self.conv(x)

class SlowFastMobileV2Stage(nn.Module):
  """
  Slowfast MobileNet V2 create a single stage for both pathways
  """
  def __init__(self, cfg, dim_in, dim_out, stride, expand_ratio, temp_kernels):
    """
    The `__init__` method of any subclass should also contain these arguments.
    SlowFastShuffleNetStage builds p streams, where p can be greater or equal to one.
    Args:
        cfg - Configuration for the run
        dim_in (list): list of p the channel dimensions of the input.
            Different channel dimensions control the input dimension of
            different pathways.
        dim_out (list): list of p the channel dimensions of the output.
            Different channel dimensions control the input dimension of
            different pathways.
        stride (int) Stride of the first layer in stage
        expand_ratio - Expansion ratio for hidden layer in inverted residual
        temp_kernel - list of p channel for the temporal kernel size
      """
    super().__init__()
    assert  len( {
        len(dim_in),
        len(dim_out),
        len(temp_kernels)
      }) == 1,  "Expect number of pathways inp {} == out {} == kernels {}".format(
          len(dim_in), len(dim_out), len(temp_kernels))

    self.num_pathways = len(dim_in)

    for pathway in range(self.num_pathways):
      stage = InvertedResidual(dim_in[pathway], dim_out[pathway], stride, 
        expand_ratio, [temp_kernels[pathway]] + cfg.MOBILENETV2.SPATIAL_KERNELS) 
      self.add_module(
          "pathway_{}".format(pathway), stage
      )

  def forward(self, inputs):
    output = []
    for pathway in range(self.num_pathways):
      x = inputs[pathway]
      m = getattr(self, "pathway_{}".format(pathway))
      x = m(x)
      output.append(x)

    return output

class SlowFastMobileNetV2NetStem(nn.Module):
  """
  Slowfast ShuffleNet input stem for both pathways 
  """
  def __init__(self, cfg, dim_in, dim_out, temp_kernels, stride, repeat, addFuse=0):
    """
    List sizes should be 2
    Args:
        dim_in (list): the list of channel dimensions of the inputs.
        dim_out (list): the output dimension of the convolution in the stem
            layer.
        temp_kernel (list): the Temporal kernels' size of the convolutions in the stem
            layers. 
        stride (list): the stride sizes of the convolutions in the stem
            layer. Temporal kernel stride, height kernel size, width kernel
            size in order.
        padding (list): Padding for the p stem pathways
        repeat: Number of repeats
        addFuse - If1 add a fast -> slow fusion layer       
    """
    super().__init__()
    assert len({
        len(dim_in),
        len(dim_out),
        len(temp_kernels),
        len(stride),
    }) == 1, "Inconsistent inputs for shuffleNet pathways"
    assert repeat == 1, "Expect just a single block in Stem"
    self.num_pathways = len(dim_in)
    
    for pathway in range(self.num_pathways):
      stem = ConvBNReLU(
          dim_in[pathway],
          dim_out[pathway],
          [temp_kernels[pathway]] + cfg.MOBILENETV2.SPATIAL_KERNELS,
          stride[pathway],
          )
      self.add_module("pathway_{}_stem".format(pathway), stem)

    self.out_chans = dim_out

    if addFuse > 0:
      fuse = FuseFastToSlow(
        dim_out[-1],      # Fast pathway
        cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
        cfg.SLOWFAST.FUSION_KERNEL_SZ,
        cfg.SLOWFAST.ALPHA,
        norm_module=get_norm(cfg),
      )
      self.add_module("fuse_stem", fuse)
      self.out_chans[0] += dim_out[-1] * cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO

  def forward(self, x):
    assert (
        len(x) == self.num_pathways
    ), "Input tensor does not contain {} pathway".format(self.num_pathways)
    for pathway in range(len(x)):
        m = getattr(self, "pathway_{}_stem".format(pathway))
        x[pathway] = m(x[pathway])

    m = getattr(self, "fuse_stem", None)
    if m is not None:
      x = m(x)

    return x

@MODEL_REGISTRY.register()
class MobileNetV2(nn.Module):
  def __init__(self, cfg):
    """
    Implements MobileNet V2 
    https://openaccess.thecvf.com/content_cvpr_2018/papers/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf

    Args:
      cfg - The run configuration parameter of interest
        MOBILENETV2.STAGE_SETTING - details the parameters used in each model stage
        MOBILENETV2.SPATIAL_KERNELS - Spatial size of the kernls
        DATA.INPUT_CHANNEL_NUM - Size of the input
        MOBILENETV2.WIDTH_MULT - Scales the width (num channels) of all layers
        MOBILENETV2.ROUND_NEAREST - rounding factor when scaling the model width Set to 1 to turn off rounding
        MODEL.NUM_CLASSES  Number of output classes
        SLOWFAST.BETA_INV - Ratio of  slow / Fast channels
        SLOWFAST.ALPHA - Ratio Fast / Slow temporal size 
        SLOWFAST.FUSION_CONV_CHANNEL_RATIO - Slow/fast fusion channels ratio
        SLOWFAST.FUSION_KERNEL_SZ - Kernel used in slow/fast fusion 
    """
    super().__init__()
    self.num_pathways = 2
    self.norm_module = get_norm(cfg)
    self.cfg = cfg
    self._construct_network(cfg)
    print("Created MobileNetV2")    

  def _construct_network(self, cfg):
    stage_settings = cfg.MOBILENETV2.STAGE_SETTING
    pool_size = [[1, 1, 1], [1, 1, 1]]
    assert len({len(pool_size), self.num_pathways}) == 1
    in_chans = cfg.DATA.INPUT_CHANNEL_NUM
    width_mult =  self.cfg.MOBILENETV2.WIDTH_MULT
    round_nearest =  self.cfg.MOBILENETV2.ROUND_NEAREST
    num_classes = self.cfg.MODEL.NUM_CLASSES

    stage_idx = 0
    layerDict = OrderedDict()
    # ExpansionFactor, NumOutputChans, NumRepeats, Stride 
    t, c, n, s, tks, tkf, fuse = stage_settings[stage_idx]
    out_chans = _make_divisible(c * width_mult, round_nearest)
    out_chans = [out_chans, out_chans // cfg.SLOWFAST.BETA_INV]
    stem = SlowFastMobileNetV2NetStem(
        cfg=cfg,
        dim_in=in_chans,
        dim_out=out_chans,
        temp_kernels=[tks, tkf],
        stride=[[1, s, s]] * 2,
        repeat=n,
        addFuse=fuse,
    )
    layerDict["s_0"] = stem
    in_chans = stem.out_chans

    for stage_idx in range(1, len(stage_settings)):
      t, c, n, s, tks, tkf, fuse = stage_settings[stage_idx]
      if t <= 0:
        break
      out_chans = _make_divisible(c * width_mult, round_nearest)
      out_chans = [out_chans, out_chans // cfg.SLOWFAST.BETA_INV]
      for i in range(n):
        stride = [1, s, s] if i == 0 else 1
        stage = SlowFastMobileV2Stage(
            cfg,
            dim_in=in_chans,
            dim_out=out_chans,
            stride=stride,
            expand_ratio=t,
            temp_kernels=[tks, tkf],
            )
        in_chans = out_chans
        layerDict["s_{}_{}".format(stage_idx, i)] = stage

      if fuse > 0:
        fuse = FuseFastToSlow(
                  out_chans[-1],    # Fast numChannels
                  cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
                  cfg.SLOWFAST.FUSION_KERNEL_SZ,
                  cfg.SLOWFAST.ALPHA,
                  norm_module=self.norm_module,
              )
        layerDict["s_{}_fuse".format(stage_idx)] = fuse
        in_chans[0] += out_chans[-1] * cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO 

    assert stage_idx == len(stage_settings) - 1, \
      "Stage settings config is unexpected Found stage_idx {} has t={} for table size {}".format(stage_idx, t, len(stage_settings))
    t, c, n, s, tks, tkf, fuse = stage_settings[stage_idx]
    out_chans = _make_divisible(c * width_mult, round_nearest)
    out_chans = [out_chans, out_chans // cfg.SLOWFAST.BETA_INV]
    stem = SlowFastMobileNetV2NetStem(
        cfg=cfg,
        dim_in=in_chans,
        dim_out=out_chans,
        temp_kernels=[tks, tkf],
        stride=[[1, s, s]] * 2,
        repeat=n,
        addFuse=fuse,
    )
    in_chans = stem.out_chans
    layerDict["s_{}".format(stage_idx)]  = stem

    layerDict['head'] = head_helper.ResNetBasicHead(
        dim_in=in_chans,
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

    self.layers = nn.Sequential(layerDict)
    # self.layers = layerDict

  def forward(self, x):
    assert (
        len(x) == self.num_pathways
    ), "Input tensor does not contain {} pathway".format(self.num_pathways)    
    y = self.layers(x)

    return y