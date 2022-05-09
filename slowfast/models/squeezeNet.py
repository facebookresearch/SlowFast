import torch
import torch.nn as nn
from collections import OrderedDict
from slowfast.models.build import MODEL_REGISTRY
from slowfast.models.batchnorm_helper import get_norm
from slowfast.models.video_model_builder import FuseFastToSlow
from slowfast.models.video_model_builder import _TEMPORAL_KERNEL_BASIS
from slowfast.models import head_helper



class Fire(nn.Module):
  '''
  SqueezeNets Firre module
  '''
  def __init__(self, inplanes, s1x1, e1x1, e3x3, temp_kernel, addMaxPool, idx, iPath):
    '''
    inp - Number of input channels
    s1x1 channel dimensions of the squeezelayer.
    e1x1 (list) list of the p 1x1 plane dimension
    e3x3 (list) - list of the p 3x3 plane dimensions
    temp_kernel - list of p channel for the temporal kernel size
    res_connect - Use a residual short cut connection
    addMaxPool - add a maxPool layer
    '''
    super().__init__()
    self.idx = idx
    self.iPath = iPath
    layerDict = OrderedDict()
    self.use_res_connect = (inplanes == (e1x1 + e3x3))
    self.squeeze = nn.Sequential(
        nn.Conv3d(inplanes, s1x1, kernel_size=[temp_kernel, 1, 1]), 
        nn.ReLU(inplace=True)
    )

    self.expand1x1 = nn.Sequential(
        nn.Conv3d(s1x1, e1x1, kernel_size=[temp_kernel, 1, 1]),
        nn.ReLU(inplace=True)
    )

    padding = [(temp_kernel - 1) // 2, 1, 1]
    self.expand3x3 = nn.Sequential(
        nn.Conv3d(s1x1, e3x3, kernel_size=[temp_kernel, 3, 3], padding=padding),
        nn.ReLU(inplace=True)
    )
    self.addMaxPool = addMaxPool
    if addMaxPool:
       self.maxPool = nn.MaxPool3d(kernel_size=[1, 3, 3], stride=(1,2,2), padding=[0, 1, 1])    
    
    self.layers = nn.Sequential(layerDict)    

  def forward(self, x):
    y = self.squeeze(x)
    y = torch.cat([self.expand1x1 (y), self.expand3x3(y)], dim=1)

    if self.use_res_connect:
      y = x + y

    if self.addMaxPool:
      y = self.maxPool(y)
    
    return y

class SlowFastSqueezeFire(nn.Module):
  """
  Slowfast squeezeNet 'Fire' module
  """
  def __init__(self, dim_in, squeeze_planes, expand1x1_planes, expand3x3_planes, temp_kernels, addMaxPool, idx):
    """
    The `__init__` method of any subclass should also contain these arguments.
    SlowFastSqueezeNetStage builds p streams, where p can be greater or equal to one.
    Args:
        cfg - Configuration for the run
        dim_in (list): list of p the channel dimensions of the input.
            Different channel dimensions control the input dimension of
            different pathways.
        squeeze_planes (list): list of p the channel dimensions of the squeezelayer.
        expand1x1_planes (list) list of the p 1x1 plane dimension
        expand3x3_planes (list) - list of the p 3x3 plane dimensions
        temp_kernel - list of p channel for the temporal kernel size
        addMaxPool - Should maxPool be added
      """
    super().__init__()
    assert  len( {
        len(dim_in),
        len(squeeze_planes),
        len(expand3x3_planes),
      }) == 1,  "Expect number of pathways inp {} == squeeze {} == expand1x1 {} == expand3x3 {} == kernels {}".format(
          len(dim_in), len(squeeze_planes), len(expand1x1_planes), len(expand3x3_planes), len(temp_kernels))

    self.num_pathways = len(dim_in)
    self.idx = idx

    for i in range(self.num_pathways):
      module = Fire(dim_in[i], squeeze_planes[i], expand1x1_planes[i], expand3x3_planes[i], temp_kernels[i], addMaxPool, idx, i)
      self.add_module("pathway_{}".format(i), module)

  def forward(self, inputs):
    output = []
    for pathway in range(self.num_pathways):
      x = inputs[pathway]
      m = getattr(self, "pathway_{}".format(pathway))
      x = m(x)
      output.append(x)

    return output

class SlowFastSqueezeNetStem(nn.Module):
  """
  Slowfast SqueezeNet input stem for both pathways 
  """
  def __init__(self, dim_in, dim_out, conv_kernels, conv_stride, pool_kernel, pool_stride):
    """
    List sizes should be 2
    Args:
        dim_in (list): the list of channel dimensions of the inputs.
        dim_out (list): the output dimension of the convolution in the stem
            layer.
        conv_kernels (list): the Temporal kernels' size of the convolutions in the stem
            layers. 
        stride (list): the stride sizes of the convolutions in the stem
            layer. Temporal kernel stride, height kernel size, width kernel
            size in order.
        padding (list): Padding for the p stem pathways
        addFuse - If1 add a fast -> slow fusion layer       
    """
    super().__init__()
    assert len({
        len(dim_in),
        len(dim_out),
        len(conv_kernels),
        len(conv_stride),
        len(pool_kernel),
        len(pool_stride)
    }) == 1, "Inconsistent inputs for SqueezeNet pathways"
    self.num_pathways = len(dim_in)
    
    for pathway in range(self.num_pathways):
      layerDict = OrderedDict()
      if isinstance(conv_kernels[pathway], list):
        padding = [ (k - 1) // 2 for k in conv_kernels[pathway] ]
      else:
        padding = (conv_kernels[pathway] - 1) // 2      
      layerDict["conv_{}".format(pathway)] = nn.Conv3d(
        dim_in[pathway], dim_out[pathway], kernel_size=conv_kernels[pathway], padding=padding, stride=tuple(conv_stride[pathway]))
      layerDict["relu_{}".format(pathway)] = nn.ReLU(inplace=True)
      if isinstance(pool_kernel[pathway], list):
        padding = [ (k - 1) // 2 for k in pool_kernel[pathway] ]
      else:
        padding = (pool_kernel[pathway] - 1) // 2         
      layerDict['maxpool_{}'.format(pathway)] = nn.MaxPool3d(
        kernel_size=pool_kernel[pathway], stride=pool_stride[pathway], padding=padding)        

      self.add_module("pathway_{}_stem".format(pathway), nn.Sequential(layerDict))

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
class Squeezenet(nn.Module):
  def __init__(self, cfg):
    """
    Implements SqueezeNet
    https://arxiv.org/pdf/1602.07360.pdf

    Args:
        cfg - Run Configuration. 


    """
    super().__init__()
    self.num_pathways = 2
    self.norm_module = get_norm(cfg)
    self.cfg = cfg
    self._construct_network(cfg)
    print("Created Squeezenet ")    

  def _construct_network(self, cfg):
    module_settings = cfg.SQUEEZENET.MODULE_SETTINGS
    pool_size = [[1, 1, 1], [1, 1, 1]]
    assert len({len(pool_size), self.num_pathways}) == 1
    in_chans = cfg.DATA.INPUT_CHANNEL_NUM

    stage_idx = 0
    layerDict = OrderedDict()
    out_chans = [cfg.SQUEEZENET.STEM_IN_CHANS, cfg.SQUEEZENET.STEM_IN_CHANS // cfg.SLOWFAST.BETA_INV]
    stem = SlowFastSqueezeNetStem(
        dim_in=in_chans,
        dim_out=out_chans,
        conv_kernels=cfg.SQUEEZENET.STEM_CONV_KERNEL,
        conv_stride=cfg.SQUEEZENET.STEM_CONV_STRIDE,
        pool_kernel=cfg.SQUEEZENET.STEM_POOL_KERNEL,
        pool_stride=cfg.SQUEEZENET.STEM_POOL_STRIDE
    )
    layerDict["fire_0"] = stem
    in_chans = out_chans

    for stage_idx in range(len(module_settings)):
      s1x1, e1x1, e3x3, tkf, fuse, addMaxPool = module_settings[stage_idx]
      s1x1s = [s1x1, s1x1 // cfg.SLOWFAST.BETA_INV]

      e1x1f = e1x1 // cfg.SLOWFAST.BETA_INV
      e3x3f = e3x3 // cfg.SLOWFAST.BETA_INV
      if fuse:
        e1x1 = e1x1 - int(e1x1f * cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO)
        e3x3 = e3x3 - int(e3x3f * cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO)

      e1x1s = [e1x1, e1x1f]
      e3x3s = [e3x3, e3x3f]
      kernels = [1, tkf]
      layerDict['fire_{}'.format(stage_idx+1)] = SlowFastSqueezeFire(
          dim_in=in_chans,
          squeeze_planes=s1x1s,
          expand1x1_planes=e1x1s,
          expand3x3_planes=e3x3s,
          temp_kernels=kernels,
          addMaxPool=addMaxPool,
          idx=stage_idx+2
      )

      # if addMaxPool:
      #   layerDict['maxpool_{}'.format(stage_idx+1)] = nn.MaxPool3d(
      #     kernel_size=[1, 3, 3],
      #     stride=(1,2.2), ceil_mode=True)

      out_chans = [x1 + x3 for x1, x3 in zip(e1x1s, e3x3s)]
      in_chans = out_chans

      if fuse > 0:
        fuse = FuseFastToSlow(
                  out_chans[-1],    # Fast numChannels
                  cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO,
                  cfg.SLOWFAST.FUSION_KERNEL_SZ,
                  cfg.SLOWFAST.ALPHA,
                  norm_module=self.norm_module,
              )
        layerDict["fire_{}_fuse".format(stage_idx+1)] = fuse
        in_chans[0] += out_chans[-1] * cfg.SLOWFAST.FUSION_CONV_CHANNEL_RATIO 

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