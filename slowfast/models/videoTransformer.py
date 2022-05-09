""" Vision Transformer (ViT) in PyTorch
A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929
The official jax code is released and available at https://github.com/google-research/vision_transformer
Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert
Hacked together by / Copyright 2020 Ross Wightman
"""

from collections import OrderedDict

import torch
import torch.nn as nn
from slowfast.models.build import MODEL_REGISTRY

def trunc_normal_(x, mean=0., std=1.):
  "Truncated normal initialization (approximation)"
  # From https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
  return x.normal_().fmod_(2).mul_(std).add_(mean)

def drop_path(x, drop_prob: float = 0., training: bool = False):
  """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
  This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
  the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
  See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
  changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
  'survival rate' as the argument.
  """
  if drop_prob == 0. or not training:
    return x
  keep_prob = 1 - drop_prob
  shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
  random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
  random_tensor.floor_()  # binarize
  output = x.div(keep_prob) * random_tensor
  return output


class DropPath(nn.Module):
  """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
  """

  def __init__(self, drop_prob=None):
    super(DropPath, self).__init__()
    self.drop_prob = drop_prob

  def forward(self, x):
    return drop_path(x, self.drop_prob, self.training)


class TAggregate(nn.Module):
  def __init__(self, clip_length=None, embed_dim=2048, n_layers=6, nhead=8):
    super(TAggregate, self).__init__()
    self.clip_length = clip_length
    # self.nvids = nvids
    # embed_dim = 2048
    drop_rate = 0.
    enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead)
    self.transformer_enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers, norm=nn.LayerNorm(
      embed_dim))

    self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    self.pos_embed = nn.Parameter(torch.zeros(1, clip_length + 1, embed_dim))
    self.pos_drop = nn.Dropout(p=drop_rate)

    with torch.no_grad():
      trunc_normal_(self.pos_embed, std=.02)
      trunc_normal_(self.cls_token, std=.02)
    self.apply(self._init_weights)

  def _init_weights(self, m):
    if isinstance(m, nn.Linear):
      with torch.no_grad():
        trunc_normal_(m.weight, std=.02)
      if isinstance(m, nn.Linear) and m.bias is not None:
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
      nn.init.constant_(m.bias, 0)
      nn.init.constant_(m.weight, 1.0)


  def forward(self, x):
    nvids = x.shape[0] // self.clip_length
    x = x.view((nvids, self.clip_length, -1))

    cls_tokens = self.cls_token.expand(nvids, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + self.pos_embed
    # x = self.pos_drop(x)

    x.transpose_(1,0)
    o = self.transformer_enc(x)

    return o[0]


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


@MODEL_REGISTRY.register()
class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, cfg):
        super().__init__()
        norm_layer = nn.LayerNorm
        self.aggregate = TAggregate(
            cfg.MODEL.FRAMES_PER_CLIP, 
            embed_dim=cfg.MODEL.EMBED_DIM,
            n_layers=cfg.MODEL.TEMP_DEPTH,
            nhead=cfg.MODEL.NUM_HEADS)
        # self.num_classes = num_classes
        self.num_features = self.embed_dim = cfg.MODEL.EMBED_DIM  # num_features for consistency with other models
        
        assert cfg.DATA.FRAME_WIDTH == cfg.DATA.FRAME_HEIGHT, "Assume square input farmes found W = {} H = {}".format(cfg.DATA.FRAME_WIDTH, cfg.DATA.FRAME_HEIGHT)
        self.patch_embed = PatchEmbed(
            img_size=cfg.DATA.FRAME_WIDTH, patch_size=cfg.MODEL.PATCH_SIZE, 
            in_chans=cfg.DATA.INPUT_CHANNEL_NUM[0], embed_dim=cfg.MODEL.EMBED_DIM)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.MODEL.EMBED_DIM))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, cfg.MODEL.EMBED_DIM))
        self.pos_drop = nn.Dropout(p=cfg.MODEL.DROP_RATE)

        dpr = [x.item() for x in torch.linspace(0, cfg.MODEL.DROP_PATH_RATE, cfg.MODEL.SPAT_DEPTH)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=cfg.MODEL.EMBED_DIM, num_heads=cfg.MODEL.NUM_HEADS, mlp_ratio=cfg.MODEL.MLP_RATIO, 
                qkv_bias=cfg.MODEL.QKV_BIAS, qk_scale=cfg.MODEL.QK_SCALE,
                drop=cfg.MODEL.DROP_RATE, attn_drop=cfg.MODEL.ATTN_DROP_RATE, drop_path=dpr[i], norm_layer=norm_layer,)
            for i in range(cfg.MODEL.SPAT_DEPTH)])
        self.norm = norm_layer(cfg.MODEL.EMBED_DIM)

        # Representation layer
        if cfg.MODEL.REPRESENTATION_SIZE:
            self.num_features = cfg.MODEL.REPRESENTATION_SIZE
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(cfg.MODEL.EMBED_DIM, cfg.MODEL.REPRESENTATION_SIZE)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        print("Created VisionTransformer")


        # Classifier head
        self.head = nn.Linear(cfg.MODEL.EMBED_DIM, cfg.MODEL.NUM_CLASSES) if cfg.MODEL.NUM_CLASSES > 0 else nn.Identity()

        with torch.no_grad():
          trunc_normal_(self.pos_embed, std=.02)
          trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for ib, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)[:, 0]
        x = self.pre_logits(x)
        return x

    def forward(self, xl):
        # Expect BxCxTxWxH Convert to 
        # BTxCxWxH
        x = xl[-1]
        B, C, T, W, H = x.shape
        x = x.permute(0, 2, 1, 3, 4).view(B*T, C, W, H)
        x = self.forward_features(x)
        if self.aggregate:
            x = self.aggregate(x)

        x = self.head(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

#
class PatchEmbed(nn.Module):
  """ Image to Patch Embedding
  """

  def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
    super().__init__()
    img_size = (img_size, img_size)
    patch_size = (patch_size, patch_size)
    num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
    self.img_size = img_size
    self.patch_size = patch_size
    self.num_patches = num_patches

    self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

  def forward(self, x):
    B, C, H, W = x.shape
    assert H == self.img_size[0] and W == self.img_size[1], \
      f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
    x = self.proj(x).flatten(2).transpose(1, 2)
    x = self.norm(x)
    return x


def STAM_224(model_params):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    args = model_params['args']
    num_classes = model_params['num_classes']
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, representation_size=None)
    img_size = args.input_size
    t_layers = 6

    aggregate = TAggregate(args.frames_per_clip, embed_dim=768, n_layers=t_layers)

    model = VisionTransformer(img_size=img_size, num_classes=num_classes, aggregate=aggregate,
                              **model_kwargs)
    return model

