#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
from slowfast.models import head_helper
from slowfast.models.attention import attention_pool
from slowfast.models.utils import calc_mvit_feature_geometry
from slowfast.models.video_model_builder import MViT

from . import head_helper, operators, resnet_helper, stem_helper  # noqa
from .build import MODEL_REGISTRY

logger = logging.get_logger(__name__)


@MODEL_REGISTRY.register()
class MaskMViT(MViT):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.pretrain_depth = cfg.MASK.PRETRAIN_DEPTH
        if self.pretrain_depth[-1] + 1 < cfg.MVIT.DEPTH:
            del self.blocks[self.pretrain_depth[-1] + 1 :]
        del self.norm
        del self.head
        self.feat_size, self.feat_stride = calc_mvit_feature_geometry(cfg)

        self.head_type = cfg.MASK.HEAD_TYPE.split("_")
        feat_sz = [self.feat_size[depth] for depth in self.pretrain_depth]
        if self.head_type[0] == "separate":
            if not cfg.MASK.PRED_HOG:
                pred_t_sz = (
                    1
                    if self.cfg.MASK.TIME_STRIDE_LOSS
                    else self.patch_stride[0]
                )
                num_classes = [
                    pred_t_sz * (self.feat_stride[depth][-1] ** 2) * 3
                    for depth in self.pretrain_depth
                ]
                self.pred_head = head_helper.MSSeparateHead(
                    self.blocks, cfg, num_classes, feat_sz
                )
            else:
                self.hogs = nn.ModuleList()
                self.nbins = 9
                self.cell_sz = 8
                self.hogs.append(
                    operators.HOGLayerC(
                        nbins=self.nbins,
                        pool=self.cell_sz,
                    )
                )
                self.ncells = [
                    (self.feat_stride[depth][-1] // self.cell_sz) ** 2
                    for depth in self.pretrain_depth
                ]
                pred_hog_classes = [self.nbins * ncell for ncell in self.ncells]
                pred_hog_classes = [
                    pred_hog_class * 3  # 3 color channels
                    for pred_hog_class in pred_hog_classes
                ]
                self.pred_head = head_helper.MSSeparateHead(
                    self.blocks, cfg, pred_hog_classes, feat_sz
                )
                self.hog_loss = "mse"
        else:
            raise NotImplementedError

        embed_dim = cfg.MVIT.EMBED_DIM
        decoder_embed_dim = cfg.MASK.DECODER_EMBED_DIM
        self.sep_pos_embed_decoder = cfg.MASK.DECODER_SEP_POS_EMBED
        self.counter = 0
        if cfg.MASK.MAE_ON:
            # ----------------------------------------------------------------
            # MAE decoder specifics
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
            dim_in = self.blocks[-1].dim_out
            self.norm = norm_layer(dim_in)
            self.decoder_embed = nn.Linear(dim_in, decoder_embed_dim, bias=True)
            num_patches = math.prod(self.patch_dims)
            if self.use_abs_pos:
                if self.sep_pos_embed_decoder:
                    self.dec_pos_embed_spatial = nn.Parameter(
                        torch.zeros(
                            1,
                            self.patch_dims[1] * self.patch_dims[2],
                            decoder_embed_dim,
                        )
                    )
                    self.dec_pos_embed_temporal = nn.Parameter(
                        torch.zeros(1, self.patch_dims[0], decoder_embed_dim)
                    )
                    if self.cls_embed_on:
                        self.dec_pos_embed_class = nn.Parameter(
                            torch.zeros(1, 1, decoder_embed_dim)
                        )
                else:
                    self.decoder_pos_embed = nn.Parameter(
                        torch.zeros(
                            1,
                            num_patches + 1
                            if self.cls_embed_on
                            else num_patches,
                            decoder_embed_dim,
                        )
                    )
        self.mask_token = nn.Parameter(
            torch.zeros(
                1, 1, decoder_embed_dim if cfg.MASK.MAE_ON else embed_dim
            )
        )
        trunc_normal_(self.mask_token, std=0.02)
        if self.use_abs_pos and cfg.MASK.MAE_ON:
            if self.sep_pos_embed_decoder:
                trunc_normal_(self.dec_pos_embed_spatial, std=0.02)
                trunc_normal_(self.dec_pos_embed_temporal, std=0.02)
                if self.cls_embed_on:
                    trunc_normal_(self.dec_pos_embed_class, std=0.02)
            else:
                trunc_normal_(self.decoder_pos_embed, std=0.02)

        if cfg.MASK.SCALE_INIT_BY_DEPTH:
            self.fix_init_weight()

        self.pred_pixel_wt = 0.0 if cfg.MASK.PRED_HOG else 1.0
        self.pred_hog_wt = 1.0 if cfg.MASK.PRED_HOG else 0.0

    @torch.jit.ignore
    def no_weight_decay(self):
        names = []
        if self.cfg.MVIT.ZERO_DECAY_POS_CLS:
            if self.use_abs_pos:
                if self.sep_pos_embed_decoder:
                    names.extend(
                        [
                            "dec_pos_embed_spatial",
                            "dec_pos_embed_temporal",
                            "dec_pos_embed_class",
                        ]
                    )
                else:
                    names.extend(["pos_embed_decoder"])
            if self.cls_embed_on:
                names.append("cls_token")

        return names

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)
        for trans in self.pred_head.transforms:
            for layer_id, layer in enumerate(trans):
                if hasattr(layer, "attn"):
                    rescale(
                        layer.attn.proj.weight.data,
                        layer_id + 1 + len(self.blocks),
                    )  # or + len(self.blocks)
                    rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _get_multiscale_mask(self, mask):
        if self.use_2d_patch:
            mask = mask.unsqueeze(0)
        output_masks = []
        for depth in self.pretrain_depth:
            size = self.feat_size[depth][-1]
            output_mask = F.interpolate(mask, size=size)
            if self.use_2d_patch:
                output_mask = output_mask[0]
            output_mask = output_mask.flatten(1).to(torch.bool)
            output_masks.append(output_mask)
        return output_masks

    def _patchify(self, imgs, p=16, time_stride_loss=True):
        N, _, T, H, W = imgs.shape
        u = 1 if time_stride_loss else self.patch_stride[0]
        assert H == W and H % p == 0 and T % u == 0
        h = w = H // p
        t = T // u
        x = imgs.reshape(shape=(N, 3, t, u, h, p, w, p))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, u * p**2 * 3))
        self.patch_info = (N, T, H, W, p, u, t, h, w)
        return x

    def _unpatchify(self, x):
        N, T, H, W, p, u, t, h, w = self.patch_info
        x = x.reshape(shape=(N, t, h, w, u, p, p, 3))
        x = torch.einsum("nthwupqc->nctuhpwq", x)
        imgs = x.reshape(shape=(N, 3, T, H, W))
        return imgs

    def _get_pixel_label_2d(self, input_img, output_masks, norm=True):
        input_img = input_img.permute(0, 2, 3, 1)
        labels = []
        for depth, output_mask in zip(self.pretrain_depth, output_masks):
            size = self.feat_stride[depth][-1]
            label = input_img.unfold(1, size, size).unfold(2, size, size)
            label = label.flatten(1, 2).flatten(2)
            label = label[output_mask]
            if norm:
                mean = label.mean(dim=-1, keepdim=True)
                var = label.var(dim=-1, keepdim=True)
                label = (label - mean) / (var + 1.0e-6) ** 0.5
            labels.append(label)
        return labels

    def _get_pixel_label_3d(
        self, input_frames, output_masks, time_stride_loss=True, norm=True
    ):
        if time_stride_loss:
            input_frames = input_frames[
                :, :, :: self.cfg.MVIT.PATCH_STRIDE[0], :, :
            ]
        imgs = input_frames
        input_frames = input_frames.permute(0, 2, 3, 4, 1)
        labels = []
        for depth, output_mask in zip(self.pretrain_depth, output_masks):
            size = self.feat_stride[depth][-1]
            label = self._patchify(
                imgs, p=size, time_stride_loss=time_stride_loss
            )
            label = label[output_mask]

            if norm:  # self.norm_pix_loss:
                mean = label.mean(dim=-1, keepdim=True)
                var = label.var(dim=-1, keepdim=True)
                label = (label - mean) / (var + 1.0e-6) ** 0.5
            labels.append(
                (label, self.pred_pixel_wt / len(self.pretrain_depth))
            )
        return labels

    def _get_hog_label_2d(self, input_frames, output_masks):
        # input_frames, B C H W
        labels = []
        for depth, output_mask in zip(self.pretrain_depth, output_masks):
            feat_size = self.feat_size[depth][-1]
            hog_list = []
            for hog in self.hogs:
                tmp_hog = hog(input_frames).flatten(1, 2)  # return B C H W
                unfold_size = tmp_hog.shape[-1] // feat_size
                tmp_hog = (
                    tmp_hog.permute(0, 2, 3, 1)
                    .unfold(1, unfold_size, unfold_size)
                    .unfold(2, unfold_size, unfold_size)
                    .flatten(1, 2)
                    .flatten(2)
                )
                tmp_hog = tmp_hog[output_mask]
                hog_list.append(tmp_hog)
            all_tlabel = torch.cat(hog_list, -1)
            labels.append((all_tlabel, self.pred_hog_wt, self.hog_loss))
        return labels

    def _get_hog_label_3d(self, input_frames, output_masks):
        input_frames = input_frames[
            :, :, :: self.cfg.MVIT.PATCH_STRIDE[0], :, :
        ]  # B C T H W
        input_frames = input_frames.transpose(1, 2)  # B T C H W
        B, T = input_frames.shape[:2]
        input_frames = input_frames.flatten(0, 1)  # BT C H W
        labels = []
        for depth, output_mask in zip(self.pretrain_depth, output_masks):
            feat_size = self.feat_size[depth][-1]
            hog_list = []
            for hog in self.hogs:
                tmp_hog = hog(input_frames).flatten(1, 2)  # BT C H W
                unfold_size = tmp_hog.shape[-1] // feat_size
                tmp_hog = (
                    tmp_hog.permute(0, 2, 3, 1)
                    .unfold(1, unfold_size, unfold_size)
                    .unfold(2, unfold_size, unfold_size)
                )  # BT h w C wh ww
                tmp_hog = tmp_hog.flatten(3).view(
                    B, T, feat_size, feat_size, -1
                )  # B T h w C (3 nbins h w)
                tmp_hog = tmp_hog.flatten(1, 3)  # B N C
                tmp_hog = tmp_hog[output_mask]
                hog_list.append(tmp_hog)
            all_tlabel = torch.cat(hog_list, -1)
            labels.append((all_tlabel, self.pred_hog_wt, self.hog_loss))
        return labels

    def _mae_random_masking(self, x, mask_ratio, mask_in=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        if mask_in is None:
            if self.cfg.AUG.MASK_TUBE:
                noise = (
                    torch.rand(N, 1, self.H * self.W, device=x.device)
                    .repeat([1, self.T, 1])
                    .reshape(N, L)
                )  # noise in [0, 1]
            else:
                noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        else:
            noise = mask_in.flatten(1)
            mask_ratio = sum(noise.flatten()) / noise.numel()  # alrdy masked
        len_keep = int(L * (1 - mask_ratio))
        assert len_keep > 1
        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
        )
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore, ids_keep

    def _mae_forward_encoder(self, x, mask_ratio, mask=None):
        x, bcthw = self.patch_embed(x, keep_spatial=False)
        bcthw = list(bcthw)
        if len(bcthw) == 4:  # Fix bcthw in case of 4D tensor
            bcthw.insert(2, torch.tensor(self.T))
        T, H, W = bcthw[-3], bcthw[-2], bcthw[-1]
        assert len(bcthw) == 5 and (T, H, W) == (self.T, self.H, self.W), bcthw
        s = 1 if self.cls_embed_on else 0
        B, N, C = x.shape

        if self.use_fixed_sincos_pos:
            x += self.pos_embed[:, s:, :]  # 0: no cls token

        if self.cfg.MASK.PER_FRAME_MASKING:
            x = x.reshape([B * T, H * W, C])
        x, mask, ids_restore, ids_keep = self._mae_random_masking(
            x, mask_ratio, None if self.cfg.MASK.MAE_RND_MASK else mask
        )
        if self.cfg.MASK.PER_FRAME_MASKING:
            x = x.view([B, -1, C])

        if self.cls_embed_on:
            # append cls token
            cls_token = self.cls_token  #
            if self.use_fixed_sincos_pos:
                cls_token = cls_token + self.pos_embed[:, :s, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos and not self.use_fixed_sincos_pos:
            if self.sep_pos_embed:
                pos_embed = self.pos_embed_spatial.repeat(
                    1, self.patch_dims[0], 1
                ) + torch.repeat_interleave(
                    self.pos_embed_temporal,
                    self.patch_dims[1] * self.patch_dims[2],
                    dim=1,
                )
                pos_embed = pos_embed.expand(x.shape[0], -1, -1)
                pos_embed = torch.gather(
                    pos_embed,
                    dim=1,
                    index=ids_keep.unsqueeze(-1).repeat(
                        1, 1, pos_embed.shape[2]
                    ),
                )
                if self.cls_embed_on:
                    pos_embed = torch.cat(
                        [
                            self.pos_embed_class.expand(
                                pos_embed.shape[0], -1, -1
                            ),
                            pos_embed,
                        ],
                        1,
                    )
                x += pos_embed
            else:
                pos_embed = self.pos_embed.expand(x.shape[0], -1, -1)
                pos_embed_sampled = torch.gather(
                    pos_embed[:, s:, :],
                    dim=1,
                    index=ids_keep.unsqueeze(-1).repeat(
                        1, 1, self.pos_embed.shape[2]
                    ),
                )
                if self.cls_embed_on:
                    pos_embed_sampled = torch.cat(
                        [pos_embed[:, :s, :], pos_embed_sampled], 1
                    )
                x += pos_embed_sampled

        # apply Transformer blocks
        B, N, C = x.shape
        thw = [T, H, W]
        for _, blk in enumerate(self.blocks):
            x, thw = blk(x, thw)
        x = self.norm(x)

        return x, mask, ids_restore, thw

    def _mae_forward_decoder(self, x, ids_restore, mask, thw):
        # embed tokens
        x = self.decoder_embed(x)
        T, H, W = self.T, self.H, self.W
        B, N, C = x.shape

        s = 1 if self.cls_embed_on else 0

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            B, T * H * W + s - x.shape[1], 1
        )  # + s: no cls token
        x_ = torch.cat([x[:, s:, :], mask_tokens], dim=1)  # no cls token
        if self.cfg.MASK.PER_FRAME_MASKING:
            x_ = x_.view([B * T, H * W, C])
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
        )  # unshuffle
        if self.cfg.MASK.PER_FRAME_MASKING:
            x_ = x_.view([B, T * H * W, C])
        x = torch.cat([x[:, :s, :], x_], dim=1)  # append cls token

        if self.sep_pos_embed_decoder:
            pos_embed = self.dec_pos_embed_spatial.repeat(
                1, self.patch_dims[0], 1
            ) + torch.repeat_interleave(
                self.dec_pos_embed_temporal,
                self.patch_dims[1] * self.patch_dims[2],
                dim=1,
            )
            pos_embed = pos_embed.expand(x.shape[0], -1, -1)
            if self.cls_embed_on:
                pos_embed = torch.cat(
                    [
                        self.dec_pos_embed_class.expand(
                            pos_embed.shape[0], -1, -1
                        ),
                        pos_embed,
                    ],
                    1,
                )
            x += pos_embed
        else:
            # add pos embed
            x = x + self.decoder_pos_embed

        pixel_outputs = self.pred_head(
            [x],
            [mask.to(torch.bool)],
            return_all=self.cfg.VIS_MASK.ENABLE,
            thw=thw,
        )

        return pixel_outputs

    def _mae_forward(self, imgs, mask_ratio=0.75, mask=None):
        latent, mask, ids_restore, thw = self._mae_forward_encoder(
            imgs, mask_ratio, mask
        )
        pred = self._mae_forward_decoder(latent, ids_restore, mask, thw)
        labels = []
        if self.pred_pixel_wt:
            if self.use_2d_patch:
                labels += self._get_pixel_label_2d(
                    imgs.detach(),
                    [mask.to(torch.bool)],
                    norm=self.cfg.MASK.NORM_PRED_PIXEL,
                )
            else:
                labels += self._get_pixel_label_3d(
                    imgs.detach(),
                    [mask.to(torch.bool)],
                    time_stride_loss=self.cfg.MASK.TIME_STRIDE_LOSS,
                    norm=self.cfg.MASK.NORM_PRED_PIXEL,
                )
        if self.pred_hog_wt:
            if self.use_2d_patch:
                labels += self._get_hog_label_2d(
                    imgs.detach(), [mask.to(torch.bool)]
                )
            else:
                labels += self._get_hog_label_3d(
                    imgs.detach(), [mask.to(torch.bool)]
                )

        self.counter += 1
        if self.cfg.VIS_MASK.ENABLE:
            return self._mae_visualize(imgs, pred, mask)
        return pred, labels

    def _mae_visualize(self, imgs, pred, mask):
        N, T, H, W, p, u, t, h, w = self.patch_info
        pred = pred[0]
        if self.cfg.MASK.TIME_STRIDE_LOSS:
            im_viz = imgs[:, :, :: self.cfg.MVIT.PATCH_STRIDE[0], :, :]
        else:
            im_viz = imgs
        reconstruct = self._unpatchify(
            pred * mask.reshape(N, t * h * w, 1)
            + self._patchify(
                im_viz, time_stride_loss=self.cfg.MASK.TIME_STRIDE_LOSS
            )
            * (1 - mask.reshape(N, t * h * w, 1))
        )
        masked = self._unpatchify(
            self._patchify(
                im_viz, time_stride_loss=self.cfg.MASK.TIME_STRIDE_LOSS
            )
            * (1 - mask.reshape(N, t * h * w, 1))
        )

        comparison = torch.stack(
            [im_viz, masked, reconstruct],
            dim=1,
        ).permute([0, 1, 3, 2, 4, 5])
        pfx = self.cfg.TEST.CHECKPOINT_FILE_PATH
        mr = self.cfg.AUG.MASK_RATIO
        for i in range(comparison.shape[0]):
            misc.plot_input_normed(
                comparison[i].cpu(),
                bboxes=(),
                texts=(),
                path=self.cfg.OUTPUT_DIR
                + "/vis_mask/vid/{}vis_video_in_mask_out_mr{}/vis_{}_{}.mp4".format(
                    pfx[pfx.rfind("/") + 1 : -5], mr, self.counter, i
                ),
                folder_path=self.cfg.OUTPUT_DIR
                + "/vis_mask/vid/{}vis_video_in_mask_out_mr{}".format(
                    pfx[pfx.rfind("/") + 1 : -5], mr
                ),
                make_grids=True,
                output_video=True,
            )
        return pred[0]

    def _maskfeat_forward(self, x, mask, return_all=False):
        x_embed, x_shape = self.patch_embed(x)
        if self.cfg.MASK.MAE_RND_MASK:
            _, mask, ids_restore, ids_keep = self._mae_random_masking(
                x_embed, self.cfg.AUG.MASK_RATIO, None
            )
            output_masks = [mask.to(torch.bool)]
        else:
            # take masks and labels from loader
            float_mask = mask.type_as(x)
            output_masks = self._get_multiscale_mask(float_mask)
        labels = []
        if self.pred_pixel_wt:
            if self.use_2d_patch:
                labels += self._get_pixel_label_2d(
                    x.detach(), output_masks, norm=self.cfg.MASK.NORM_PRED_PIXEL
                )
            else:
                labels += self._get_pixel_label_3d(
                    x.detach(), output_masks, norm=self.cfg.MASK.NORM_PRED_PIXEL
                )
        if self.pred_hog_wt:
            if self.use_2d_patch:
                labels += self._get_hog_label_2d(x.detach(), output_masks)
            else:
                labels += self._get_hog_label_3d(x.detach(), output_masks)

        x = x_embed
        T, H, W = self.T, self.H, self.W
        B, N, C = x.shape

        # switch input tokens by mask_token
        mask_tokens = self.mask_token.expand(B, N, -1)
        if self.cfg.MASK.MAE_RND_MASK:
            float_mask = mask.unsqueeze(-1)
        else:
            if self.use_2d_patch:
                float_mask = F.interpolate(
                    float_mask.unsqueeze(0), size=(H, W)
                )[0]
            else:
                float_mask = F.interpolate(float_mask, size=(H, W))
            float_mask = float_mask.flatten(1).unsqueeze(-1)
        x = x * (1 - float_mask) + mask_tokens * float_mask

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.use_abs_pos:
            if self.sep_pos_embed:
                pos_embed = self.pos_embed_spatial.repeat(
                    1, self.patch_dims[0], 1
                ) + torch.repeat_interleave(
                    self.pos_embed_temporal,
                    self.patch_dims[1] * self.patch_dims[2],
                    dim=1,
                )
                if self.cls_embed_on:
                    pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
                x = x + pos_embed
            else:
                x = x + self.pos_embed

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]
        block_outputs = []
        for idx, blk in enumerate(self.blocks):
            x, thw = blk(x, thw)
            if idx in self.pretrain_depth:
                block_outputs.append(x)

        model_outputs = []
        if self.pred_pixel_wt:
            pixel_outputs = self.pred_head(
                block_outputs,
                output_masks,
                return_all=return_all,
                thw=thw,
            )
            model_outputs += pixel_outputs
        if self.pred_hog_wt:
            hog_outputs = self.pred_head(
                block_outputs,
                output_masks,
                return_all=return_all,
                thw=thw,
            )
            model_outputs += hog_outputs

        return model_outputs, labels

    def forward(self, x, return_all=False):
        if len(x) > 1:
            x, meta, mask = x
        else:
            x, mask = x[0], None

        if self.cfg.MASK.MAE_ON:
            return self._mae_forward(
                x, mask_ratio=self.cfg.AUG.MASK_RATIO, mask=mask
            )
        else:
            return self._maskfeat_forward(x, mask, return_all)
