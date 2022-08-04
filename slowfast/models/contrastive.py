#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import slowfast.models.losses as losses
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
from slowfast.models.video_model_builder import X3D, MViT, ResNet, SlowFast

from .build import MODEL_REGISTRY

logger = logging.get_logger(__name__)

# Supported model types
_MODEL_TYPES = {
    "slowfast": SlowFast,
    "slow": ResNet,
    "c2d": ResNet,
    "i3d": ResNet,
    "slow_c2d": ResNet,
    "x3d": X3D,
    "mvit": MViT,
}


@MODEL_REGISTRY.register()
class ContrastiveModel(nn.Module):
    """
    Contrastive Model, currently mainly focused on memory bank and CSC.
    """

    def __init__(self, cfg):
        super(ContrastiveModel, self).__init__()
        # Construct the model.
        self.backbone = _MODEL_TYPES[cfg.MODEL.ARCH](cfg)
        self.type = cfg.CONTRASTIVE.TYPE
        self.T = cfg.CONTRASTIVE.T
        self.dim = cfg.CONTRASTIVE.DIM
        self.length = cfg.CONTRASTIVE.LENGTH
        self.k = cfg.CONTRASTIVE.QUEUE_LEN
        self.mmt = cfg.CONTRASTIVE.MOMENTUM
        self.momentum_annealing = cfg.CONTRASTIVE.MOMENTUM_ANNEALING
        self.duration = 1
        self.cfg = cfg
        self.num_gpus = cfg.NUM_GPUS
        self.l2_norm = Normalize()
        self.knn_num_imgs = 0
        self.knn_on = cfg.CONTRASTIVE.KNN_ON
        self.train_labels = np.zeros((0,), dtype=np.int32)
        self.num_pos = 2
        self.num_crops = (
            self.cfg.DATA.TRAIN_CROP_NUM_TEMPORAL
            * self.cfg.DATA.TRAIN_CROP_NUM_SPATIAL
        )
        self.nce_loss_fun = losses.get_loss_func("contrastive_loss")(
            reduction="mean"
        )
        assert self.cfg.MODEL.LOSS_FUNC == "contrastive_loss"
        self.softmax = nn.Softmax(dim=1).cuda()

        if self.type == "mem":
            self.mem_type = cfg.CONTRASTIVE.MEM_TYPE
            if self.mem_type == "1d":
                self.memory = Memory1D(
                    self.length, self.duration, self.dim, cfg
                )
            else:
                self.memory = Memory(self.length, self.duration, self.dim, cfg)
            self.examplar_type = "video"
            self.interp = cfg.CONTRASTIVE.INTERP_MEMORY
        elif self.type == "self":
            pass
        elif self.type == "moco" or self.type == "byol":
            # MoCo components
            self.backbone_hist = _MODEL_TYPES[cfg.MODEL.ARCH](cfg)
            for p in self.backbone_hist.parameters():
                p.requires_grad = False
            self.register_buffer("ptr", torch.tensor([0]))
            self.ptr.requires_grad = False
            stdv = 1.0 / math.sqrt(self.dim / 3)
            self.register_buffer(
                "queue_x",
                torch.rand(self.k, self.dim).mul_(2 * stdv).add_(-stdv),
            )
            self.register_buffer("iter", torch.zeros([1], dtype=torch.long))
            self._batch_shuffle_on = (
                False
                if (
                    "sync" in cfg.BN.NORM_TYPE
                    and cfg.BN.NUM_SYNC_DEVICES == cfg.NUM_GPUS
                )
                or self.type == "byol"
                else True
            )
        elif self.type == "swav":
            self.swav_use_public_code = True
            if self.swav_use_public_code:
                self.swav_prototypes = nn.Linear(
                    self.dim, 1000, bias=False
                )  # for orig implementation
            else:
                self.swav_prototypes = nn.Parameter(
                    torch.randn((self.dim, 1000), dtype=torch.float)
                )
            self.swav_eps_sinkhorn = 0.05
            self.swav_use_the_queue = False
            # optionally starts a queue
            if self.cfg.CONTRASTIVE.SWAV_QEUE_LEN > 0:
                self.register_buffer(
                    "queue_swav",
                    torch.zeros(
                        2,  # = args.crops_for_assign
                        self.cfg.CONTRASTIVE.SWAV_QEUE_LEN
                        // du.get_world_size(),
                        self.dim,
                    ),
                )
        elif self.type == "simclr":
            self._simclr_precompute_pos_neg_mask_multi()
        self.simclr_dist_on = cfg.CONTRASTIVE.SIMCLR_DIST_ON

        # self.knn_mem = Memory1D(self.length, 1, self.dim, cfg) #  does not work
        if self.knn_on:
            self.knn_mem = Memory(self.length, 1, self.dim, cfg)

    @torch.no_grad()
    def knn_mem_update(self, q_knn, index):
        if self.knn_on:
            self.knn_mem.update(
                q_knn,
                momentum=1.0,
                ind=index,
                time=torch.zeros_like(index),
                interp=False,
            )

    @torch.no_grad()
    def init_knn_labels(self, train_loader):
        logger.info("initializing knn labels")
        self.num_imgs = len(train_loader.dataset._labels)
        self.train_labels = np.zeros((self.num_imgs,), dtype=np.int32)
        for i in range(self.num_imgs):
            self.train_labels[i] = train_loader.dataset._labels[i]
        self.train_labels = torch.LongTensor(self.train_labels).cuda()
        if self.length != self.num_imgs:
            logger.error(
                "Kinetics dataloader size: {} differs from memorybank length {}".format(
                    self.num_imgs, self.length
                )
            )
            self.knn_mem.resize(self.num_imgs, 1, self.dim)

    @torch.no_grad()
    def _update_history(self):
        # momentum update
        iter = int(self.iter)
        m = self.mmt
        dist = {}
        for name, p in self.backbone.named_parameters():
            dist[name] = p

        if iter == 0:
            for name, p in self.backbone_hist.named_parameters():
                p.data.copy_(dist[name].data)

        for name, p in self.backbone_hist.named_parameters():
            p.data = dist[name].data * (1.0 - m) + p.data * m

    @torch.no_grad()
    def _batch_shuffle(self, x):
        if len(x) == 2:
            another_crop = True
        else:
            another_crop = False
        if another_crop:
            x, x_crop = x[0], x[1]
        else:
            x = x[0]

        world_size = self.cfg.NUM_GPUS * self.cfg.NUM_SHARDS
        if self.num_gpus > 1:
            if self.cfg.CONTRASTIVE.LOCAL_SHUFFLE_BN:
                x = du.cat_all_gather(x, local=True)
                if another_crop:
                    x_crop = du.cat_all_gather(x_crop, local=True)
                world_size = du.get_local_size()
                gpu_idx = du.get_local_rank()
            else:
                x = du.cat_all_gather(x)
                if another_crop:
                    x_crop = du.cat_all_gather(x_crop)
                gpu_idx = torch.distributed.get_rank()

        idx_randperm = torch.randperm(x.shape[0]).cuda()
        if self.num_gpus > 1:
            torch.distributed.broadcast(idx_randperm, src=0)
        else:
            gpu_idx = 0
        idx_randperm = idx_randperm.view(world_size, -1)
        x = x[idx_randperm[gpu_idx, :]]
        if another_crop:
            x_crop = x_crop[idx_randperm[gpu_idx, :]]

        idx_restore = torch.argsort(idx_randperm.view(-1))
        idx_restore = idx_restore.view(world_size, -1)
        if another_crop:
            return [x, x_crop], idx_restore
        else:
            return [x], idx_restore

    @torch.no_grad()
    def _batch_unshuffle(self, x, idx_restore):
        if self.num_gpus > 1:
            if self.cfg.CONTRASTIVE.LOCAL_SHUFFLE_BN:
                x = du.cat_all_gather(x, local=True)
                gpu_idx = du.get_local_rank()
            else:
                x = du.cat_all_gather(x)
                gpu_idx = torch.distributed.get_rank()
        else:
            gpu_idx = 0

        idx = idx_restore[gpu_idx, :]
        x = x[idx]
        return x

    @torch.no_grad()
    def eval_knn(self, q_knn, knn_k=200):
        with torch.no_grad():
            dist = torch.einsum(
                "nc,mc->nm",
                q_knn.view(q_knn.size(0), -1),
                self.knn_mem.memory.view(self.knn_mem.memory.size(0), -1),
            )
            yd, yi = dist.topk(knn_k, dim=1, largest=True, sorted=True)
        return yd, yi

    def sim_loss(self, q, k):
        similarity = torch.einsum("nc,nc->n", [q, k])  # N-dim
        # similarity += delta_t # higher if time distance is larger
        # sim = sim - max_margin + delta_t * k
        similarity /= self.T  # history-compatible
        loss = -similarity.mean()
        return loss

    @torch.no_grad()
    def momentum_anneal_cosine(self, epoch_exact):
        self.mmt = (
            1
            - (1 - self.cfg.CONTRASTIVE.MOMENTUM)
            * (
                math.cos(math.pi * epoch_exact / self.cfg.SOLVER.MAX_EPOCH)
                + 1.0
            )
            * 0.5
        )

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, extra_keys=None):
        ptr = int(self.ptr.item())
        if (
            not self.cfg.CONTRASTIVE.MOCO_MULTI_VIEW_QUEUE
        ):  # TODO: add multiview negatives
            keys_queue_update = [keys[0]]
        else:
            assert (
                len(keys) > 0
            ), "need to have multiple views for adding them to queue"
            keys_queue_update = []
            keys_queue_update += keys
            if extra_keys:
                keys_queue_update += [
                    item for sublist in extra_keys for item in sublist
                ]
        for key in keys_queue_update:
            # write the current feat into queue, at pointer
            num_items = int(key.size(0))

            assert self.k % num_items == 0
            assert ptr + num_items <= self.k
            self.queue_x[ptr : ptr + num_items, :] = key
            # move pointer
            ptr += num_items
            # reset pointer
            if ptr == self.k:
                ptr = 0
            self.ptr[0] = ptr

    @torch.no_grad()
    def batch_clips(self, clips):
        clips_batched = [None] * len(clips[0])
        for i, clip in enumerate(clips):
            for j, view in enumerate(clip):
                if i == 0:
                    clips_batched[j] = view
                else:
                    clips_batched[j] = torch.cat(
                        [clips_batched[j], view], dim=0
                    )
                del view
        return clips_batched

    @torch.no_grad()
    def compute_key_feat(
        self, clips_k, compute_predictor_keys=False, batched_inference=True
    ):
        assert self.training
        # momentum update key encoder
        self._update_history()
        self.iter += 1
        n_clips = len(clips_k)
        bsz = clips_k[0][0].shape[0]
        if n_clips * bsz * clips_k[0][0].numel() > 4 * 64 * 3 * 8 * 224 * 224:
            batched_inference = False  # hack to avoid oom on large inputs
        assert n_clips > 0
        if batched_inference and all(
            [
                clips_k[i][j].shape[1:] == clips_k[0][j].shape[1:]
                for i in range(len(clips_k))
                for j in range(len(clips_k[i]))
            ]
        ):
            clips_k = [self.batch_clips(clips_k)]
            batched = True
        else:
            batched = False

        keys, pred_keys = [], []
        for k in range(0, len(clips_k)):
            clip_k = clips_k[k]
            if self._batch_shuffle_on:
                with torch.no_grad():
                    clip_k, idx_restore = self._batch_shuffle(clip_k)
            with torch.no_grad():
                hist_feat = self.backbone_hist(clip_k)
                if isinstance(hist_feat, list):
                    hist_time = hist_feat[1:]
                    hist_feat = hist_feat[0]
                    if compute_predictor_keys:
                        tks = []
                        for tk in hist_time:
                            tk = self.l2_norm(tk)
                            if self._batch_shuffle_on:
                                tk = self._batch_unshuffle(
                                    tk, idx_restore
                                ).detach()
                            tks.append(tk)
                        pred_keys.append(tks)
                x_hist = self.l2_norm(hist_feat)
                if self._batch_shuffle_on:
                    x_hist = self._batch_unshuffle(x_hist, idx_restore).detach()
            keys.append(x_hist)
        if batched:
            assert len(keys) == 1, "batched input uses single clip"
            batched_key = keys[0]
            if compute_predictor_keys:
                batched_pred_key = pred_keys[0]
            keys, pred_keys = [], []
            for k in range(0, n_clips):
                keys.append(batched_key[k * bsz : (k + 1) * bsz])
                if compute_predictor_keys:
                    pred_keys.append(batched_pred_key[k * bsz : (k + 1) * bsz])
        if compute_predictor_keys:
            return keys, pred_keys
        else:
            return keys

    def forward(
        self, clips, index=None, time=None, epoch_exact=None, keys=None
    ):
        if epoch_exact is not None and self.momentum_annealing:
            self.momentum_anneal_cosine(epoch_exact)

        if self.type == "mem":
            batch_size = clips[0].size(0)
            q = self.backbone(clips)
            if index is None:
                return q
            q = self.l2_norm(q)

            if not self.training:
                assert self.knn_mem.duration == 1
                return self.eval_knn(q)
            time *= self.duration - 1
            clip_ind = torch.randint(
                0,
                self.length,
                size=(
                    batch_size,
                    self.k + 1,
                ),
            ).cuda()
            clip_ind.select(1, 0).copy_(index.data)

            if self.mem_type == "2d":
                if self.interp:
                    time_ind = (
                        torch.empty(batch_size, self.k + 1)
                        .uniform_(0, self.duration - 1)
                        .cuda()
                    )
                else:
                    time_ind = torch.randint(
                        0,
                        self.duration - 1,
                        size=(
                            batch_size,
                            self.k + 1,
                        ),
                    ).cuda()
            else:
                time_ind = torch.zeros(
                    size=(batch_size, self.k + 1), dtype=int
                ).cuda()

            if self.examplar_type == "clip":
                # Diff clip from same video are negative.
                time_ind.select(1, 0).copy_(time.data)
            elif self.examplar_type == "video":
                # Diff clip from same video are positive.
                pass
            else:
                raise NotImplementedError(
                    "unsupported examplar_type {}".format(self.examplar_type)
                )
            k = self.memory.get(clip_ind, time_ind, self.interp)
            # q: N x C, k: N x K x C
            prod = torch.einsum("nc,nkc->nk", q, k)
            prod = torch.div(prod, self.T)

            loss = self.nce_loss_fun(prod)

            self.memory.update(
                q, momentum=self.mmt, ind=index, time=time, interp=self.interp
            )
            self.knn_mem_update(q, index)
            return prod, 0.0, True
        elif self.type == "moco":
            if isinstance(clips[0], list):
                n_clips = len(clips)
                ind_clips = np.arange(
                    n_clips
                )  # clips come ordered temporally from decoder

                clip_q = clips[ind_clips[0]]
                clips_k = [clips[i] for i in ind_clips[1:]]
                # rearange time
                time_q = time[:, ind_clips[0], :]
                time_k = (
                    time[:, ind_clips[1:], :]
                    if keys is None
                    else time[:, ind_clips[0] + 1 :, :]
                )
            else:
                clip_q = clips

            feat_q = self.backbone(clip_q)
            extra_projs = []
            if isinstance(feat_q, list):
                extra_projs = feat_q[1:]
                feat_q = feat_q[0]
                extra_projs = [self.l2_norm(feat) for feat in extra_projs]

            if index is None:
                return feat_q
            q = self.l2_norm(feat_q)
            q_knn = q

            if not self.training:
                return self.eval_knn(q_knn)

            if keys is None:
                keys = self.compute_key_feat(
                    clips_k, compute_predictor_keys=False
                )
                auto_enqueue_keys = True
            else:
                auto_enqueue_keys = False

            # score computation
            queue_neg = torch.einsum(
                "nc,kc->nk", [q, self.queue_x.clone().detach()]
            )

            for k, key in enumerate(keys):
                out_pos = torch.einsum("nc,nc->n", [q, key]).unsqueeze(-1)
                lgt_k = torch.cat([out_pos, queue_neg], dim=1)
                if k == 0:
                    logits = lgt_k
                else:
                    logits = torch.cat([logits, lgt_k], dim=0)

            logits = torch.div(logits, self.T)

            loss = self.nce_loss_fun(logits)
            # update queue
            if self.training and auto_enqueue_keys:
                self._dequeue_and_enqueue(keys)

            self.knn_mem_update(q_knn, index)
            return logits, loss

        elif self.type == "byol":
            clips_key = [None] * len(clips)
            for i, clip in enumerate(clips):
                p = []
                for path in clip:
                    p.append(path)
                clips_key[i] = p
            batch_clips = False
            if isinstance(clips[0], list):
                n_clips = len(clips)
                ind_clips = np.arange(
                    n_clips
                )  # clips come ordered temporally from decoder
                if batch_clips and n_clips > 1:
                    clips_batched = self.batch_clips(clips)
                    clips_key = [clips_batched]
                    clip_q = clips_batched
                else:
                    clip_q = clips[0]
            else:
                clip_q = clips

            feat_q = self.backbone(clip_q)
            predictors = []
            if isinstance(feat_q, list):
                predictors = feat_q[1:]
                feat_q = feat_q[0]
                predictors = [self.l2_norm(feat) for feat in predictors]
            else:
                raise NotImplementedError("BYOL: predictor is missing")
            assert len(predictors) == 1
            if index is None:
                return feat_q
            q = self.l2_norm(feat_q)

            q_knn = q  # projector

            if not self.training:
                return self.eval_knn(q_knn)

            ind_clips = np.arange(
                n_clips
            )  # clips come ordered temporally from decoder

            # rest down is for training
            if keys is None:
                keys = self.compute_key_feat(
                    clips_key, compute_predictor_keys=False
                )

            if self.cfg.CONTRASTIVE.SEQUENTIAL:
                loss_reg = self.sim_loss(predictors[0], keys[0])
                for i in range(1, len(keys)):
                    loss_reg += self.sim_loss(predictors[0], keys[i])
                loss_reg /= len(keys)
            else:
                if batch_clips:
                    bs = predictors[0].shape[0] // 2
                    loss_reg = self.sim_loss(
                        predictors[0][:bs, :], keys[0][bs:, :]
                    ) + self.sim_loss(predictors[0][bs:, :], keys[0][:bs, :])
                    q_knn = q_knn[:bs, :]
                    del clips_batched[0]
                else:
                    loss_q1 = self.sim_loss(predictors[0], keys[1])
                    assert len(clips) == 2
                    clip_q2 = clips[1]
                    feat_q2 = self.backbone(clip_q2)
                    predictors2 = feat_q2[1:]
                    # feat_q2 = feat_q2[0] # not used
                    predictors2 = [self.l2_norm(feat) for feat in predictors2]
                    assert len(predictors2) == 1

                    loss_q2 = self.sim_loss(predictors2[0], keys[0])
                    loss_reg = loss_q1 + loss_q2

            # loss_pos = self.sim_loss(q1_proj, q2_proj)
            dummy_logits = torch.cat(
                (
                    9999.0
                    * torch.ones((len(index), 1), dtype=torch.float).cuda(),
                    torch.zeros((len(index), self.k), dtype=torch.float).cuda(),
                ),
                dim=1,
            )

            self.knn_mem_update(q_knn, index)

            return dummy_logits, loss_reg

        elif self.type == "swav":
            if not isinstance(clips[0], list):
                if self.swav_use_public_code:
                    proj_1, _ = self.run_swav_orig_encoder_q(clips)
                else:
                    proj_1, _ = self.run_swav_encoder_q(clips)
                if index is None:
                    return proj_1
                if not self.training:
                    return self.eval_knn(proj_1)
            n_clips = len(clips)
            ind_clips = np.arange(
                n_clips
            )  # clips come ordered temporally from decoder
            clip_q = clips[0]

            if self.swav_use_public_code:
                # uses official code of SwAV from
                # https://github.com/facebookresearch/swav/blob/master/main_swav.py
                with torch.no_grad():
                    m = self.module if hasattr(self, "module") else self
                    w = m.swav_prototypes.weight.data.clone()
                    w = nn.functional.normalize(w, dim=1, p=2)
                    m.swav_prototypes.weight.copy_(w)

                bs = clips[0][0].size(0)
                output, embedding = [], []

                for i, clip_q in enumerate(clips):
                    x = self.run_swav_orig_encoder_q(clip_q)
                    embedding.append(x[0])
                    output.append(x[1])
                q_knn = embedding[0]
                embedding = torch.cat(embedding, dim=0)
                output = torch.cat(output, dim=0)

                loss_swav = 0
                swav_extra_crops = n_clips - 2
                self.swav_crops_for_assign = np.arange(
                    n_clips - swav_extra_crops
                )
                for i, crop_id in enumerate(self.swav_crops_for_assign):
                    with torch.no_grad():
                        out = output[bs * crop_id : bs * (crop_id + 1)]
                        if (
                            self.cfg.CONTRASTIVE.SWAV_QEUE_LEN > 0
                            and epoch_exact >= 15.0
                        ):
                            if self.swav_use_the_queue or not torch.all(
                                self.queue_swav[i, -1, :] == 0
                            ):
                                self.swav_use_the_queue = True
                                out = torch.cat(
                                    (
                                        torch.mm(
                                            self.queue_swav[i],
                                            m.swav_prototypes.weight.t(),
                                        ),
                                        out,
                                    )
                                )
                            self.queue_swav[i, bs:] = self.queue_swav[
                                i, :-bs
                            ].clone()
                            self.queue_swav[i, :bs] = embedding[
                                crop_id * bs : (crop_id + 1) * bs
                            ]
                        q = out / self.swav_eps_sinkhorn
                        q = torch.exp(q).t()
                        q = (
                            self.distributed_sinkhorn(q, 3)[-bs:]
                            if self.cfg.NUM_SHARDS > 1
                            else self.sinkhorn(q.t(), 3)[-bs:]
                        )
                    subloss = 0
                    for v in np.delete(np.arange(n_clips), crop_id):
                        p = self.softmax(output[bs * v : bs * (v + 1)] / self.T)
                        subloss -= torch.mean(
                            torch.sum(q * torch.log(p), dim=1)
                        )
                    loss_swav += subloss / (n_clips - 1)
                loss_swav /= len(self.swav_crops_for_assign)
            else:
                proj_1, out_1 = self.run_swav_encoder_q(clip_q)
                q_knn = proj_1
                if not self.training:
                    return self.eval_knn(q_knn)
                proj_2, out_2 = self.run_swav_encoder_q(clips[1])
                bs = proj_1.shape[0]
                if self.cfg.CONTRASTIVE.SWAV_QEUE_LEN > 0:
                    if epoch_exact >= 15.0 and not torch.all(
                        self.queue_swav[0, -1, :] == 0
                    ):
                        swav_prototypes = F.normalize(
                            self.swav_prototypes, dim=0, p=2
                        ).detach()
                        out_1 = torch.cat(
                            (
                                torch.mm(
                                    self.queue_swav[0].detach(), swav_prototypes
                                ),
                                out_1,
                            )
                        )
                        out_2 = torch.cat(
                            (
                                torch.mm(
                                    self.queue_swav[1].detach(), swav_prototypes
                                ),
                                out_2,
                            )
                        )
                    # fill the queue
                    self.queue_swav[0, bs:] = self.queue_swav[0, :-bs].clone()
                    self.queue_swav[0, :bs] = proj_1.detach()
                    self.queue_swav[1, bs:] = self.queue_swav[1, :-bs].clone()
                    self.queue_swav[1, :bs] = proj_2.detach()

                with torch.no_grad():
                    code_1 = self.get_code(out_1)
                    code_2 = self.get_code(out_2)
                loss12 = self.KLDivLoss(out_1[-bs:], code_2[-bs:].detach())
                loss21 = self.KLDivLoss(out_2[-bs:], code_1[-bs:].detach())
                loss_swav = loss12 + loss21
            self.knn_mem_update(q_knn, index)
            dummy_logits = torch.cat(
                (
                    9999.0
                    * torch.ones((len(index), 1), dtype=torch.float).cuda(),
                    torch.zeros((len(index), self.k), dtype=torch.float).cuda(),
                ),
                dim=1,
            )
            return dummy_logits, loss_swav

        elif self.type == "simclr":
            if isinstance(clips[0], list):
                n_clips = len(clips)
                clip_q = clips[0]
            else:
                clip_q = clips
            feat_q = self.backbone(clip_q)
            q = self.l2_norm(feat_q)
            if index is None:
                return q
            q_knn = q
            if not self.training:
                return self.eval_knn(q_knn)
            q2 = self.backbone(clips[1])
            q2 = self.l2_norm(q2)
            distributed_loss = False
            if distributed_loss and self.cfg.NUM_GPUS > 1:
                out = torch.cat([q, q2], dim=0)
                if self.cfg.CONTRASTIVE.SIMCLR_DIST_ON:
                    out_all = du.cat_all_gather(out)
                else:
                    out_all = out
                similarity = torch.exp(torch.mm(out, out_all.t()) / self.T)
                Z, loss = 0.0, 0.0
                for loss_id in range(len(self.pos_mask)):
                    pos = torch.sum(similarity * self.pos_mask[loss_id], 1)
                    neg = torch.sum(similarity * self.neg_mask, 1)
                    idx = (
                        1 - torch.sum(self.pos_mask[loss_id], 1) > 0
                    ).detach()
                    term_prob = pos / (pos + neg)
                    term_prob[idx] = 1.0
                    term_loss = torch.log(term_prob)
                    Z += torch.sum(~idx).detach()
                    loss -= torch.sum(term_loss)
                loss /= Z
            else:
                cat_across_gpus = True
                if cat_across_gpus and self.cfg.NUM_GPUS > 1:
                    q = du.AllGatherWithGradient.apply(q)
                    q2 = du.AllGatherWithGradient.apply(q2)
                out = torch.cat([q, q2], dim=0)
                # [2*B, 2*B]
                sim_matrix = torch.exp(
                    torch.mm(out, out.t().contiguous()) / self.T
                )
                # SANITY:
                mask = (
                    torch.ones_like(sim_matrix)
                    - torch.eye(out.shape[0], device=sim_matrix.device)
                ).bool()
                # [2*B, 2*B-1]
                sim_matrix = sim_matrix.masked_select(mask).view(
                    out.shape[0], -1
                )
                # compute loss
                pos_sim = torch.exp(torch.sum(q * q2, dim=-1) / self.T)
                # [2*B]
                pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
                loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
            self.knn_mem_update(q_knn, index)
            dummy_logits = torch.cat(
                (
                    9999.0
                    * torch.ones((len(index), 1), dtype=torch.float).cuda(),
                    torch.zeros((len(index), self.k), dtype=torch.float).cuda(),
                ),
                dim=1,
            )
            return dummy_logits, loss
        else:
            raise NotImplementedError()

    def _simclr_precompute_pos_neg_mask_multi(self):
        # computed once at the beginning of training
        distributed = self.cfg.CONTRASTIVE.SIMCLR_DIST_ON
        if distributed:
            total_images = self.cfg.TRAIN.BATCH_SIZE * self.cfg.NUM_SHARDS
            world_size = du.get_world_size()
            rank = du.get_rank()
        else:
            total_images = self.cfg.TRAIN.BATCH_SIZE
            world_size = du.get_local_size()
            rank = du.get_local_rank()
        local_orig_images = total_images // world_size
        local_crops = local_orig_images * self.num_crops

        pos_temps = []
        for d in np.arange(self.num_crops):
            pos_temp, neg_temp = [], []
            for i in range(world_size):
                if i == rank:
                    pos = np.eye(local_crops, k=d * local_orig_images) + np.eye(
                        local_crops, k=-local_crops + d * local_orig_images
                    )
                    neg = np.ones((local_crops, local_crops))
                else:
                    pos = np.zeros((local_crops, local_crops))
                    neg = np.zeros((local_crops, local_crops))
                pos_temp.append(pos)
                neg_temp.append(neg)
            pos_temps.append(np.hstack(pos_temp))
            neg_temp = np.hstack(neg_temp)

        pos_mask = []
        for i in range(self.num_crops - 1):
            pos_mask.append(torch.from_numpy(pos_temps[1 + i]))
        neg_mask = torch.from_numpy(neg_temp - sum(pos_temps))

        if self.num_gpus:
            for i in range(len(pos_mask)):
                pos_mask[i] = pos_mask[i].cuda(non_blocking=True)
            neg_mask = neg_mask.cuda(non_blocking=True)
        self.pos_mask, self.neg_mask = pos_mask, neg_mask

    def run_swav_encoder_q(self, im):
        proj = self.backbone(im)  # Nx512, Nx128
        proj = F.normalize(proj, dim=1)  # always normalize
        swav_prototypes = F.normalize(self.swav_prototypes, dim=0, p=2)
        out = proj @ swav_prototypes
        return proj, out

    @torch.no_grad()
    def get_code(self, out):
        with torch.no_grad():
            Q = torch.exp(out / self.swav_eps_sinkhorn)  # BxK
            if self.cfg.NUM_SHARDS > 1:
                Q_sink = self.distributed_sinkhorn(Q.t(), 3)  # BxK
            else:
                Q_sink = self.sinkhorn(Q, 3)  # BxK
        return Q_sink

    def run_swav_orig_encoder_q(self, x):
        x = self.backbone(x)  # Nx512, Nx128
        x = nn.functional.normalize(x, dim=1, p=2)
        if self.swav_prototypes is not None:
            return x, self.swav_prototypes(x)
        return x

    @torch.no_grad()
    def sinkhorn(self, Q, iters):
        with torch.no_grad():
            Q = Q.t()  # KxB
            sum_Q = torch.sum(Q)
            Q /= sum_Q

            r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
            c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / Q.shape[1]

            for _ in range(iters):
                Q *= (r / torch.sum(Q, dim=1)).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)

            Q = Q / torch.sum(Q, dim=0, keepdim=True)
            return Q.t().float()

    def distributed_sinkhorn(self, Q, nmb_iters):
        with torch.no_grad():
            sum_Q = torch.sum(Q)
            du.all_reduce([sum_Q], average=False)
            Q /= sum_Q

            u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
            r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
            c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (
                du.get_world_size() * Q.shape[1]
            )

            curr_sum = torch.sum(Q, dim=1)
            du.all_reduce([curr_sum], average=False)

            for _ in range(nmb_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1)
                du.all_reduce([curr_sum], average=False)
            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def KLDivLoss(self, out, code):
        softmax = nn.Softmax(dim=1).cuda()
        p = softmax(out / self.T)
        loss = torch.mean(-torch.sum(code * torch.log(p), dim=1))
        return loss


def l2_loss(x, y):
    return 2 - 2 * (x * y).sum(dim=-1)


class Normalize(nn.Module):
    def __init__(self, power=2, dim=1):
        super(Normalize, self).__init__()
        self.dim = dim
        self.power = power

    def forward(self, x):
        norm = (
            x.pow(self.power).sum(self.dim, keepdim=True).pow(1.0 / self.power)
        )
        out = x.div(norm)
        return out


class Memory(nn.Module):
    def __init__(self, length, duration, dim, cfg):
        super(Memory, self).__init__()
        self.length = length
        self.duration = duration
        self.dim = dim
        stdv = 1.0 / math.sqrt(dim / 3)
        self.register_buffer(
            "memory",
            torch.rand(length, duration, dim).mul_(2 * stdv).add_(-stdv),
        )
        self.device = self.memory.device
        self.l2_norm = Normalize(dim=1)
        self.l2_norm2d = Normalize(dim=2)
        self.num_gpus = cfg.NUM_GPUS

    def resize(self, length, duration, dim):
        self.length = length
        self.duration = duration
        self.dim = dim
        stdv = 1.0 / math.sqrt(dim / 3)
        del self.memory
        self.memory = (
            torch.rand(length, duration, dim, device=self.device)
            .mul_(2 * stdv)
            .add_(-stdv)
            .cuda()
        )

    def get(self, ind, time, interp=False):
        batch_size = ind.size(0)
        with torch.no_grad():
            if interp:
                # mem_idx = self.memory[ind.view(-1), :, :]
                t0 = time.floor().long()  # - 1
                t0 = torch.clamp(t0, 0, self.memory.shape[1] - 1)
                t1 = t0 + 1
                t1 = torch.clamp(t1, 0, self.memory.shape[1] - 1)

                mem_t0 = self.memory[ind.view(-1), t0.view(-1), :]
                mem_t1 = self.memory[ind.view(-1), t1.view(-1), :]
                w2 = time.view(-1, 1) / self.duration
                w_t1 = (time - t0).view(-1, 1).float()
                w_t1 = 1 - w_t1  # hack for inverse
                selected_mem = mem_t0 * (1 - w_t1) + mem_t1 * w_t1
            else:
                # logger.info("1dmem get ind shape {} time shape {}".format(ind.shape, time.shape))
                selected_mem = self.memory[
                    ind.view(-1), time.long().view(-1), :
                ]

        out = selected_mem.view(batch_size, -1, self.dim)
        return out

    def update(self, mem, momentum, ind, time, interp=False):
        if self.num_gpus > 1:
            mem, ind, time = du.all_gather([mem, ind, time])
        with torch.no_grad():
            if interp:
                t0 = time.floor().long()  # - 1
                t0 = torch.clamp(t0, 0, self.memory.shape[1] - 1)
                t1 = t0 + 1
                t1 = torch.clamp(t1, 0, self.memory.shape[1] - 1)
                mem_t0 = self.memory[ind.view(-1), t0.view(-1), :]
                mem_t1 = self.memory[ind.view(-1), t1.view(-1), :]
                w2 = time.float().view(-1, 1) / float(self.duration)
                w_t1 = (time - t0).view(-1, 1).float()
                w_t1 = 1 - w_t1  # hack for inverse

                w_t0 = 1 - w_t1
                # mem = mem.squeeze()
                duo_update = False
                if duo_update:
                    update_t0 = (
                        mem * w_t0 + mem_t0 * w_t1
                    ) * momentum + mem_t0 * (1 - momentum)
                    update_t1 = (
                        mem * w_t1 + mem_t1 * w_t0
                    ) * momentum + mem_t1 * (1 - momentum)
                else:
                    update_t0 = mem * w_t0 * momentum + mem_t0 * (1 - momentum)
                    update_t1 = mem * w_t1 * momentum + mem_t1 * (1 - momentum)

                update_t0 = self.l2_norm(update_t0)
                update_t1 = self.l2_norm(update_t1)

                self.memory[ind.view(-1), t0.view(-1), :] = update_t0.squeeze()
                self.memory[ind.view(-1), t1.view(-1), :] = update_t1.squeeze()
            else:
                mem = mem.view(mem.size(0), 1, -1)
                mem_old = self.get(ind, time, interp=interp)
                mem_update = mem * momentum + mem_old * (1 - momentum)
                mem_update = self.l2_norm2d(mem_update)
                # logger.info("1dmem set ind shape {} time shape {}".format(ind.shape, time.shape))

                # my version
                self.memory[
                    ind.view(-1), time.long().view(-1), :
                ] = mem_update.squeeze()
                return

    def forward(self, inputs):
        pass


class Memory1D(nn.Module):
    def __init__(self, length, duration, dim, cfg):
        super(Memory1D, self).__init__()
        assert duration == 1
        self.length = length
        self.duration = duration
        self.dim = dim
        stdv = 1.0 / math.sqrt(dim / 3)
        self.register_buffer(
            "memory", torch.rand(length, dim).mul_(2 * stdv).add_(-stdv)
        )
        self.l2_norm = Normalize(dim=1)
        self.num_gpus = cfg.NUM_GPUS

    @torch.no_grad()
    def get(self, ind, time, interp=False):
        batch_size = ind.size(0)
        if len(ind.shape) == 1:
            return torch.index_select(self.memory, 0, ind.view(-1)).view(
                batch_size, self.dim
            )
        else:
            return torch.index_select(self.memory, 0, ind.view(-1)).view(
                batch_size, -1, self.dim
            )

    @torch.no_grad()
    def update(self, mem, momentum, ind, time, interp=False):
        if self.num_gpus > 1:
            mem, ind, time = du.all_gather([mem, ind, time])
        mem = mem.view(mem.size(0), -1)
        ind, time = ind.long(), time.long()

        mem_old = self.get(ind, time, interp=interp)
        mem_update = mem_old * (1 - momentum) + mem * momentum
        mem_update = self.l2_norm(mem_update)

        self.memory.index_copy_(0, ind, mem_update)
        return


def contrastive_parameter_surgery(model, cfg, epoch_exact, cur_iter):

    # cancel some gradients in first epoch of SwAV
    if (
        cfg.MODEL.MODEL_NAME == "ContrastiveModel"
        and cfg.CONTRASTIVE.TYPE == "swav"
        and epoch_exact <= 1.0
    ):
        for name, p in model.named_parameters():
            if "swav_prototypes" in name:
                p.grad = None

    iters_noupdate = 0
    if (
        cfg.MODEL.MODEL_NAME == "ContrastiveModel"
        and cfg.CONTRASTIVE.TYPE == "moco"
    ):
        assert (
            cfg.CONTRASTIVE.QUEUE_LEN % (cfg.TRAIN.BATCH_SIZE * cfg.NUM_SHARDS)
            == 0
        )
        iters_noupdate = (
            cfg.CONTRASTIVE.QUEUE_LEN // cfg.TRAIN.BATCH_SIZE // cfg.NUM_SHARDS
        )

    if cur_iter < iters_noupdate and epoch_exact < 1:  #  for e.g. MoCo
        logger.info(
            "Not updating parameters {}/{}".format(cur_iter, iters_noupdate)
        )
        update_param = False
    else:
        update_param = True

    return model, update_param


def contrastive_forward(model, cfg, inputs, index, time, epoch_exact, scaler):
    if cfg.CONTRASTIVE.SEQUENTIAL:
        perform_backward = False
        mdl = model.module if hasattr(model, "module") else model
        keys = (
            mdl.compute_key_feat(
                inputs,
                compute_predictor_keys=False,
                batched_inference=True if len(inputs) < 2 else False,
            )
            if cfg.CONTRASTIVE.TYPE == "moco" or cfg.CONTRASTIVE.TYPE == "byol"
            else [None] * len(inputs)
        )
        for k, vid in enumerate(inputs):
            other_keys = keys[:k] + keys[k + 1 :]
            time_cur = torch.cat(
                [
                    time[:, k : k + 1, :],
                    time[:, :k, :],
                    time[:, k + 1 :, :],
                ],
                1,
            )  # q, kpre, kpost
            vids = [vid]
            if (
                cfg.CONTRASTIVE.TYPE == "swav"
                or cfg.CONTRASTIVE.TYPE == "simclr"
            ):
                if k < len(inputs) - 1:
                    vids = inputs[k : k + 2]
                else:
                    break
            lgt_k, loss_k = model(
                vids, index, time_cur, epoch_exact, keys=other_keys
            )
            scaler.scale(loss_k).backward()
            if k == 0:
                preds, partial_loss = lgt_k, loss_k.detach()
            else:
                preds = torch.cat([preds, lgt_k], dim=0)
                partial_loss += loss_k.detach()
        partial_loss /= len(inputs) * 2.0  # to have same loss as symm model
        if cfg.CONTRASTIVE.TYPE == "moco":
            mdl._dequeue_and_enqueue(keys)
    else:
        perform_backward = True
        preds, partial_loss = model(inputs, index, time, epoch_exact, keys=None)
    return model, preds, partial_loss, perform_backward
