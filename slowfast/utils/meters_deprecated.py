# TODO: refactor the following parts.
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import datetime
from collections import deque

import numpy as np
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
from slowfast.utils.timer_deprecated import Timer


class ScalarMeter(object):
    """Measures a scalar value (adapted from Detectron)."""

    def __init__(self, window_size):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        return np.median(self.deque)

    def get_win_avg(self):
        return np.mean(self.deque)

    def get_global_avg(self):
        return self.total / self.count


class TrainMeter(object):
    """Measures training stats."""

    def __init__(self, epoch_iters, cfg):
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        # Current minibatch errors (smoothed over a window)
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Number of misclassified examples
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0

    def reset(self, timer=False):
        if timer:
            self.iter_timer.reset()
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, top1_err, top5_err, loss, lr, mb_size):
        # Current minibatch stats
        self.mb_top1_err.add_value(top1_err)
        self.mb_top5_err.add_value(top5_err)
        self.loss.add_value(loss)
        self.lr = lr
        # Aggregate stats
        self.num_top1_mis += top1_err * mb_size
        self.num_top5_mis += top5_err * mb_size
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

    def get_iter_stats(self, cur_epoch, cur_iter):
        eta_sec = self.iter_timer.average_time * (
            self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        mem_usage = misc.gpu_mem_usage()
        stats = dict(
            _type="train_iter",
            epoch="{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            iter="{}/{}".format(cur_iter + 1, self.epoch_iters),
            time_avg=self.iter_timer.average_time,
            time_diff=self.iter_timer.diff,
            eta=eta,
            top1_err=self.mb_top1_err.get_win_median(),
            top5_err=self.mb_top5_err.get_win_median(),
            loss=self.loss.get_win_median(),
            lr=self.lr,
            mem=int(np.ceil(mem_usage)),
        )
        return stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        stats = self.get_iter_stats(cur_epoch, cur_iter)
        logging.log_json_stats(stats)

    def get_epoch_stats(self, cur_epoch):
        eta_sec = self.iter_timer.average_time * (
            self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        mem_usage = misc.gpu_mem_usage()
        top1_err = self.num_top1_mis / self.num_samples
        top5_err = self.num_top5_mis / self.num_samples
        avg_loss = self.loss_total / self.num_samples
        stats = dict(
            _type="train_epoch",
            epoch="{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            time_avg=self.iter_timer.average_time,
            eta=eta,
            top1_err=top1_err,
            top5_err=top5_err,
            loss=avg_loss,
            lr=self.lr,
            mem=int(np.ceil(mem_usage)),
        )
        return stats

    def log_epoch_stats(self, cur_epoch):
        stats = self.get_epoch_stats(cur_epoch)
        logging.log_json_stats(stats)


class ValMeter(object):
    """Measures evaluting stats."""

    def __init__(self, MAX_EPOCH, cfg):
        self._cfg = cfg
        self.MAX_EPOCH = MAX_EPOCH
        self.iter_timer = Timer()
        # Current minibatch errors (smoothed over a window)
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Min errors (over the full val set)
        self.min_top1_err = 100.0
        self.min_top5_err = 100.0
        # Number of misclassified examples
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0

    def reset(self, min_errs=False):
        if min_errs:
            self.min_top1_err = 100.0
            self.min_top5_err = 100.0
        self.iter_timer.reset()
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0

    def iter_tic(self):
        self.iter_timer.tic()

    def iter_toc(self):
        self.iter_timer.toc()

    def update_stats(self, top1_err, top5_err, mb_size):
        self.mb_top1_err.add_value(top1_err)
        self.mb_top5_err.add_value(top5_err)
        self.num_top1_mis += top1_err * mb_size
        self.num_top5_mis += top5_err * mb_size
        self.num_samples += mb_size

    def get_iter_stats(self, cur_epoch, cur_iter):
        eta_sec = self.iter_timer.average_time * (self.MAX_EPOCH - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        mem_usage = misc.gpu_mem_usage()
        iter_stats = dict(
            _type="val_iter",
            epoch="{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            iter="{}/{}".format(cur_iter + 1, self.MAX_EPOCH),
            time_avg=self.iter_timer.average_time,
            time_diff=self.iter_timer.diff,
            eta=eta,
            top1_err=self.mb_top1_err.get_win_median(),
            top5_err=self.mb_top5_err.get_win_median(),
            mem=int(np.ceil(mem_usage)),
        )
        return iter_stats

    def log_iter_stats(self, cur_epoch, cur_iter):
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        stats = self.get_iter_stats(cur_epoch, cur_iter)
        logging.log_json_stats(stats)

    def get_epoch_stats(self, cur_epoch):
        top1_err = self.num_top1_mis / self.num_samples
        top5_err = self.num_top5_mis / self.num_samples
        self.min_top1_err = min(self.min_top1_err, top1_err)
        self.min_top5_err = min(self.min_top5_err, top5_err)
        mem_usage = misc.gpu_mem_usage()
        stats = dict(
            _type="val_epoch",
            epoch="{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            time_avg=self.iter_timer.average_time,
            top1_err=top1_err,
            top5_err=top5_err,
            min_top1_err=self.min_top1_err,
            min_top5_err=self.min_top5_err,
            mem=int(np.ceil(mem_usage)),
        )
        return stats

    def log_epoch_stats(self, cur_epoch):
        stats = self.get_epoch_stats(cur_epoch)
        logging.log_json_stats(stats)
