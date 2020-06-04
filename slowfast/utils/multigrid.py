#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Helper functions for multigrid training."""

import numpy as np

import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)


class MultigridSchedule(object):
    """
    This class defines multigrid training schedule and update cfg accordingly.
    """

    def init_multigrid(self, cfg):
        """
        Update cfg based on multigrid settings.
        Args:
            cfg (configs): configs that contains training and multigrid specific
                hyperparameters. Details can be seen in
                slowfast/config/defaults.py.
        Returns:
            cfg (configs): the updated cfg.
        """
        self.schedule = None
        # We may modify cfg.TRAIN.BATCH_SIZE, cfg.DATA.NUM_FRAMES, and
        # cfg.DATA.TRAIN_CROP_SIZE during training, so we store their original
        # value in cfg and use them as global variables.
        cfg.MULTIGRID.DEFAULT_B = cfg.TRAIN.BATCH_SIZE
        cfg.MULTIGRID.DEFAULT_T = cfg.DATA.NUM_FRAMES
        cfg.MULTIGRID.DEFAULT_S = cfg.DATA.TRAIN_CROP_SIZE

        if cfg.MULTIGRID.LONG_CYCLE:
            self.schedule = self.get_long_cycle_schedule(cfg)
            cfg.SOLVER.STEPS = [0] + [s[-1] for s in self.schedule]
            # Fine-tuning phase.
            cfg.SOLVER.STEPS[-1] = (
                cfg.SOLVER.STEPS[-2] + cfg.SOLVER.STEPS[-1]
            ) // 2
            cfg.SOLVER.LRS = [
                cfg.SOLVER.GAMMA ** s[0] * s[1][0] for s in self.schedule
            ]
            # Fine-tuning phase.
            cfg.SOLVER.LRS = cfg.SOLVER.LRS[:-1] + [
                cfg.SOLVER.LRS[-2],
                cfg.SOLVER.LRS[-1],
            ]

            cfg.SOLVER.MAX_EPOCH = self.schedule[-1][-1]

        elif cfg.MULTIGRID.SHORT_CYCLE:
            cfg.SOLVER.STEPS = [
                int(s * cfg.MULTIGRID.EPOCH_FACTOR) for s in cfg.SOLVER.STEPS
            ]
            cfg.SOLVER.MAX_EPOCH = int(
                cfg.SOLVER.MAX_EPOCH * cfg.MULTIGRID.EPOCH_FACTOR
            )
        return cfg

    def update_long_cycle(self, cfg, cur_epoch):
        """
        Before every epoch, check if long cycle shape should change. If it
            should, update cfg accordingly.
        Args:
            cfg (configs): configs that contains training and multigrid specific
                hyperparameters. Details can be seen in
                slowfast/config/defaults.py.
            cur_epoch (int): current epoch index.
        Returns:
            cfg (configs): the updated cfg.
            changed (bool): do we change long cycle shape at this epoch?
        """
        base_b, base_t, base_s = get_current_long_cycle_shape(
            self.schedule, cur_epoch
        )
        if base_s != cfg.DATA.TRAIN_CROP_SIZE or base_t != cfg.DATA.NUM_FRAMES:

            cfg.DATA.NUM_FRAMES = base_t
            cfg.DATA.TRAIN_CROP_SIZE = base_s
            cfg.TRAIN.BATCH_SIZE = base_b * cfg.MULTIGRID.DEFAULT_B

            bs_factor = (
                float(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS)
                / cfg.MULTIGRID.BN_BASE_SIZE
            )

            if bs_factor < 1:
                cfg.BN.NORM_TYPE = "sync_batchnorm"
                cfg.BN.NUM_SYNC_DEVICES = int(1.0 / bs_factor)
            elif bs_factor > 1:
                cfg.BN.NORM_TYPE = "sub_batchnorm"
                cfg.BN.NUM_SPLITS = int(bs_factor)
            else:
                cfg.BN.NORM_TYPE = "batchnorm"

            cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE = cfg.DATA.SAMPLING_RATE * (
                cfg.MULTIGRID.DEFAULT_T // cfg.DATA.NUM_FRAMES
            )
            logger.info("Long cycle updates:")
            logger.info("\tBN.NORM_TYPE: {}".format(cfg.BN.NORM_TYPE))
            if cfg.BN.NORM_TYPE == "sync_batchnorm":
                logger.info(
                    "\tBN.NUM_SYNC_DEVICES: {}".format(cfg.BN.NUM_SYNC_DEVICES)
                )
            elif cfg.BN.NORM_TYPE == "sub_batchnorm":
                logger.info("\tBN.NUM_SPLITS: {}".format(cfg.BN.NUM_SPLITS))
            logger.info("\tTRAIN.BATCH_SIZE: {}".format(cfg.TRAIN.BATCH_SIZE))
            logger.info(
                "\tDATA.NUM_FRAMES x LONG_CYCLE_SAMPLING_RATE: {}x{}".format(
                    cfg.DATA.NUM_FRAMES, cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE
                )
            )
            logger.info(
                "\tDATA.TRAIN_CROP_SIZE: {}".format(cfg.DATA.TRAIN_CROP_SIZE)
            )
            return cfg, True
        else:
            return cfg, False

    def get_long_cycle_schedule(self, cfg):
        """
        Based on multigrid hyperparameters, define the schedule of a long cycle.
        Args:
            cfg (configs): configs that contains training and multigrid specific
                hyperparameters. Details can be seen in
                slowfast/config/defaults.py.
        Returns:
            schedule (list): Specifies a list long cycle base shapes and their
                corresponding training epochs.
        """

        steps = cfg.SOLVER.STEPS

        default_size = float(
            cfg.DATA.NUM_FRAMES * cfg.DATA.TRAIN_CROP_SIZE ** 2
        )
        default_iters = steps[-1]

        # Get shapes and average batch size for each long cycle shape.
        avg_bs = []
        all_shapes = []
        for t_factor, s_factor in cfg.MULTIGRID.LONG_CYCLE_FACTORS:
            base_t = int(round(cfg.DATA.NUM_FRAMES * t_factor))
            base_s = int(round(cfg.DATA.TRAIN_CROP_SIZE * s_factor))
            if cfg.MULTIGRID.SHORT_CYCLE:
                shapes = [
                    [
                        base_t,
                        cfg.MULTIGRID.DEFAULT_S
                        * cfg.MULTIGRID.SHORT_CYCLE_FACTORS[0],
                    ],
                    [
                        base_t,
                        cfg.MULTIGRID.DEFAULT_S
                        * cfg.MULTIGRID.SHORT_CYCLE_FACTORS[1],
                    ],
                    [base_t, base_s],
                ]
            else:
                shapes = [[base_t, base_s]]

            # (T, S) -> (B, T, S)
            shapes = [
                [int(round(default_size / (s[0] * s[1] * s[1]))), s[0], s[1]]
                for s in shapes
            ]
            avg_bs.append(np.mean([s[0] for s in shapes]))
            all_shapes.append(shapes)

        # Get schedule regardless of cfg.MULTIGRID.EPOCH_FACTOR.
        total_iters = 0
        schedule = []
        for step_index in range(len(steps) - 1):
            step_epochs = steps[step_index + 1] - steps[step_index]

            for long_cycle_index, shapes in enumerate(all_shapes):
                cur_epochs = (
                    step_epochs * avg_bs[long_cycle_index] / sum(avg_bs)
                )

                cur_iters = cur_epochs / avg_bs[long_cycle_index]
                total_iters += cur_iters
                schedule.append((step_index, shapes[-1], cur_epochs))

        iter_saving = default_iters / total_iters

        final_step_epochs = cfg.SOLVER.MAX_EPOCH - steps[-1]

        # We define the fine-tuning phase to have the same amount of iteration
        # saving as the rest of the training.
        ft_epochs = final_step_epochs / iter_saving * avg_bs[-1]

        schedule.append((step_index + 1, all_shapes[-1][2], ft_epochs))

        # Obtrain final schedule given desired cfg.MULTIGRID.EPOCH_FACTOR.
        x = (
            cfg.SOLVER.MAX_EPOCH
            * cfg.MULTIGRID.EPOCH_FACTOR
            / sum(s[-1] for s in schedule)
        )

        final_schedule = []
        total_epochs = 0
        for s in schedule:
            epochs = s[2] * x
            total_epochs += epochs
            final_schedule.append((s[0], s[1], int(round(total_epochs))))
        print_schedule(final_schedule)
        return final_schedule


def print_schedule(schedule):
    """
    Log schedule.
    """
    logger.info("Long cycle index\tBase shape\tEpochs")
    for s in schedule:
        logger.info("{}\t{}\t{}".format(s[0], s[1], s[2]))


def get_current_long_cycle_shape(schedule, epoch):
    """
    Given a schedule and epoch index, return the long cycle base shape.
    Args:
        schedule (configs): configs that contains training and multigrid specific
            hyperparameters. Details can be seen in
            slowfast/config/defaults.py.
        cur_epoch (int): current epoch index.
    Returns:
        shapes (list): A list describing the base shape in a long cycle:
            [batch size relative to default,
            number of frames, spatial dimension].
    """
    for s in schedule:
        if epoch < s[-1]:
            return s[1]
    return schedule[-1][1]
