# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)


def round_width(width, multiplier, min_width=1, divisor=1, verbose=False):
    if not multiplier:
        return width
    width *= multiplier
    min_width = min_width or divisor
    if verbose:
        logger.info(f"min width {min_width}")
        logger.info(f"width {width} divisor {divisor}")
        logger.info(f"other {int(width + divisor / 2) // divisor * divisor}")

    width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
    if width_out < 0.9 * width:
        width_out += divisor
    return int(width_out)


def validate_checkpoint_wrapper_import(checkpoint_wrapper):
    """
    Check if checkpoint_wrapper is imported.
    """
    if checkpoint_wrapper is None:
        raise ImportError("Please install fairscale.")
