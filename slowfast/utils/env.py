#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Set up Environment."""

import slowfast.utils.logging as logging


def setup_environment():
    # Setup logging format.
    logging.setup_logging()
