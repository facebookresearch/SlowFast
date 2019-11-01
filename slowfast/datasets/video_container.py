#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import av


def get_video_container(path_to_vid):
    """
    Given the path to the video, return the pyav video container.
    Args:
        path_to_vid (str): patth to the video.
    Returns:
        container (container): pyav video container.
    """
    container = av.open(path_to_vid)
    return container
