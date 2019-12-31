#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import av


def get_video_container(path_to_vid, multi_thread_decode=False):
    """
    Given the path to the video, return the pyav video container.
    Args:
        path_to_vid (str): path to the video.
        multi_thread_decode (bool): if True, perform multi-thread decoding.
    Returns:
        container (container): pyav video container.
    """
    container = av.open(path_to_vid)
    if multi_thread_decode:
        # Enable multiple threads for decoding.
        container.streams.video[0].thread_type = "AUTO"
    return container
