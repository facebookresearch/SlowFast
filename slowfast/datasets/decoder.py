#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
import random
import tempfile

import av
import numpy as np
import torch

# TODO: import from torchvision directly after the video reader is steady.
# _TORCHVISION_DECODER_PATH = (
#     "//deeplearning/projects/classy_vision/fb/dataset/video_reader/csrc:"
#     "VideoReader"
# )
# torch.ops.load_library(_TORCHVISION_DECODER_PATH)


def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    frames = torch.index_select(frames, 0, index)
    return frames


def get_start_end_idx(video_size, clip_size, clip_idx, num_clips):
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the video to
    num_clips clips, and select the start and end index of clip_idx-th video
    clip.
    Args:
        video_size (int): number of overall frames.
        clip_size (int): size of the clip to sample from the frames.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the start and end index of the clip_idx-th video
            clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given video for testing.
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
    """
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.uniform(0, delta)
    else:
        # Uniformly sample the clip with the given index.
        start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    return start_idx, end_idx


# def torchvision_decode(
#     video_handle,
#     sampling_rate,
#     num_frames,
#     clip_idx,
#     video_meta,
#     num_clips=10,
#     target_fps=30,
# ):
#     """
#     If video_meta is not empty, perform temporal selective decoding to sample a
#     clip from the video with TorchVision decoder. If video_meta is empty, decode
#     the entire video and update the video_meta.
#     Args:
#         video_handle (bytes): raw bytes of the video file.
#         sampling_rate (int): frame sampling rate (interval between two sampled
#             frames).
#         num_frames (int): number of frames to sample.
#         clip_idx (int): if clip_idx is -1, perform random temporal
#             sampling. If clip_idx is larger than -1, uniformly split the
#             video to num_clips clips, and select the
#             clip_idx-th video clip.
#         video_meta (dict): a dict contains "fps", "timebase", and
#             "max_pts":
#             `fps` is the frames per second of the given video.
#             `timebase` is the video timebase.
#             `max_pts` is the largest pts from the video.
#         num_clips (int): overall number of clips to uniformly
#             sample from the given video.
#         target_fps (int): the input video may has different fps, convert it to
#             the target video fps.
#     Returns:
#         frames (tensor): decoded frames from the video.
#         fps (float): the number of frames per second of the video.
#         decode_all_video (bool): If True, the entire video was decoded.
#     """
#     # Convert the bytes to a tensor.
#     video_tensor = torch.from_numpy(np.frombuffer(video_handle, dtype=np.uint8))
#
#     # Define parameters.
#     seek_frame_margin = 4.0
#     width, height, min_dimension = 0, 0, 0
#
#     if len(video_meta) == 0:
#         # The video_meta is None, decode the entire video.
#         decode_all_video = True
#         video_start_pts, video_end_pts = 0, 0
#         timebase, timebase_den = 0, 1
#     else:
#         # Using the information from video_meta to perform selective
#         # decoding.
#         decode_all_video = False
#         fps = video_meta["fps"]
#         timebase = video_meta["timebase"][0]
#         timebase_den = video_meta["timebase"][1]
#         num_all_frames = video_meta["max_pts"] / timebase_den * fps
#
#         start_idx, end_idx = get_start_end_idx(
#             num_all_frames,
#             sampling_rate * num_frames / target_fps * fps,
#             clip_idx,
#             num_clips
#         )
#
#         # Convert frame index to pts.
#         pts_per_frame = timebase_den / fps
#         video_start_pts = int(start_idx * pts_per_frame)
#         video_end_pts = int(end_idx * pts_per_frame)
#
#     # Decode the raw video with the tv decoder.
#     tv_result = torch.ops.video_reader.read_video_from_memory(
#         video_tensor,
#         seek_frame_margin,
#         0,  # getPtsOnly
#         width,
#         height,
#         min_dimension,
#         video_start_pts,
#         video_end_pts,
#         timebase,
#         timebase_den,
#         0,  # samples,
#         0,  # channels,
#         0,  # audio_start_pts,
#         0,  # audio_end_pts,
#         0,  # audio_timebase_num,
#         0,  # audio_timebase_den,
#     )
#     frames, frame_pts, timebase, fps, _, _, _ = tv_result
#
#     # Prepare video meta information.
#     if len(video_meta) == 0:
#         video_meta["max_pts"] = frame_pts.numpy().tolist()[-1]
#         video_meta["timebase"] = timebase.numpy().tolist()
#         video_meta["fps"] = fps.numpy()[0]
#     return frames, video_meta["fps"], decode_all_video


def pyav_decode_stream(
    container, start_pts, end_pts, stream, stream_name, buffer_size=0
):
    """
    Decode the video with PyAV decoder.
    Args:
        container (container): PyAV container.
        start_pts (int): the starting Presentation TimeStamp to fetch the
            video frames.
        end_pts (int): the ending Presentation TimeStamp of the decoded frames.
        stream (stream): PyAV stream.
        stream_name (dict): a dictionary of streams. For example, {"video": 0}
            means video stream at stream index 0.
        buffer_size (int): number of additional frames to decode beyond end_pts.
    Returns:
        result (list): list of frames decoded.
        max_pts (int): max Presentation TimeStamp of the video sequence.
    """
    # Seeking in the stream is imprecise. Thus, seek to an ealier PTS by a
    # margin pts.
    margin = 1024
    seek_offset = max(start_pts - margin, 0)

    container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    frames = {}
    buffer_count = 0
    max_pts = 0
    for frame in container.decode(**stream_name):
        max_pts = max(max_pts, frame.pts)
        if frame.pts < start_pts:
            continue
        if frame.pts <= end_pts:
            frames[frame.pts] = frame
        else:
            buffer_count += 1
            frames[frame.pts] = frame
            if buffer_count >= buffer_size:
                break
    result = [frames[pts] for pts in sorted(frames)]
    return result, max_pts


def pyav_decode(
    video_handle,
    sampling_rate,
    num_frames,
    clip_idx,
    num_clips=10,
    target_fps=30,
):
    """
    Convert the video from its original fps to the target_fps. If the video
    support selective decoding (contain decoding information in the video head),
    the perform temporal selective decoding and sample a clip from the video
    with the PyAV decoder. If the video does not support selective decoding,
    decode the entire video.

    Args:
        video_handle (str): path to the raw video file.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames.
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the clip_idx-th video clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given video.
        target_fps (int): the input video may has different fps, convert it to
            the target video fps before frame sampling.
    Returns:
        frames (tensor): decoded frames from the video. Return None if the no
            video stream was found.
        fps (float): the number of frames per second of the video.
        decode_all_video (bool): If True, the entire video was decoded.
    """
    # Open video file with PyAV decoder.
    container = av.open(video_handle)
    # Try to fetch the decoding information from the video head. Some of the
    # videos does not support fetching the decoding information, for that case
    # it will get None duration.
    fps = float(container.streams.video[0].average_rate)
    frames_length = container.streams.video[0].frames
    duration = container.streams.video[0].duration

    if duration is None:
        # If failed to fetch the decoding information, decode the entire video.
        decode_all_video = True
        video_start_pts, video_end_pts = 0, math.inf
    else:
        # Perform selective decoding.
        decode_all_video = False
        start_idx, end_idx = get_start_end_idx(
            frames_length,
            sampling_rate * num_frames / target_fps * fps,
            clip_idx,
            num_clips,
        )
        timebase = duration / frames_length
        video_start_pts = int(start_idx * timebase)
        video_end_pts = int(end_idx * timebase)

    frames = None
    # If video stream was found, fetch video frames from the video.
    if container.streams.video:
        video_frames, max_pts = pyav_decode_stream(
            container,
            video_start_pts,
            video_end_pts,
            container.streams.video[0],
            {"video": 0},
        )
        container.close()

        frames = [frame.to_rgb().to_ndarray() for frame in video_frames]
        frames = torch.as_tensor(np.stack(frames))
    return frames, fps, decode_all_video


def decode(
    raw_video,
    sampling_rate,
    num_frames,
    clip_idx=-1,
    num_clips=10,
    video_meta=None,
    py_av=True,
    target_fps=30,
):
    """
    Decode the video and perform temporal sampling.
    Args:
        raw_video (BytesIO): raw video bytes.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames).
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal
            sampling. If clip_idx is larger than -1, uniformly split the
            video to num_clips clips, and select the
            clip_idx-th video clip.
        num_clips (int): overall number of clips to uniformly
            sample from the given video.
        video_meta (dict): a dict contains "fps", "timebase", and
            "max_pts":
            `fps` is the frames per second of the given video.
            `timebase` is the video timebase.
            `max_pts` is the largest pts from the video.
        py_av (bool): if True, decode with the PyAV decoder, else decode with
            the TorchVision decoder.
        target_fps (int): the input video may have different fps, convert it to
            the target video fps before frame sampling.
    Returns:
        frames (tensor): decoded frames from the video.
    """
    # Currently support two decoders: 1) PyAV, and 2) TorchVision.
    assert clip_idx >= -1, "Not valied clip_idx {}".format(clip_idx)

    raw_video = raw_video.getvalue()
    if py_av:
        with tempfile.NamedTemporaryFile(delete=True) as f:
            f.write(raw_video)
            f.flush()
            video_handle = f.name
            try:
                frames, fps, decode_all_video = pyav_decode(
                    video_handle,
                    sampling_rate,
                    num_frames,
                    clip_idx,
                    num_clips,
                    target_fps,
                )
            except Exception as e:
                print("Failed to decode with pyav with exception: {}".format(e))
                return None
    # else:
    #     try:
    #         frames, fps, decode_all_video = torchvision_decode(
    #             raw_video,
    #             sampling_rate,
    #             num_frames,
    #             clip_idx,
    #             video_meta,
    #             num_clips,
    #             target_fps,
    #         )
    #     except Exception as e:
    #         print("Failed to decode by torchvision with exception: {}".format(e))
    #         return None

    # Return None if the frames was not decoded successfully.
    if frames is None:
        return frames

    start_idx, end_idx = get_start_end_idx(
        frames.shape[0],
        num_frames * sampling_rate * fps / target_fps,
        clip_idx if decode_all_video else 0,
        num_clips if decode_all_video else 1,
    )
    # Perform temporal sampling from the decoded video.
    frames = temporal_sampling(frames, start_idx, end_idx, num_frames)
    return frames
