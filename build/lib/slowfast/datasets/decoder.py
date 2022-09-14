#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import logging
import math
import numpy as np
import random
import torch
import torchvision.io as io

from . import transform as transform

logger = logging.getLogger(__name__)


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


def get_start_end_idx(
    video_size, clip_size, clip_idx, num_clips_uniform, use_offset=False
):
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the video to
    num_clips_uniform clips, and select the start and end index of clip_idx-th video
    clip.
    Args:
        video_size (int): number of overall frames.
        clip_size (int): size of the clip to sample from the frames.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips_uniform
            clips, and select the start and end index of the clip_idx-th video
            clip.
        num_clips_uniform (int): overall number of clips to uniformly sample from the
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
        if use_offset:
            if num_clips_uniform == 1:
                # Take the center clip if num_clips_uniform is 1.
                start_idx = math.floor(delta / 2)
            else:
                # Uniformly sample the clip with the given index.
                start_idx = clip_idx * math.floor(
                    delta / (num_clips_uniform - 1)
                )
        else:
            # Uniformly sample the clip with the given index.
            start_idx = delta * clip_idx / num_clips_uniform
    end_idx = start_idx + clip_size - 1

    return start_idx, end_idx, start_idx / delta if delta != 0 else 0.0


def get_multiple_start_end_idx(
    video_size,
    clip_sizes,
    clip_idx,
    num_clips_uniform,
    min_delta=0,
    max_delta=math.inf,
    use_offset=False,
):
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the video to
    num_clips_uniform clips, and select the start and end index of clip_idx-th video
    clip.
    Args:
        video_size (int): number of overall frames.
        clip_sizes (list): size of the clip to sample from the frames.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips_uniform
            clips, and select the start and end index of the clip_idx-th video
            clip.
        num_clips_uniform (int): overall number of clips to uniformly sample from the
            given video for testing.
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
    """

    def sample_clips(
        video_size,
        clip_sizes,
        clip_idx,
        num_clips_uniform,
        min_delta=0,
        max_delta=math.inf,
        num_retries=100,
        use_offset=False,
    ):
        se_inds = np.empty((0, 2))
        dt = np.empty((0))
        for clip_size in clip_sizes:
            for i_try in range(num_retries):
                # clip_size = int(clip_size)
                max_start = max(video_size - clip_size, 0)
                if clip_idx == -1:
                    # Random temporal sampling.
                    start_idx = random.uniform(0, max_start)
                else:  # Uniformly sample the clip with the given index.
                    if use_offset:
                        if num_clips_uniform == 1:
                            # Take the center clip if num_clips is 1.
                            start_idx = math.floor(max_start / 2)
                        else:
                            start_idx = clip_idx * math.floor(
                                max_start / (num_clips_uniform - 1)
                            )
                    else:
                        start_idx = max_start * clip_idx / num_clips_uniform

                end_idx = start_idx + clip_size - 1

                se_inds_new = np.append(se_inds, [[start_idx, end_idx]], axis=0)
                if se_inds.shape[0] < 1:
                    se_inds = se_inds_new
                    break

                se_inds_new = np.sort(se_inds_new, 0)
                t_start, t_end = se_inds_new[:, 0], se_inds_new[:, 1]
                dt = t_start[1:] - t_end[:-1]
                if (
                    any(dt < min_delta) or any(dt > max_delta)
                ) and i_try < num_retries - 1:
                    continue  # there is overlap
                else:
                    se_inds = se_inds_new
                    break
        return se_inds, dt

    num_retries, goodness = 100, -math.inf
    for _ in range(num_retries):
        se_inds, dt = sample_clips(
            video_size,
            clip_sizes,
            clip_idx,
            num_clips_uniform,
            min_delta,
            max_delta,
            100,
            use_offset,
        )
        success = not (any(dt < min_delta) or any(dt > max_delta))
        if success or clip_idx != -1:
            se_final, dt_final = se_inds, dt
            break
        else:
            cur_goodness = np.r_[dt[dt < min_delta], -dt[dt > max_delta]].sum()
            if goodness < cur_goodness:
                se_final, dt_final = se_inds, dt
                goodness = cur_goodness

    delta_clips = np.concatenate((np.array([0]), dt_final))
    start_end_delta_time = np.c_[se_final, delta_clips]

    return start_end_delta_time


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


def torchvision_decode(
    video_handle,
    sampling_rate,
    num_frames,
    clip_idx,
    video_meta,
    num_clips_uniform=10,
    target_fps=30,
    modalities=("visual",),
    max_spatial_scale=0,
    use_offset=False,
    min_delta=-math.inf,
    max_delta=math.inf,
):
    """
    If video_meta is not empty, perform temporal selective decoding to sample a
    clip from the video with TorchVision decoder. If video_meta is empty, decode
    the entire video and update the video_meta.
    Args:
        video_handle (bytes): raw bytes of the video file.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames).
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal
            sampling. If clip_idx is larger than -1, uniformly split the
            video to num_clips_uniform clips, and select the clip_idx-th video clip.
        video_meta (dict): a dict contains VideoMetaData. Details can be found
            at `pytorch/vision/torchvision/io/_video_opt.py`.
        num_clips_uniform (int): overall number of clips to uniformly sample from the
            given video.
        target_fps (int): the input video may has different fps, convert it to
            the target video fps.
        modalities (tuple): tuple of modalities to decode. Currently only
            support `visual`, planning to support `acoustic` soon.
        max_spatial_scale (int): the resolution of the spatial shorter
            edge size during decoding.
        min_delta (int): minimum distance between clips when sampling multiple.
        max_delta (int): max distance between clips when sampling multiple.
    Returns:
        frames (tensor): decoded frames from the video.
        fps (float): the number of frames per second of the video.
        decode_all_video (bool): if True, the entire video was decoded.
    """
    # Convert the bytes to a tensor.
    video_tensor = torch.from_numpy(np.frombuffer(video_handle, dtype=np.uint8))

    decode_all_video = True
    video_start_pts, video_end_pts = 0, -1
    # The video_meta is empty, fetch the meta data from the raw video.
    if len(video_meta) == 0:
        # Tracking the meta info for selective decoding in the future.
        meta = io._probe_video_from_memory(video_tensor)
        # Using the information from video_meta to perform selective decoding.
        video_meta["video_timebase"] = meta.video_timebase
        video_meta["video_numerator"] = meta.video_timebase.numerator
        video_meta["video_denominator"] = meta.video_timebase.denominator
        video_meta["has_video"] = meta.has_video
        video_meta["video_duration"] = meta.video_duration
        video_meta["video_fps"] = meta.video_fps
        video_meta["audio_timebas"] = meta.audio_timebase
        video_meta["audio_numerator"] = meta.audio_timebase.numerator
        video_meta["audio_denominator"] = meta.audio_timebase.denominator
        video_meta["has_audio"] = meta.has_audio
        video_meta["audio_duration"] = meta.audio_duration
        video_meta["audio_sample_rate"] = meta.audio_sample_rate

    fps = video_meta["video_fps"]

    if len(video_meta) > 0 and (
        video_meta["has_video"]
        and video_meta["video_denominator"] > 0
        and video_meta["video_duration"] > 0
        and fps * video_meta["video_duration"]
        > sum(T * tau for T, tau in zip(num_frames, sampling_rate))
    ):
        decode_all_video = False  # try selective decoding

        clip_sizes = [
            np.maximum(1.0, sampling_rate[i] * num_frames[i] / target_fps * fps)
            for i in range(len(sampling_rate))
        ]
        start_end_delta_time = get_multiple_start_end_idx(
            fps * video_meta["video_duration"],
            clip_sizes,
            clip_idx,
            num_clips_uniform,
            min_delta=min_delta,
            max_delta=max_delta,
            use_offset=use_offset,
        )
        frames_out = [None] * len(num_frames)
        for k in range(len(num_frames)):
            pts_per_frame = (
                video_meta["video_denominator"] / video_meta["video_fps"]
            )
            video_start_pts = int(start_end_delta_time[k, 0] * pts_per_frame)
            video_end_pts = int(start_end_delta_time[k, 1] * pts_per_frame)

            # Decode the raw video with the tv decoder.
            v_frames, _ = io._read_video_from_memory(
                video_tensor,
                seek_frame_margin=1.0,
                read_video_stream="visual" in modalities,
                video_width=0,
                video_height=0,
                video_min_dimension=max_spatial_scale,
                video_pts_range=(video_start_pts, video_end_pts),
                video_timebase_numerator=video_meta["video_numerator"],
                video_timebase_denominator=video_meta["video_denominator"],
                read_audio_stream=0,
            )
            if v_frames is None or v_frames.shape == torch.Size([0]):
                decode_all_video = True
                logger.info("TV decode FAILED try decode all")
                break
            frames_out[k] = v_frames

    if decode_all_video:
        # failed selective decoding
        decode_all_video = True
        video_start_pts, video_end_pts = 0, -1
        start_end_delta_time = None
        v_frames, _ = io._read_video_from_memory(
            video_tensor,
            seek_frame_margin=1.0,
            read_video_stream="visual" in modalities,
            video_width=0,
            video_height=0,
            video_min_dimension=max_spatial_scale,
            video_pts_range=(video_start_pts, video_end_pts),
            video_timebase_numerator=video_meta["video_numerator"],
            video_timebase_denominator=video_meta["video_denominator"],
            read_audio_stream=0,
        )
        if v_frames.shape == torch.Size([0]):
            v_frames = None
            logger.info("TV decode FAILED try cecode all")

        frames_out = [v_frames]

    if any([t.shape[0] < 0 for t in frames_out]):
        frames_out = [None]
        logger.info("TV decode FAILED: Decoded empty video")

    return frames_out, fps, decode_all_video, start_end_delta_time


def pyav_decode(
    container,
    sampling_rate,
    num_frames,
    clip_idx,
    num_clips_uniform=10,
    target_fps=30,
    use_offset=False,
):
    """
    Convert the video from its original fps to the target_fps. If the video
    support selective decoding (contain decoding information in the video head),
    the perform temporal selective decoding and sample a clip from the video
    with the PyAV decoder. If the video does not support selective decoding,
    decode the entire video.

    Args:
        container (container): pyav container.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames.
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips_uniform
            clips, and select the clip_idx-th video clip.
        num_clips_uniform (int): overall number of clips to uniformly sample from the
            given video.
        target_fps (int): the input video may has different fps, convert it to
            the target video fps before frame sampling.
    Returns:
        frames (tensor): decoded frames from the video. Return None if the no
            video stream was found.
        fps (float): the number of frames per second of the video.
        decode_all_video (bool): If True, the entire video was decoded.
    """
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
        clip_size = np.maximum(
            1.0, np.ceil(sampling_rate * (num_frames - 1) / target_fps * fps)
        )
        start_idx, end_idx, fraction = get_start_end_idx(
            frames_length,
            clip_size,
            clip_idx,
            num_clips_uniform,
            use_offset=use_offset,
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
    container,
    sampling_rate,
    num_frames,
    clip_idx=-1,
    num_clips_uniform=10,
    video_meta=None,
    target_fps=30,
    backend="pyav",
    max_spatial_scale=0,
    use_offset=False,
    time_diff_prob=0.0,
    gaussian_prob=0.0,
    min_delta=-math.inf,
    max_delta=math.inf,
    temporally_rnd_clips=True,
):
    """
    Decode the video and perform temporal sampling.
    Args:
        container (container): pyav container.
        sampling_rate (list of ints): frame sampling rate (interval between two sampled
            frames).
        num_frames (list of ints): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal
            sampling. If clip_idx is larger than -1, uniformly split the
            video to num_clips_uniform clips, and select the
            clip_idx-th video clip.
        num_clips_uniform (int): overall number of clips to uniformly
            sample from the given video.
        video_meta (dict): a dict contains VideoMetaData. Details can be find
            at `pytorch/vision/torchvision/io/_video_opt.py`.
        target_fps (int): the input video may have different fps, convert it to
            the target video fps before frame sampling.
        backend (str): decoding backend includes `pyav` and `torchvision`. The
            default one is `pyav`.
        max_spatial_scale (int): keep the aspect ratio and resize the frame so
            that shorter edge size is max_spatial_scale. Only used in
            `torchvision` backend.
    Returns:
        frames (tensor): decoded frames from the video.
    """
    # Currently support two decoders: 1) PyAV, and 2) TorchVision.
    assert clip_idx >= -1, "Not valied clip_idx {}".format(clip_idx)
    assert len(sampling_rate) == len(num_frames)
    num_decode = len(num_frames)
    num_frames_orig = num_frames
    if num_decode > 1 and temporally_rnd_clips:
        ind_clips = np.random.permutation(num_decode)
        sampling_rate = [sampling_rate[i] for i in ind_clips]
        num_frames = [num_frames[i] for i in ind_clips]
    else:
        ind_clips = np.arange(
            num_decode
        )  # clips come temporally ordered from decoder
    try:
        if backend == "pyav":
            assert (
                min_delta == -math.inf and max_delta == math.inf
            ), "delta sampling not supported in pyav"
            frames_decoded, fps, decode_all_video = pyav_decode(
                container,
                sampling_rate,
                num_frames,
                clip_idx,
                num_clips_uniform,
                target_fps,
                use_offset=use_offset,
            )
        elif backend == "torchvision":
            (
                frames_decoded,
                fps,
                decode_all_video,
                start_end_delta_time,
            ) = torchvision_decode(
                container,
                sampling_rate,
                num_frames,
                clip_idx,
                video_meta,
                num_clips_uniform,
                target_fps,
                ("visual",),
                max_spatial_scale,
                use_offset=use_offset,
                min_delta=min_delta,
                max_delta=max_delta,
            )
        else:
            raise NotImplementedError(
                "Unknown decoding backend {}".format(backend)
            )
    except Exception as e:
        print("Failed to decode by {} with exception: {}".format(backend, e))
        return None, None, None

    # Return None if the frames was not decoded successfully.
    if frames_decoded is None or None in frames_decoded:
        return None, None, None

    if not isinstance(frames_decoded, list):
        frames_decoded = [frames_decoded]
    num_decoded = len(frames_decoded)
    clip_sizes = [
        np.maximum(1.0, sampling_rate[i] * num_frames[i] / target_fps * fps)
        for i in range(len(sampling_rate))
    ]

    if decode_all_video:  # full video was decoded (not trimmed yet)
        assert num_decoded == 1 and start_end_delta_time is None
        start_end_delta_time = get_multiple_start_end_idx(
            frames_decoded[0].shape[0],
            clip_sizes,
            clip_idx if decode_all_video else 0,
            num_clips_uniform if decode_all_video else 1,
            min_delta=min_delta,
            max_delta=max_delta,
            use_offset=use_offset,
        )

    frames_out, start_inds, time_diff_aug = (
        [None] * num_decode,
        [None] * num_decode,
        [None] * num_decode,
    )
    augment_vid = gaussian_prob > 0.0 or time_diff_prob > 0.0
    for k in range(num_decode):
        T = num_frames[k]
        # Perform temporal sampling from the decoded video.

        if decode_all_video:
            frames = frames_decoded[0]
            if augment_vid:
                frames = frames.clone()
            start_idx, end_idx = (
                start_end_delta_time[k, 0],
                start_end_delta_time[k, 1],
            )
        else:
            frames = frames_decoded[k]
            # video is already trimmed so we just need subsampling
            start_idx, end_idx, clip_position = get_start_end_idx(
                frames.shape[0], clip_sizes[k], 0, 1
            )
        if augment_vid:
            frames, time_diff_aug[k] = transform.augment_raw_frames(
                frames, time_diff_prob, gaussian_prob
            )
        frames_k = temporal_sampling(frames, start_idx, end_idx, T)
        frames_out[k] = frames_k

    # if we shuffle, need to randomize the output, otherwise it will always be past->future
    if num_decode > 1 and temporally_rnd_clips:
        frames_out_, time_diff_aug_ = [None] * num_decode, [None] * num_decode
        start_end_delta_time_ = np.zeros_like(start_end_delta_time)
        for i, j in enumerate(ind_clips):
            frames_out_[j] = frames_out[i]
            start_end_delta_time_[j, :] = start_end_delta_time[i, :]
            time_diff_aug_[j] = time_diff_aug[i]

        frames_out = frames_out_
        start_end_delta_time = start_end_delta_time_
        time_diff_aug = time_diff_aug_
        assert all(
            frames_out[i].shape[0] == num_frames_orig[i]
            for i in range(num_decode)
        )

    return frames_out, start_end_delta_time, time_diff_aug
