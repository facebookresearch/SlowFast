#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import math
import numpy as np
import random
import torch
import torchvision.io as io
import librosa


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
    num_clips=10,
    target_fps=30,
    modalities=("visual",),
    max_spatial_scale=0,
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
            video to num_clips clips, and select the clip_idx-th video clip.
        video_meta (dict): a dict contains VideoMetaData. Details can be found
            at `pytorch/vision/torchvision/io/_video_opt.py`.
        num_clips (int): overall number of clips to uniformly sample from the
            given video.
        target_fps (int): the input video may has different fps, convert it to
            the target video fps.
        modalities (tuple): tuple of modalities to decode. Currently only
            support `visual`, planning to support `acoustic` soon.
        max_spatial_scale (int): the maximal resolution of the spatial shorter
            edge size during decoding.
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

    if (
        video_meta["has_video"]
        and video_meta["video_denominator"] > 0
        and video_meta["video_duration"] > 0
    ):
        decode_all_video = False
        start_idx, end_idx = get_start_end_idx(
            video_meta["video_fps"] * video_meta["video_duration"],
            sampling_rate * num_frames / target_fps * video_meta["video_fps"],
            clip_idx,
            num_clips,
        )
        # Convert frame index to pts.
        pts_per_frame = (
            video_meta["video_denominator"] / video_meta["video_fps"]
        )
        video_start_pts = int(start_idx * pts_per_frame)
        video_end_pts = int(end_idx * pts_per_frame)

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
    )
    return v_frames, video_meta["video_fps"], decode_all_video


def gen_logmel(y, orig_sr, sr, win_sz, step_sz, n_mels):
	"""
    Generate log-mel-spectrogram features from audio waveform

    Args:
        y (ndarray): audio waveform input.
        orig_sr (int): original sampling rate of audio inputs.
        sr (int): targeted sampling rate.
        win_sz (int): window step size in ms.
        step_sz (int): step size in ms.
        n_mels (int): number of frequency bins.
    Returns:
        logS (ndarray): log-mel-spectrogram computed from the input waveform.
    """
    n_fft = int(float(sr) / 1000 * win_sz)
    hop_length = int(float(sr) / 1000 * step_sz)
    win_length = int(float(sr) / 1000 * win_sz)
    eps = 1e-8
    y = y.reshape(-1)
    y = np.asfortranarray(y)
    y_resample = librosa.resample(y, orig_sr, sr, res_type='polyphase')
    T = len(y_resample) / sr
    S = librosa.feature.melspectrogram(y=y_resample, sr=sr, n_fft=n_fft, 
                                   win_length=win_length, hop_length=hop_length, 
                                   n_mels=n_mels, htk=True, center=False)
    logS = np.log(S+eps)
    return logS


def pyav_decode(
    container, sampling_rate, num_frames, clip_idx, num_clips=10, target_fps=30,
    decode_audio=False, extract_logmel=True, decode_all_audio=False, 
    au_sr=16000, au_win_sz=32, au_step_sz=16, au_n_mels=40,
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

    frames, audio_frames, au_raw_sr = None, None, None
    meta = {}
    
    # If video stream was found, fetch video frames from the video.
    if container.streams.video:
        video_frames, max_pts = pyav_decode_stream(
            container,
            video_start_pts,
            video_end_pts,
            container.streams.video[0],
            {"video": 0},
        )
        frames = [frame.to_rgb().to_ndarray() for frame in video_frames]
        frames = torch.as_tensor(np.stack(frames))
    
    meta.update({
        'video_start': video_start_pts / duration,
        'video_end': video_end_pts / duration,
    })
    
    # If audio stream was found, extract audio waveform from the video.
    if decode_audio and container.streams.audio:
        au_raw_sr = container.streams.audio[0].codec_context.sample_rate
        audio_duration = container.streams.audio[0].duration
        # audio_frames_length = container.streams.audio[0].frames
        # audio_timebase = audio_duration / audio_frames_length
        if decode_all_video or decode_all_audio:
            audio_start_pts = 0
            audio_end_pts = math.inf
        else:
            audio_start_pts = int(start_idx / frames_length * audio_duration)
            audio_end_pts = int(end_idx / frames_length * audio_duration)
        audio_frames, audio_max_pts = pyav_decode_stream(
            container,
            audio_start_pts,
            audio_end_pts,
            container.streams.audio[0],
            {"audio": 0},
        )
        
        audio_frames = [frame.to_ndarray() for frame in audio_frames]
        if len({x.shape[1] for x in audio_frames}) == 1:
            # This is a bit faster then the alternative
            audio_frames = np.concatenate([x[None] for x in audio_frames], axis=0)
            audio_frames = np.mean(audio_frames, axis=1)
            audio_frames = audio_frames.reshape(-1)
        else:
            audio_frames = [np.mean(x, axis=0) for x in audio_frames]
            audio_frames = np.concatenate(audio_frames, axis=0)
        meta.update({
            'audio_start': audio_start_pts / audio_duration,
            'audio_end': audio_end_pts / audio_duration,
        })
    
        # Extract log-mel-spectrogram features.
        if extract_logmel:
            audio_frames = gen_logmel(audio_frames, au_raw_sr, au_sr, 
                                      au_win_sz, au_step_sz, au_n_mels)
            audio_frames = audio_frames.transpose(1, 0) # [F,T]->[T,F]
        audio_frames = torch.as_tensor(audio_frames)
    
    meta.update({
        'decode_all_video': decode_all_video,
        'decode_all_audio': decode_all_audio,
    })
    
    container.close()
    
    return frames, fps, audio_frames, au_raw_sr, meta


def sample_misaligned_start(start_idx, gap, frames):
	"""
    Decide the starting point of a misaligned (i.e., negative) audio sample,
    which can be used for audiovisual synchronization training for self and 
    semi-supervised training.

    Args:
        start_idx (float): starting point of the positive sample.
        gap (int): the minimal gap to maintain between positive and negative samples.
        frames (tensor): decoded log-mel-spectrogram features.
    Returns:
        misaligned_start (float): starting point of the misaligned sample.
    """
    total_frames = frames.shape[0]
    pre_sample_region = (0, max(start_idx - gap, 0))
    post_sample_region = (min(start_idx + gap, total_frames), total_frames)
    pre_size = pre_sample_region[1] - pre_sample_region[0]
    post_size = post_sample_region[1] - post_sample_region[0]
    misaligned_start = random.random() * (pre_size + post_size)
    if misaligned_start > pre_size:
        misaligned_start = misaligned_start - pre_size + post_sample_region[0]
    return misaligned_start


def decode(
    container,
    sampling_rate,
    num_frames,
    clip_idx=-1,
    num_clips=10,
    video_meta=None,
    target_fps=30,
    backend="pyav",
    max_spatial_scale=0,
    # audio-related
    decode_audio=False,
    get_misaligned_audio=False,
    extract_logmel=False,
    au_sr=16000,
    au_win_sz=32, 
    au_step_sz=16, 
    num_audio_frames=128,
    au_n_mels=40,
    au_misaligned_gap=32,
):
    """
    Decode the video and perform temporal sampling.
    Args:
        container (container): pyav container.
        sampling_rate (int): frame sampling rate (interval between two sampled
            frames).
        num_frames (int): number of frames to sample.
        clip_idx (int): if clip_idx is -1, perform random temporal
            sampling. If clip_idx is larger than -1, uniformly split the
            video to num_clips clips, and select the
            clip_idx-th video clip.
        num_clips (int): overall number of clips to uniformly
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
    if decode_audio: assert backend == "pyav", 'Use PyAV for audio decoding'
    frames, audio_frames, misaligned_audio_frames = None, None, None
    try:
        if backend == "pyav":
            frames, fps, audio_frames, au_raw_sr, meta = pyav_decode(
                container,
                sampling_rate,
                num_frames,
                clip_idx,
                num_clips,
                target_fps,
                decode_audio=decode_audio,
                extract_logmel=extract_logmel,
                decode_all_audio=get_misaligned_audio,
                au_sr=au_sr, 
                au_win_sz=au_win_sz, 
                au_step_sz=au_step_sz, 
                au_n_mels=au_n_mels,
            )
            decode_all_video = meta['decode_all_video']
        elif backend == "torchvision":
            frames, fps, decode_all_video = torchvision_decode(
                container,
                sampling_rate,
                num_frames,
                clip_idx,
                video_meta,
                num_clips,
                target_fps,
                ("visual",),
                max_spatial_scale,
            )
        else:
            raise NotImplementedError(
                "Unknown decoding backend {}".format(backend)
            )
    except Exception as e:
        print("Failed to decode by {} with exception: {}".format(backend, e))
        return frames, audio_frames, misaligned_audio_frames

    # Return None if the frames was not decoded successfully.
    if frames is None or frames.size(0) == 0:
        return frames, audio_frames, misaligned_audio_frames

    start_idx, end_idx = get_start_end_idx(
        frames.shape[0],
        num_frames * sampling_rate * fps / target_fps,
        clip_idx if decode_all_video else 0,
        num_clips if decode_all_video else 1,
    )
    if decode_audio and audio_frames is not None:
        if get_misaligned_audio:
            video_start = meta['video_start']
            video_end = meta['video_end']
            video_duration = video_end - video_start
            audio_start_idx = (video_start + start_idx / frames.shape[0] * \
                               video_duration) * audio_frames.shape[0]
            audio_end_idx = (video_start + end_idx / frames.shape[0] * \
                             video_duration) * audio_frames.shape[0]
        else:
            audio_start_idx = start_idx / frames.shape[0] * audio_frames.shape[0]
            audio_end_idx = end_idx / frames.shape[0] * audio_frames.shape[0]
            # audio_end_idx = audio_start_idx + num_audio_frames - 1
    
    # Perform temporal sampling from the decoded video.
    frames = temporal_sampling(frames, start_idx, end_idx, num_frames)
    
    # Perform temporal sampling from the decoded audio.
    if decode_audio and audio_frames is not None:
        if get_misaligned_audio:
            audio_frame_len = audio_end_idx - audio_start_idx
            misaligned_audio_start_idx = sample_misaligned_start(
                audio_start_idx, 
                au_misaligned_gap, 
                audio_frames,
            )
            misaligned_audio_end_idx = misaligned_audio_start_idx + audio_frame_len
            misaligned_audio_frames = temporal_sampling(
                audio_frames, 
                misaligned_audio_start_idx, 
                misaligned_audio_end_idx, 
                num_audio_frames
            )
            misaligned_audio_frames = misaligned_audio_frames.reshape(
                1, 
                1, 
                misaligned_audio_frames.size(0), 
                misaligned_audio_frames.size(1)
            )        
        audio_frames = temporal_sampling(audio_frames, audio_start_idx, 
                                         audio_end_idx, num_audio_frames)
        audio_frames = audio_frames.reshape(1, 1, \
                        audio_frames.size(0), audio_frames.size(1))
    
    return frames, audio_frames, misaligned_audio_frames
