#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import atexit
import copy
import queue
import threading
import time
import cv2

import slowfast.utils.logging as logging
from slowfast.visualization.utils import TaskInfo

logger = logging.get_logger(__name__)


class VideoManager:
    """
    VideoManager object for getting frames from video source for inference.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        """
        assert (
            cfg.DEMO.WEBCAM > -1 or cfg.DEMO.INPUT_VIDEO != ""
        ), "Must specify a data source as input."

        self.source = (
            cfg.DEMO.WEBCAM if cfg.DEMO.WEBCAM > -1 else cfg.DEMO.INPUT_VIDEO
        )

        self.display_width = cfg.DEMO.DISPLAY_WIDTH
        self.display_height = cfg.DEMO.DISPLAY_HEIGHT

        self.cap = cv2.VideoCapture(self.source)

        if self.display_width > 0 and self.display_height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.display_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.display_height)
        else:
            self.display_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.display_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not self.cap.isOpened():
            raise IOError("Video {} cannot be opened".format(self.source))

        self.output_file = None
        if cfg.DEMO.OUTPUT_FPS == -1:
            self.output_fps = self.cap.get(cv2.CAP_PROP_FPS)
        else:
            self.output_fps = cfg.DEMO.OUTPUT_FPS
        if cfg.DEMO.OUTPUT_FILE != "":
            self.output_file = self.get_output_file(
                cfg.DEMO.OUTPUT_FILE, fps=self.output_fps
            )
        self.id = -1
        self.buffer = []
        self.buffer_size = cfg.DEMO.BUFFER_SIZE
        self.seq_length = cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE
        self.test_crop_size = cfg.DATA.TEST_CROP_SIZE
        self.clip_vis_size = cfg.DEMO.CLIP_VIS_SIZE

    def __iter__(self):
        return self

    def __next__(self):
        """
        Read and return the required number of frames for 1 clip.
        Returns:
            was_read (bool): False if not enough frames to return.
            task (TaskInfo object): object contains metadata for the current clips.
        """
        self.id += 1
        task = TaskInfo()

        task.img_height = self.display_height
        task.img_width = self.display_width
        task.crop_size = self.test_crop_size
        task.clip_vis_size = self.clip_vis_size

        frames = []
        if len(self.buffer) != 0:
            frames = self.buffer
        was_read = True
        while was_read and len(frames) < self.seq_length:
            was_read, frame = self.cap.read()
            frames.append(frame)
        if was_read and self.buffer_size != 0:
            self.buffer = frames[-self.buffer_size :]

        task.add_frames(self.id, frames)
        task.num_buffer_frames = 0 if self.id == 0 else self.buffer_size

        return was_read, task

    def get_output_file(self, path, fps=30):
        """
        Return a video writer object.
        Args:
            path (str): path to the output video file.
            fps (int or float): frames per second.
        """
        return cv2.VideoWriter(
            filename=path,
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=float(fps),
            frameSize=(self.display_width, self.display_height),
            isColor=True,
        )

    def display(self, task):
        """
        Either display a single frame (BGR image) to a window or write to
        an output file if output path is provided.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for prediction visualization. (e.g. visualized frames.)
        """
        for frame in task.frames[task.num_buffer_frames :]:
            if self.output_file is None:
                cv2.imshow("SlowFast", frame)
                time.sleep(1 / self.output_fps)
            else:
                self.output_file.write(frame)

    def clean(self):
        """
        Clean up open video files and windows.
        """
        self.cap.release()
        if self.output_file is None:
            cv2.destroyAllWindows()
        else:
            self.output_file.release()

    def start(self):
        return self

    def join(self):
        pass


class ThreadVideoManager:
    """
    VideoManager object for getting frames from video source for inference
    using multithreading for read and write frames.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        """
        assert (
            cfg.DEMO.WEBCAM > -1 or cfg.DEMO.INPUT_VIDEO != ""
        ), "Must specify a data source as input."

        self.source = (
            cfg.DEMO.WEBCAM if cfg.DEMO.WEBCAM > -1 else cfg.DEMO.INPUT_VIDEO
        )

        self.display_width = cfg.DEMO.DISPLAY_WIDTH
        self.display_height = cfg.DEMO.DISPLAY_HEIGHT

        self.cap = cv2.VideoCapture(self.source)

        if self.display_width > 0 and self.display_height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.display_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.display_height)
        else:
            self.display_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.display_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not self.cap.isOpened():
            raise IOError("Video {} cannot be opened".format(self.source))

        self.output_file = None

        if cfg.DEMO.OUTPUT_FPS == -1:
            self.output_fps = self.cap.get(cv2.CAP_PROP_FPS)
        else:
            self.output_fps = cfg.DEMO.OUTPUT_FPS
        if cfg.DEMO.OUTPUT_FILE != "":
            self.output_file = self.get_output_file(
                cfg.DEMO.OUTPUT_FILE, fps=self.output_fps
            )
        self.num_skip = cfg.DEMO.NUM_CLIPS_SKIP + 1
        self.get_id = -1
        self.put_id = -1
        self.buffer = []
        self.buffer_size = cfg.DEMO.BUFFER_SIZE
        self.seq_length = cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE
        self.test_crop_size = cfg.DATA.TEST_CROP_SIZE
        self.clip_vis_size = cfg.DEMO.CLIP_VIS_SIZE

        self.read_queue = queue.Queue()
        self.write_queue = {}
        self.not_end = True
        self.write_lock = threading.Lock()
        self.put_id_lock = threading.Lock()
        self.input_lock = threading.Lock()
        self.output_lock = threading.Lock()
        self.stopped = False
        atexit.register(self.clean)

    def get_output_file(self, path, fps=30):
        """
        Return a video writer object.
        Args:
            path (str): path to the output video file.
            fps (int or float): frames per second.
        """
        return cv2.VideoWriter(
            filename=path,
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=float(fps),
            frameSize=(self.display_width, self.display_height),
            isColor=True,
        )

    def __iter__(self):
        return self

    def put_fn(self):
        """
        Grabbing frames from VideoCapture.
        """
        was_read = True
        while was_read and not self.stopped:
            task = TaskInfo()

            task.img_height = self.display_height
            task.img_width = self.display_width
            task.crop_size = self.test_crop_size
            task.clip_vis_size = self.clip_vis_size
            frames = []
            if len(self.buffer) != 0:
                frames = self.buffer
            self.input_lock.acquire()
            while was_read and len(frames) < self.seq_length:
                was_read, frame = self.cap.read()
                if was_read:
                    frames.append(frame)
            self.input_lock.release()
            if was_read:
                self.buffer = frames[-self.buffer_size :]

            task.add_frames(self.put_id + 1, frames)
            task.num_buffer_frames = (
                0 if self.put_id == -1 else self.buffer_size
            )
            with self.put_id_lock:
                self.put_id += 1
                self.not_end = was_read
            # If mode is to read the most recent clip or we reach task
            # index that is not supposed to be skipped.
            if self.num_skip == 0 or self.put_id % self.num_skip == 0:
                self.read_queue.put((was_read, copy.deepcopy(task)))
            else:
                with self.write_lock:
                    self.write_queue[task.id] = (was_read, copy.deepcopy(task))

    def __next__(self):
        # If there is nothing in the task queue.
        if self.read_queue.qsize() == 0:
            return self.not_end, None
        else:
            with self.put_id_lock:
                put_id = self.put_id
            was_read, task = None, None
            # If mode is to predict most recent read clip.
            if self.num_skip == 0:
                # Write all previous clips to write queue.
                with self.write_lock:
                    while True:
                        was_read, task = self.read_queue.get()
                        if task.id == put_id:
                            break
                        self.write_queue[task.id] = (was_read, task)
            else:
                was_read, task = self.read_queue.get()
            # If we reach the end of the video.
            if not was_read:
                # Put to write queue.
                with self.write_lock:
                    self.write_queue[put_id] = was_read, copy.deepcopy(task)
                task = None
            return was_read, task

    def get_fn(self):
        while not self.stopped:
            with self.put_id_lock:
                put_id = self.put_id
                not_end = self.not_end

            with self.write_lock:
                # If video ended and we have display all frames.
                if not not_end and self.get_id == put_id:
                    break
                # If the next frames are not available, wait.
                if (
                    len(self.write_queue) == 0
                    or self.write_queue.get(self.get_id + 1) is None
                ):
                    time.sleep(0.02)
                    continue
                else:
                    self.get_id += 1
                    was_read, task = self.write_queue[self.get_id]
                    del self.write_queue[self.get_id]

            with self.output_lock:
                for frame in task.frames[task.num_buffer_frames :]:
                    if self.output_file is None:
                        cv2.imshow("SlowFast", frame)
                        time.sleep(1 / self.output_fps)
                    else:
                        self.output_file.write(frame)

    def display(self, task):
        """
        Add the visualized task to the write queue for display/write to outputfile.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for prediction visualization. (e.g. visualized frames.)
        """
        with self.write_lock:
            self.write_queue[task.id] = (True, task)

    def start(self):
        """
        Start threads to read and write frames.
        """
        self.put_thread = threading.Thread(
            target=self.put_fn, args=(), name="VidRead-Thread", daemon=True
        )
        self.put_thread.start()
        self.get_thread = threading.Thread(
            target=self.get_fn, args=(), name="VidDisplay-Thread", daemon=True
        )
        self.get_thread.start()

        return self

    def join(self):
        self.get_thread.join()

    def clean(self):
        """
        Clean up open video files and windows.
        """
        self.stopped = True
        self.input_lock.acquire()
        self.cap.release()
        self.input_lock.release()
        self.output_lock.acquire()
        if self.output_file is None:
            cv2.destroyAllWindows()
        else:
            self.output_file.release()
        self.output_lock.release()
