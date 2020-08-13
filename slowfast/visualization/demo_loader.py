#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import atexit
import copy
import threading
import time
import cv2

import slowfast.utils.logging as logging
from slowfast.visualization.utils import TaskInfo

logger = logging.get_logger(__name__)


class VideoReader:
    """
    VideoReader object for getting frames from video source for inference.
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
        if cfg.DEMO.OUTPUT_FILE != "":
            if cfg.DEMO.OUTPUT_FPS == -1:
                output_fps = self.cap.get(cv2.CAP_PROP_FPS)
            else:
                output_fps = cfg.DEMO.OUTPUT_FPS
            self.output_file = self.get_output_file(
                cfg.DEMO.OUTPUT_FILE, fps=output_fps
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


class ThreadVideoReader:
    """
    VideoReader object for getting frames from video source for inference
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
        if cfg.DEMO.OUTPUT_FILE != "":
            if cfg.DEMO.OUTPUT_FPS == -1:
                output_fps = self.cap.get(cv2.CAP_PROP_FPS)
            else:
                output_fps = cfg.DEMO.OUTPUT_FPS
            self.output_file = self.get_output_file(
                cfg.DEMO.OUTPUT_FILE, fps=output_fps
            )
        self.get_id = -1
        self.put_id = -1
        self.buffer = []
        self.buffer_size = cfg.DEMO.BUFFER_SIZE
        self.seq_length = cfg.DATA.NUM_FRAMES * cfg.DATA.SAMPLING_RATE
        self.test_crop_size = cfg.DATA.TEST_CROP_SIZE
        self.clip_vis_size = cfg.DEMO.CLIP_VIS_SIZE

        self.task_queue = {}
        self.not_end = True
        self.taskqueue_lock = threading.Lock()
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
            self.taskqueue_lock.acquire()
            self.put_id += 1
            self.not_end = was_read
            self.task_queue[self.put_id] = (was_read, task)
            self.taskqueue_lock.release()

    def __next__(self):
        self.taskqueue_lock.acquire()
        # If there is nothing in the task queue.
        if len(self.task_queue) == 0:
            self.taskqueue_lock.release()
            return self.not_end, None
        else:
            # If we have already consume the latest task.
            if self.task_queue.get(self.put_id) is None:
                self.taskqueue_lock.release()
                time.sleep(0.02)
                return self.not_end, None
            was_read, task = self.task_queue[self.put_id]
            # If we reach the end of the video.
            if not was_read:
                self.taskqueue_lock.release()
                return was_read, None

            task = copy.deepcopy(task)
            del self.task_queue[self.put_id]
            self.taskqueue_lock.release()
            return was_read, task

    def get_fn(self):
        while not self.stopped:
            self.taskqueue_lock.acquire()
            # If video ended and we have display all frames.
            if not self.not_end and self.get_id == self.put_id:
                self.taskqueue_lock.release()
                break
            # If the next frames are not available, wait.
            if (
                len(self.task_queue) == 0
                or self.task_queue.get(self.get_id + 1) is None
            ):
                self.taskqueue_lock.release()
                time.sleep(0.02)
                continue
            else:
                self.get_id += 1
                was_read, task = self.task_queue[self.get_id]
                task = copy.deepcopy(task)
                del self.task_queue[self.get_id]
                self.taskqueue_lock.release()
                self.output_lock.acquire()
                for frame in task.frames[task.num_buffer_frames :]:
                    if self.output_file is None:
                        cv2.imshow("SlowFast", frame)
                    else:
                        self.output_file.write(frame)
                self.output_lock.release()

    def display(self, task):
        """
        Add the visualized task to the write queue for display/write to outputfile.
        Args:
            task (TaskInfo object): task object that contain
                the necessary information for prediction visualization. (e.g. visualized frames.)
        """
        self.task_queue[task.id] = (True, task)

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
