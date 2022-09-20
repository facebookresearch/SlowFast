from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin import pytorch
import nvidia.dali.ops as ops
import nvidia.dali.fn as fn

import nvidia.dali.types as types


class VideoReaderPipeline(Pipeline):
    def __init__(self, batch_size, sequence_length, num_threads, device_id, file_list, crop_size):
        super(VideoReaderPipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        # self.reader = fn.readers.video(
        #     device="gpu",
        #     file_list=file_list,
        #     sequence_length=sequence_length,
        #     normalized=False,
        #     random_shuffle=True,
        #     image_type=types.RGB,
        #     dtype=types.UINT8,
        #     initial_fill=16,
        #     enable_frame_num=True,
        # )

        self.sequence_length = sequence_length
        self.file_list = file_list

        self.uniform = fn.random.uniform(range=(0.0, 1.0))
        self.coin = fn.random.coin_flip(probability=0.5)
        self.crop_size = crop_size

    def define_graph(self):
        input = fn.readers.video(
            name="Reader",
            device="gpu",
            file_list=self.file_list,
            sequence_length=self.sequence_length,
            normalized=False,
            random_shuffle=True,
            image_type=types.RGB,
            dtype=types.UINT8,
            initial_fill=16,
            enable_frame_num=True,
        )
        crop_pos_x = fn.random.uniform(range=(0.0, 1.0))
        crop_pos_y = fn.random.uniform(range=(0.0, 1.0))
        cropped = fn.crop(
            input[0],
            device="gpu",
            crop=self.crop_size,
            dtype=types.FLOAT,
            crop_pos_x=crop_pos_x,
            crop_pos_y=crop_pos_y,
        )
        is_flipped = fn.random.coin_flip(probability=0.5)
        flipped = fn.flip(cropped, horizontal=is_flipped)
        output = fn.transpose(flipped, device="gpu", perm=[3, 0, 1, 2])
        # Change what you want from the dataloader.
        # input[1]: label, input[2]: starting frame number indexed from zero
        return output, input[1], input[2], crop_pos_x, crop_pos_y


class DALILoader:
    def __init__(self, batch_size, file_list, sequence_length, crop_size):
        self.pipeline = VideoReaderPipeline(
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_threads=2,
            device_id=0,
            file_list=file_list,
            crop_size=crop_size,
        )
        self.pipeline.build()
        self.epoch_size = self.pipeline.epoch_size("Reader")
        self.dali_iterator = pytorch.DALIGenericIterator(
            self.pipeline,
            ["data", "label", "frame_num", "crop_pos_x", "crop_pos_y"],
            self.epoch_size,
            auto_reset=True,
        )

    def __len__(self):
        return int(self.epoch_size)

    def __iter__(self):
        return self.dali_iterator.__iter__()

    def __next__(self):
        return self.dali_iterator.__next__()


# output format of dataloader
# => for cur_iter, (inputs, labels, index, time, meta) in enumerate(mcheck_loader):
