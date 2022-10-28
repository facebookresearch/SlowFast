from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin import pytorch
import nvidia.dali.fn as fn

import nvidia.dali.types as types


class dali_transform:
    def __init__(self, crop_size, size, dali_cpu=False, is_training=True):
        self.crop_size = crop_size
        self.size = size
        self.dali_device = "cpu" if dali_cpu else "gpu"
        self.is_training = is_training

    def __call__(self, input):
        crop_pos_x = fn.random.uniform(range=(0.0, 1.0))
        crop_pos_y = fn.random.uniform(range=(0.0, 1.0))
        resized = fn.resize(
            input,
            device="gpu",
            dtype=types.FLOAT,
            mode="not_smaller",
            size=self.size,
        )
        cropped = fn.crop(
            resized,
            device="gpu",
            crop=self.crop_size,
            dtype=types.FLOAT,
            crop_pos_x=crop_pos_x,
            crop_pos_y=crop_pos_y,
        )
        is_flipped = fn.random.coin_flip(probability=0.5)
        flipped = fn.flip(cropped, horizontal=is_flipped)
        output = fn.transpose(flipped, device="gpu", perm=[3, 0, 1, 2])

        return output


class MultiViewDaliInjector(object):
    def __init__(self, *args):
        self.transforms = args[0]

    def __call__(self, sample):
        output = [transform(sample) for transform in self.transforms]
        return output


class VideoReaderPipeline(Pipeline):
    def __init__(
        self, batch_size, sequence_length, num_threads, device_id, file_list, crop_size, split
    ):
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
        self.split = split

    def define_graph(self):
        input = fn.readers.video(
            name="Reader",
            device="gpu",
            file_list=self.file_list,
            sequence_length=self.sequence_length,
            normalized=False,
            random_shuffle=False,
            image_type=types.RGB,
            dtype=types.UINT8,
            initial_fill=16,
            enable_frame_num=True,
            enable_timestamps=True,
        )

        if self.split == "train":
            # transforming
            single_transform = dali_transform(
                crop_size=self.crop_size,
                size=[224, 224],
                dali_cpu=False,
            )
            Mulinjector = MultiViewDaliInjector([single_transform, single_transform])
            output1, output2 = Mulinjector(input[0])
            # dummy index
            index = 1
            return output1, output2, input[1], index, input[2], input[3]
        else:
            single_transform = dali_transform(
                crop_size=self.crop_size,
                size=[224, 224],
                dali_cpu=False,
            )
            output = single_transform(input[0])
            index = 1
            return output, input[1], index, input[2], input[3]
        # Change what you want from the dataloader.
        # input[1]: label, input[2]: starting frame number indexed from zero
        # input[3]: timestamps


class DALILoader:
    def __init__(self, batch_size, file_list, sequence_length, crop_size, split):
        self.pipeline = VideoReaderPipeline(
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_threads=2,
            device_id=0,
            file_list=file_list,
            crop_size=crop_size,
            split=split,
        )
        self.pipeline.build()
        self.epoch_size = self.pipeline.epoch_size("Reader")
        if split == "train":
            self.dali_iterator = pytorch.DALIGenericIterator(
                self.pipeline,
                ["data1", "data2", "label", "index", "frame_num", "timestamp"],
                self.epoch_size,
                auto_reset=True,
            )
        else:
            self.dali_iterator = pytorch.DALIGenericIterator(
                self.pipeline,
                ["data", "label", "index", "frame_num", "timestamp"],
                self.epoch_size,
                auto_reset=True,
            )

        # ["data", "label", "frame_num", "crop_pos_x", "crop_pos_y"],
        # ["data", "labels", "index", "time", "meta"]

    def __len__(self):
        return int(self.epoch_size)

    def __iter__(self):
        return self.dali_iterator.__iter__()

    def __next__(self):
        return self.dali_iterator.__next__()
