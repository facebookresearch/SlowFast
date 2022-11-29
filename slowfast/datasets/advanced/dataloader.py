from typing import Optional
import torch
import multiprocessing as mp
from torch.utils.data.dataloader import DataLoader

# https://github.com/pytorch/pytorch/issues/15849
class _RepeatSampler(object):
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(torch.utils.data.dataloader.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class SuperFastDataLoader:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.q = mp.Queue()
        self.known_len = None
        self.loader: Optional[DataLoader] = None

        self.proc = mp.Process(target=self._proc)
        self.proc.start()
        self.known_len = self.q.get()

    def _proc(self):
        self.loader = DataLoader(*self.args, **self.kwargs)
        print(self.args, self.kwargs)
        print("Init loader in other proc", self.known_len)
        self.q.put(len(self.loader))
        for x in self.loader:
            print("Putting data")
            self.q.put(x)
        self.q.put(None)

    def __len__(self):
        return self.known_len

    def __iter__(self):
        while True:
            print("Wait for data ", self.known_len)
            x = self.q.get()
            if x is None:
                break
            yield x


class DummyInputGenerator:
    def __init__(self, batch_size, length=30, input_shape=96):
        self.batch_size = batch_size
        self.length = length
        self.a = torch.rand((self.batch_size, 3, input_shape, input_shape)).share_memory_()
        self.b = torch.rand((self.batch_size, 3, input_shape, input_shape)).share_memory_()
        print(f"Using 3x{input_shape}x{input_shape}")

    def __len__(self):
        return self.length

    def __iter__(self):
        for _ in range(self.length):
            yield (self.a, self.b), None
