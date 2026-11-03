"""Just enough of torch for engines/seedvr2_runner.py, for CPU tests."""
import numpy as np

float16 = np.float16


class _Tensor:
    def __init__(self, a):
        self.a = a

    def to(self, dtype):
        return _Tensor(self.a.astype(dtype))


def from_numpy(a):
    return _Tensor(a)


class cuda:
    @staticmethod
    def get_device_name(_i):
        return "fake GPU"

    @staticmethod
    def reset_peak_memory_stats():
        pass

    @staticmethod
    def max_memory_allocated():
        return 0
