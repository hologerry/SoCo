import numpy as np
from torch.utils.data import Sampler


class SubsetSlidingWindowSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices, window_stride, window_size, shuffle_per_epoch=False):
        self.window_stride = window_stride
        self.window_size = window_size
        self.shuffle_per_epoch = shuffle_per_epoch
        self.indices = indices
        np.random.shuffle(self.indices)
        self.start_index = 0

    def __iter__(self):
        # optionally shuffle all indices per epoch
        if self.shuffle_per_epoch and self.start_index + self.window_size > len(self):
            np.random.shuffle(self.indices)

        # get indices of sampler in the current window
        indices = np.mod(np.arange(self.window_size, dtype=np.int) + self.start_index, len(self))
        window_indices = self.indices[indices]

        # shuffle the current window
        np.random.shuffle(window_indices)

        # move start index to next window
        self.start_index = (self.start_index + self.window_stride) % len(self)

        return iter(window_indices.tolist())

    def __len__(self):
        return len(self.indices)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {"start_index": self.start_index}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)
