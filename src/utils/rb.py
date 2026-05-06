import numpy as np
import torch


class ReplayBuffer:
    """
    Replay buffer implemented with torch tensor.
    """
    def __init__(self, buffer_size: int, keys: list[str]) -> None:
        """
        Args:
            buffer_size (int): Predefined size of buffer.
            keys (list[str]): Keys for each buffer.
        """
        self.buffer_size = int(buffer_size)
        self.keys = keys
        self.idx = 0
        self.full = False

        # Initialize buffer upon first registeration
        self.buffer = dict()

    def __len__(self):
        return self.idx if not self.full else self.buffer_size

    def clear(self) -> None:
        """
        Reset all the buffers.
        """
        self.idx = 0
        self.full = False
        for k in self.keys:
            self.buffer[k] = torch.zeros(self.buffer[k].shape)

    def register(self, data: dict) -> None:
        """
        Save the data given in dictionary format to buffer.

        Args:
            data (dict): Data given in dictionary format.
        """
        for k, v in data.items():
            if isinstance(v, (np.ndarray, list, np.ScalarType)):
                v = torch.tensor([v]).to(torch.float32)

            if v.ndim >= 2 and v.size()[0] == 1:
                # Deal with the batch data of size 1
                v = torch.squeeze(v, dim=0)

            # Initialize buffer if the given data was not given in the past
            if k not in self.buffer:
                buffer_shape = [self.buffer_size] + list(v.shape)
                self.buffer[k] = torch.zeros(buffer_shape)

            self.buffer[k][self.idx] = v

        self.idx += 1
        self.idx %= self.buffer_size

        if self.idx == 0:
            self.full = True

    def get_data(self, n_data: int) -> dict:
        """Get specified number of data from the front of replay buffers.

        Args:
            n_data (int): Number of samples to retrieve.

        Returns:
            dict: Samples retrieved from replay buffer.
        """
        cur_mx = self.buffer_size if self.full else self.idx

        assert n_data <= cur_mx, "n_data must be smaller than or equal to the number of currently registered data"

        data = {}
        for k in self.keys:
            data[k] = self.buffer[k][:n_data]

        return data

    def sample(self, sample_size: int) -> dict:
        """
        Sample specified number of data from replay buffer.

        Args:
            sample_size (int): Number of samples to retrieve.

        Returns:
            dict: Samples retrieved from replay buffer.
        """
        cur_mx = self.buffer_size if self.full else self.idx
        sample_idx = torch.randint(0, cur_mx, (sample_size,))
        samples = {}
        for k in self.keys:
            samples[k] = self.buffer[k][sample_idx]

        return samples
