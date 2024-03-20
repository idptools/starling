import numpy as np
import torch
from IPython import embed


class RunningStats:
    def __init__(self, height, width, epsilon=1e-5, alpha=0.9):
        self.shape = (height, width)

        self.mean = np.zeros(self.shape)
        self.var = np.zeros(self.shape)
        self.count = 0
        self.epsilon = epsilon
        self.alpha = alpha

    def update(self, data):
        data = np.round(data, 2)
        non_zero_counts = np.count_nonzero(data, axis=0)
        non_zero_counts[non_zero_counts == 0] = 1
        batch_mean = data.sum(axis=0) / non_zero_counts

        batch_var = np.zeros(self.shape)
        for dt in data:
            padding_starts = np.sum(dt != 0, axis=1)[0][0]
            dt_var = (
                abs(
                    dt[0, :padding_starts, :padding_starts]
                    - batch_mean[0, :padding_starts, :padding_starts]
                )
                ** 2
            )
            dt_var = self.MaxPad(dt_var)
            batch_var += dt_var

        # batch_var /= non_zero_counts[0]

        # Update count
        self.count += data.shape[0]

        # Update mean using exponential moving average
        delta = batch_mean[0] - self.mean
        self.mean += delta * data.shape[0] / self.count

        # Update variance using the update formula for variance
        m_a = self.var * (self.count - data.shape[0])
        m_b = batch_var * data.shape[0]
        M2 = m_a + m_b + delta**2 * self.count * data.shape[0] / self.count
        self.var = M2 / self.count
        embed()

    def normalize(self, data):
        normalized_data = (data - self.mean) / (self.var**0.5 + self.epsilon)

        return normalized_data

    def scale(self, data):
        max_matrix = data.max(axis=0)
        min_matrix = data.min(axis=0)

    def get_stats(self):
        var = self.M2 / (self.count - 1)  # Corrected sample variance
        return self.mean, np.sqrt(var + self.epsilon)

    def save_stats(self):
        pass

    def MaxPad(self, original_array):
        # Pad the distance map to a desired shape, here we are using
        # (768, 768) because largest sequences are 750 residues long
        # and 768 can be divided by 2 a bunch of times leading to nice
        # behavior during conv2d and conv2transpose down- and up-sampling
        pad_height = max(0, self.shape[0] - original_array.shape[0])
        pad_width = max(0, self.shape[1] - original_array.shape[1])
        return np.pad(
            original_array,
            ((0, pad_height), (0, pad_width)),
            mode="constant",
            constant_values=0,
        )


class OnlineStatistics2DBatch:
    def __init__(self, dim):
        self.dim = dim
        self.count = 0
        self.mean = np.zeros(dim)
        self.M2 = np.zeros(dim)

    def update_batch(self, batch):
        batch_size = batch.shape[0]
        self.count += batch_size
        non_zero_counts = np.count_nonzero(batch, axis=0)
        non_zero_counts[non_zero_counts == 0] = 1

        delta = np.zeros(self.dim)
        for b in batch:
            padding_starts = np.sum(b != 0, axis=1)[0][0]
            b_var = (
                b[0, :padding_starts, :padding_starts]
                - self.mean[:padding_starts, :padding_starts]
            )
            delta[:padding_starts, :padding_starts] += b_var
        self.mean += delta / non_zero_counts[0]

        delta2 = np.zeros(self.dim)
        for b in batch:
            padding_starts = np.sum(b != 0, axis=1)[0][0]
            b_var = (
                b[0, :padding_starts, :padding_starts]
                - self.mean[:padding_starts, :padding_starts]
            )
            delta2[:padding_starts, :padding_starts] += b_var
        self.M2 += delta * delta2
        embed()

    def get_mean(self):
        return self.mean if self.count > 0 else None

    def get_variance(self):
        return self.M2 / self.count if self.count > 1 else None

    def get_stddev(self):
        return np.sqrt(self.get_variance())
