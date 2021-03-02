import torch
import numpy as np


def iq2spiketrain(x, out_w=28, out_h=28):
    """Convert each I/Q sample to a spike in the I/Q plane over time.
    Assumption: the shape of X is (batch, 2, num_timesteps).
    """
    # Generate spike trains
    batch_size, _, num_timesteps = x.shape
    spike_trains = np.zeros((batch_size, num_timesteps, 1, out_w, out_h))
    t_start = 0
    t_end = num_timesteps
    for i, t in enumerate(range(t_start, t_end)):
        # Obtain I/Q values
        I_value = x[:, 0, t]
        Q_value = x[:, 1, t]

        # Quantize to cells in image (scale, clamp, convert to int)
        cell_I = (I_value * (out_w - 1)).int()
        cell_Q = (Q_value * (out_h - 1)).int()
        # Assign events to samples
        for b in range(batch_size):
            spike_trains[b, i, 0, cell_Q[b], cell_I[b]] = 1

    # The shape of `all_target` is (max_duration, batch_size, target_size)
    # all_target = np.repeat(y[np.newaxis, :, :], num_timesteps, axis=0)

    return torch.Tensor(spike_trains)
