import os
import h5py
import torch
import numpy as np
from torch.utils import data
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader


class RadioMLDataset(data.Dataset):
    """RadioML data.
    Available here: https://www.deepsig.io/datasets.
    """

    def __init__(self, data_dir, train, normalize=False, fake_height=True, skip_1=False,
                 min_snr=6, max_snr=30, per_h5_frac=0.5, train_frac=0.9, per_sample_frac=1.0, classes=24):

        self.train = train

        if not os.path.exists(os.path.join(data_dir, 'class23_snr30.hdf5')):
            # Split huge HDF5 file into per-SNR, per-class HDF5 files

            data_path = os.path.join(data_dir, 'GOLD_XYZ_OSC.0001_1024.hdf5')
            full_h5f = h5py.File(data_path, 'r')

            # Contents of HDF5 file:
            # 'X': (2555904, 1024, 2) float32 array [represents signal]
            # 'Y': (2555904, 24) int64 array [represents class label]
            # 'Z': (2555904, 1) int64 array [represents SNR]

            # Important note:
            # The data is ordered by class, i.e.
            # the first ~1/24th consists of examples of the first class,
            # the second ~1/24th consists of examples of the second class, and so on.

            # There are 24 different modulation classes.
            # For each class, there are 2555904 / 24 = 106496 examples.
            # There are 26 different SNR levels: -20, -18, -16, ..., 28, 30.
            # For each (class, SNR) pair, there are 106496 / 26 = 4096 examples.

            # Convert one-hot labels back to argmax
            Y = np.argmax(full_h5f['Y'], axis=1)

            for class_idx in range(24):
                class_X = full_h5f['X'][Y == class_idx, :, :]
                class_Z = full_h5f['Z'][Y == class_idx, 0]
                for snr in range(-26, 32, 2):
                    class_snr_name = 'class%d_snr%d.hdf5' % (class_idx, snr)
                    h5f_path = os.path.join(data_dir, class_snr_name)
                    h5f = h5py.File(h5f_path, 'w')
                    h5f.create_dataset('X', data=class_X[class_Z == snr, :, :])
                    h5f.close()
                    print('Wrote (SNR {z}, class {cl}) data to `{path}`.'.format(
                        z=snr, cl=class_idx, path=h5f_path))
                class_X = None
                class_Z = None
            Y = None
            full_h5f.close()

        # Min/max values across entire dataset
        # Want to use the same values for train/test normalization

        # The data for each (class, SNR) pair
        # will be truncated to the first PER_H5_SIZE examples
        per_h5_size = int(per_h5_frac * 4096)
        snr_count = (max_snr - min_snr) // 2 + 1
        train_split_size = int(train_frac * per_h5_size)
        if train:
            split_size = train_split_size
        else:
            split_size = per_h5_size - train_split_size
        total_size = classes * snr_count * split_size

        sample_portion = int(1024*per_sample_frac)
        self.X = np.zeros((total_size, sample_portion, 2), dtype=np.float32)
        self.Y = np.zeros(total_size, dtype=np.int64)
        for class_idx in range(classes):
            for snr_idx, snr in enumerate(range(min_snr, max_snr + 2, 2)):
                class_snr_name = 'class%d_snr%d.hdf5' % (class_idx, snr)
                h5f_path = os.path.join(data_dir, class_snr_name)
                h5f = h5py.File(h5f_path, 'r')
                X = h5f['X'][:]
                if train:
                    X_split = X[:train_split_size, :sample_portion, :]
                else:
                    X_split = X[train_split_size:per_h5_size, :sample_portion, :]
                # Interleave
                start_idx = (class_idx * snr_count) + snr_idx
                self.X[start_idx::classes*snr_count] = X_split
                self.Y[start_idx::classes*snr_count] = class_idx
                h5f.close()
                X = None
                X_split = None

        if fake_height:
            # Add fake height dim (TODO remove if switching to 1D convolutions)
            self.X = self.X.transpose(0, 2, 1)[:, :, np.newaxis, :]
        else:
            self.X = self.X.transpose(0, 2, 1)[:, :, :]

        if normalize:
            xr = self.X[:, 0, :]
            xi = self.X[:, 1, :]
            xr = (xr - xr.min(1)[:, None]) / (xr.max(1) - xr.min(1))[:, None]
            xi = (xi - xi.min(1)[:, None]) / (xi.max(1) - xi.min(1))[:, None]
            self.X[:, 0, :] = xr
            self.X[:, 1, :] = xi

        if skip_1:
            self.X = self.X[:, :, np.mod(np.arange(self.X.shape[2]), 2) != 0]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


def get_radio_ml_loader(batch_size, train, **kwargs):
    data_dir = kwargs['data_dir']
    min_snr = kwargs.get('min_snr', 6)
    max_snr = kwargs.get('max_snr', 30)
    per_h5_frac = kwargs.get('per_h5_frac', 0.5)
    train_frac = kwargs.get('train_frac', 0.9)
    per_sample_frac = kwargs.get('per_sample_frac', 1.0)
    normalize = kwargs.get('normalize', True)
    skip_1 = kwargs.get('skip_1', False)
    fake_height = kwargs.get('fake_height', False)
    classes = kwargs.get('classes', 24)
    dataset = RadioMLDataset(data_dir, train,
                             normalize=normalize,
                             fake_height=fake_height,
                             min_snr=min_snr,
                             max_snr=max_snr,
                             per_h5_frac=per_h5_frac,
                             train_frac=train_frac,
                             skip_1=skip_1,
                             per_sample_frac=per_sample_frac,
                             classes=classes)

    identifier = 'train' if train else 'test'
    print('[%s] dataset size: %d' % (identifier, len(dataset)))

    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=train)
    loader.name = 'RadioML_{}'.format(identifier)

    return loader
