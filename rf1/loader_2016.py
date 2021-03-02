import pickle
import numpy as np
from torch.utils import data
from torch.utils.data.dataloader import DataLoader


def to_notonehot(yy):
    return yy


def get_numpy_2016(data_dir, min_snr, max_snr, train_frac=0.9):
    f = open(data_dir, 'rb')
    Xd = pickle.load(f, encoding='bytes')
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    X = []
    lbl = []

    new_snrs = []
    for i in range(min_snr, max_snr + 1, 2):
        new_snrs.append(i)

    for mod in mods:
        for snr in new_snrs:
            X.append(Xd[(mod, snr)])
            for i in range(Xd[(mod, snr)].shape[0]):  lbl.append((mod, snr))
    X = np.vstack(X)

    # Partition the data
    #  into training and test sets of the form we can train/test on
    #  while keeping SNR and Mod labels handy for each
    n_examples = X.shape[0]
    n_train = int(n_examples * train_frac)
    train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
    test_idx = list(set(range(0, n_examples)) - set(train_idx))
    X_train = X[train_idx]
    X_test = X[test_idx]

    Y_train = to_notonehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
    Y_test = to_notonehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

    return X_train, X_test, Y_train, Y_test


class RadioMLDataset2016(data.Dataset):
    """RadioML data.
    Available here: https://www.deepsig.io/datasets.
    """

    def __init__(self, X, Y, normalize=False):

        self.X = np.array(X, dtype=np.float32)
        self.Y = np.array(Y, dtype=np.int64)

        if normalize:
            xr = self.X[:, 0, :]
            xi = self.X[:, 1, :]
            xr = (xr - xr.min(1)[:, None]) / (xr.max(1) - xr.min(1))[:, None]
            xi = (xi - xi.min(1)[:, None]) / (xi.max(1) - xi.min(1))[:, None]
            self.X[:, 0, :] = xr
            self.X[:, 1, :] = xi

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]


def get_radio_ml_loader_2016(batch_size, X, Y, train, normalize):
    dataset = RadioMLDataset2016(X, Y, normalize=normalize)

    identifier = 'train' if train else 'test'
    print('[%s] dataset size: %d' % (identifier, len(dataset)))

    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=train)
    loader.name = 'RadioML2016_{}'.format(identifier)

    return loader
