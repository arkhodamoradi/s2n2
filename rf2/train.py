import torch
import numpy as np
import os
import argparse
import datetime
from tensorboardX import SummaryWriter
from data.load_radio_ml import get_radio_ml_loader as get_loader
from data.data_utils import iq2spiketrain as to_spike_train
import matplotlib.pyplot as plt
import tqdm
from mpl_toolkits import mplot3d


if __name__ == '__main__':
    classes = ['32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK', 'BPSK', '8PSK', 'AM-SSB-SC', '4ASK',
               '16PSK', '64APSK', '128QAM', '128APSK', 'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM', 'AM-DSB-WC',
               'OOK', '16QAM']

    modulation_idx = classes.index('32PSK')  # OOK 32PSK 64QAM

    torch.manual_seed(123)
    np.random.seed(123)

    get_loader_kwargs = {}
    to_st_train_kwargs = {}

    # Set "get loader" kwargs
    get_loader_kwargs['data_dir'] = '/mnt/013c8c34-4de2-4dab-9e29-16618f093336/playground/RFSNN/2018.01'
    get_loader_kwargs['min_snr'] = 6
    get_loader_kwargs['max_snr'] = 6
    get_loader_kwargs['per_h5_frac'] = 0.25
    get_loader_kwargs['train_frac'] = 0.9
    get_loader_kwargs['per_sample_frac'] = 1.0
    get_loader_kwargs['normalize'] = True
    get_loader_kwargs['fake_height'] = False
    get_loader_kwargs['skip_1'] = False
    get_loader_kwargs['classes'] = 24
    # Set "to spike train" kwargs

    wh = 16
    to_st_train_kwargs['out_w'] = wh #args.I_resolution
    to_st_train_kwargs['out_h'] = wh #args.Q_resolution

    train_data = get_loader(24, train=True, **get_loader_kwargs)
    gen_train = iter(train_data)

    fig, ax = plt.subplots(2, 1)
    plt.ion()
    plt.show()
    for step in range(10):
        try:
            input, labels = next(gen_train)
        except StopIteration:
            gen_train = iter(train_data)
            input, labels = next(gen_train)

        input_spikes = to_spike_train(input, **to_st_train_kwargs)

        for idx in range(24):
            if labels[idx] == modulation_idx:
                img = None
                im3d = np.zeros((1024, wh, wh), dtype=np.uint8)
                for i in range(1024):

                    im3d[i] = input_spikes[idx, i, 0, :, :]
                    if False:
                        ax[0].clear()
                        ax[1].clear()
                        if img is None:
                            img = input_spikes[idx, i, 0, :, :]
                        else:
                            img += input_spikes[idx, i, 0, :, :]
                        ax[0].imshow(img)
                        ax[0].title.set_text(classes[labels[idx]])
                        xx = input[idx, 0, i] * 2 - 1
                        yy = input[idx, 1, i] * 2 - 1
                        ax[1].scatter(xx, -1 * yy)
                        ax[1].set_xlim(-1, 1)
                        ax[1].set_ylim(-1, 1)
                        plt.pause(0.0001)
                print("done")
                pos = np.where(im3d == 1)
                fig2 = plt.axes(projection='3d')
                fig2.scatter3D(pos[0], pos[1], pos[2], c=pos[0])
                ys = np.arange(0,512)
                plt.figure()
                plt.plot(np.array(input[idx,0])[0:512],ys, 'g')
                plt.plot(np.array(input[idx,1])[0:512],ys, 'b')
                plt.pause(100)
