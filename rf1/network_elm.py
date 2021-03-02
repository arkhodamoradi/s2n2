import torch.nn as nn
import math
import numpy as np
import torch.utils.model_zoo as model_zoo
import torch
from torch.autograd import Variable

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)
# thresh = 0.2  # neuronal threshold (org: 0.5)
lens = 0.05  # hyper-parameters of approximate function (org: 5/3)
decay = 0.9  # decay constants (org: 0.8)


# approximate firing function
class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        _input = args[0]
        threshold = args[1]
        ctx.threshold = threshold
        ctx.save_for_backward(_input)
        return _input.gt(threshold).float()

    @staticmethod
    def backward(ctx, *grad_output):
        _input, = ctx.saved_tensors
        threshold = ctx.threshold
        grad_input = grad_output[0].clone()
        temp = torch.exp(-(_input - threshold) ** 2 / (2 * lens ** 2)) / (
                (2 * lens * np.pi) ** 0.5)  # TODO: explain?
        return grad_input * temp.float(), None  # to match the number of inputs in forward()


# membrane potential update
class Synapse(object):
    def __init__(self, threshold, reset=True):
        super(Synapse, self).__init__()
        self.threshold = threshold
        self.reset = reset

    def mem_update(self, x, mem, spk):
        if self.reset:
            mem = mem * decay * (1. - spk) + x    # with rest mechanism
        else:
            mem = mem * decay + x                 # no rest mechanism
        spk = ActFun.apply(mem, self.threshold)  # act_fun: approximation firing function
        return spk, mem, spk


class SpikingResNet(nn.Module):
    def __init__(self, input_size, synapse_reset, threshs, nb_classes=24):
        super(SpikingResNet, self).__init__()

        self.nb_classes = nb_classes
        [b_s, _, _ch, _w, _h] = input_size
        self.batch_size = b_s
        self.ch = _ch
        self.w = _w
        self.h = _h
        self.synapse_reset = synapse_reset

        self.conv1_custom = nn.Conv2d(self.ch, 64, kernel_size=7, stride=1, padding=1, bias=False)  # org: (stride=2) with avgpool
        self.synapse0 = Synapse(threshold=threshs[0], reset=self.synapse_reset)

        self.custom_pool1 = nn.Conv2d(64, 32, kernel_size=7, bias=False)
        self.synapse1 = Synapse(threshold=threshs[1], reset=self.synapse_reset)

        self.custom_pool2 = nn.Conv2d(32, 4, kernel_size=3, bias=False)
        self.synapse2 = Synapse(threshold=threshs[2], reset=self.synapse_reset)

        self.fc_custom = nn.Linear(64, nb_classes, bias=False)
        self.synapse3 = Synapse(threshold=threshs[3], reset=self.synapse_reset)

        # --- init variables and create memory for the SNN
        self.c_spk = self.c_mem = None
        self.cp1_spk = self.cp1_mem = None
        self.cp2_spk = self.cp2_mem = None
        self.fc_spk = self.fc_mem = None
        self.fc_sumspike = None

    def forward(self, _input, time_window, reset):
        if reset:
            # --- init variables and create memory for the SNN
            self.c_spk = self.c_mem = torch.zeros(self.batch_size, 64, 12, 12, device=device)
            self.cp1_spk = self.cp1_mem = torch.zeros(self.batch_size, 32, 6, 6, device=device)
            self.cp2_spk = self.cp2_mem = torch.zeros(self.batch_size, 4, 4, 4, device=device)
            self.fc_spk = self.fc_mem = torch.zeros(self.batch_size, self.nb_classes, device=device)  # //2 due to stride=2
        else:
            self.c_mem = self.c_mem.detach()
            self.c_spk = self.c_spk.detach()

            self.cp1_mem = self.cp1_mem.detach()
            self.cp1_spk = self.cp1_spk.detach()

            self.cp2_mem = self.cp2_mem.detach()
            self.cp2_spk = self.cp2_spk.detach()

            self.fc_mem = self.fc_mem.detach()
            self.fc_spk = self.fc_spk.detach()

        self.fc_sumspike = torch.zeros(self.batch_size, self.nb_classes, device=device)

        # --- main SNN window
        r = [0] * 10
        for step in range(time_window):
            x = _input[:, step, :, :, :]

            x = self.conv1_custom(x)
            x, self.c_mem, self.c_spk = self.synapse0.mem_update(x, self.c_mem, self.c_spk)
            r[0] += x.nonzero().size(0) / x.numel()

            x = self.custom_pool1(x)
            x, self.cp1_mem, self.cp1_spk = self.synapse1.mem_update(x, self.cp1_mem, self.cp1_spk)
            r[1] += x.nonzero().size(0) / x.numel()

            x = self.custom_pool2(x)
            x, self.cp2_mem, self.cp2_spk = self.synapse2.mem_update(x, self.cp2_mem, self.cp2_spk)
            r[2] += x.nonzero().size(0) / x.numel()

            x = x.view(x.size(0), -1)
            x = self.fc_custom(x)
            x, self.fc_mem, self.fc_spk = self.synapse3.mem_update(x, self.fc_mem, self.fc_spk)
            r[3] += self.fc_spk.nonzero().size(0) / self.fc_spk.numel()

            self.fc_sumspike += self.fc_spk
        self.fc_sumspike = self.fc_sumspike / time_window
        return self.fc_sumspike, np.array(r)/time_window


def spiking_resnet_18(input_size, synapse_reset, threshs, nb_classes=101):
    model = SpikingResNet(input_size, synapse_reset, threshs, nb_classes)
    return model

