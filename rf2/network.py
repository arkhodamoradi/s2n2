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
lens = 0.5/3  # hyper-parameters of approximate function (org: 5/3), 0.16
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
        temp = torch.exp(-(_input - threshold) ** 2 / (2 * (lens ** 2))) / ((2 * lens * np.pi) ** 0.5)  # TODO: explain?
        #temp = torch.exp(-(_input - threshold) ** 2 / (2 * lens)) / ((2 * lens * np.pi) ** 0.5)  # 26 at https://www.frontiersin.org/articles/10.3389/fnins.2018.00331/full
        #temp = (1/lens) * torch.exp((threshold - _input) / lens) / ((1+torch.exp((threshold - _input)))**2)  # 25 at https://www.frontiersin.org/articles/10.3389/fnins.2018.00331/full
        #temp[torch.isnan(temp)] = 0.001
        #temp[torch.isinf(temp)] = 0.1
        return grad_input * temp.float(), None  # to match the number of inputs in forward()


ActFunn = torch.nn.ReLU6()
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
        # spk = ActFunn(mem)
        return spk, mem, spk


class SpikingBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, thresh, synapse_reset, stride=1):
        super(SpikingBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.drop1 = nn.Dropout(0.6)   # This brings imbalacement
        # self.drop2 = nn.Dropout(0.2)
        self.synapse = Synapse(threshold=thresh, reset=synapse_reset)

    def forward(self, x, c1_mem, c2_mem, c1_spk, c2_spk):
        residual = x  # batch_size, 64, 32, 32
        x = self.conv1(x)  # batch_size, 64, 32, 32
        c1_mem, c1_spk = self.synapse.mem_update(x, c1_mem, c1_spk)
        x = self.conv2(c1_spk)  # batch_size, 64, 32, 32

        x += residual

        c2_mem, c2_spk = self.synapse.mem_update(x, c2_mem, c2_spk)
        # x = self.drop2(x)
        return c2_spk, c1_mem, c2_mem, c1_spk, c2_spk


class SpikingResNet(nn.Module):
    def __init__(self, block, input_size, synapse_reset, threshs, nb_classes=24):
        super(SpikingResNet, self).__init__()

        self.block_expansion = 1
        self.nb_classes = nb_classes
        [b_s, _, _ch, _w, _h] = input_size
        self.batch_size = b_s
        self.ch = _ch
        self.w = _w
        self.h = _h
        self.synapse_reset = synapse_reset

        self.size_devide = np.array([4, 4, 4, 4])
        self.planes = [64] * 4
        self.out1plane = 64
        self.num_sl = 8

        self.conv1_custom = nn.Conv2d(self.ch, self.out1plane, kernel_size=5, stride=1, padding=2, bias=False)  # org: (stride=2) with avgpool TODO: reduce stride
        self.synapse0 = Synapse(threshold=threshs[0], reset=self.synapse_reset)

        self.custom_pool1 = nn.Conv2d(64, 64, kernel_size=5, bias=False)
        self.synapse1 = Synapse(threshold=threshs[1], reset=self.synapse_reset)

        self.custom_pool2 = nn.Conv2d(64, 128, kernel_size=3, bias=False)
        self.synapse2 = Synapse(threshold=threshs[2], reset=self.synapse_reset)

        self.custom_pool22 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.synapse22 = Synapse(threshold=threshs[3], reset=self.synapse_reset)

        #self.custom_pool23 = nn.Conv2d(128, 32, kernel_size=3, padding=1, bias=False)
        #self.synapse23 = Synapse(threshold=threshs[3], reset=self.synapse_reset)

        self.fc_custom1 = nn.Linear(12800, 1024, bias=False)
        self.synapse3 = Synapse(threshold=threshs[4], reset=self.synapse_reset)

        self.fc_custom2 = nn.Linear(1024, nb_classes, bias=False)
        self.synapse4 = Synapse(threshold=threshs[5], reset=self.synapse_reset)

        # --- init variables and create memory for the SNN
        self.c_spk = self.c_mem = None
        self.c1_spk = self.c1_mem = None
        self.c2_spk = self.c2_mem = None
        self.c22_spk = self.c22_mem = None
        #self.c23_spk = self.c23_mem = None
        self.fc1_spk = self.fc1_mem = None
        self.fc2_spk = self.fc2_mem = None
        self.fc_sumspike = None

    def forward(self, _input, time_window, reset):
        if reset:
            # --- init variables and create memory for the SNN
            self.c_spk = self.c_mem = torch.zeros(self.batch_size, self.out1plane, 16, 16, device=device)
            self.c1_spk = self.c1_mem = torch.zeros(self.batch_size, 64, 12, 12, device=device)
            self.c2_spk = self.c2_mem = torch.zeros(self.batch_size, 128, 10, 10, device=device)
            self.c22_spk = self.c22_mem = torch.zeros(self.batch_size, 128, 10, 10, device=device)
            #self.c23_spk = self.c23_mem = torch.zeros(self.batch_size, 32, 8, 8, device=device)
            self.fc1_spk = self.fc1_mem = torch.zeros(self.batch_size, 1024, device=device)
            self.fc2_spk = self.fc2_mem = torch.zeros(self.batch_size, self.nb_classes, device=device)
            self.fc_sumspike = torch.zeros(self.batch_size, self.nb_classes, device=device)
        else:
            self.c_mem = self.c_mem.detach()
            self.c_spk = self.c_spk.detach()
            self.c1_mem = self.c1_mem.detach()
            self.c1_spk = self.c1_spk.detach()
            self.c2_mem = self.c2_mem.detach()
            self.c2_spk = self.c2_spk.detach()
            self.c22_mem = self.c22_mem.detach()
            self.c22_spk = self.c22_spk.detach()
            #self.c23_mem = self.c23_mem.detach()
            #self.c23_spk = self.c23_spk.detach()
            self.fc1_mem = self.fc1_mem.detach()
            self.fc1_spk = self.fc1_spk.detach()
            self.fc2_mem = self.fc2_mem.detach()
            self.fc2_spk = self.fc2_spk.detach()
            self.fc_sumspike = self.fc_sumspike.detach()

        #self.fc_sumspike = torch.zeros(self.batch_size, self.nb_classes, device=device)

        # --- main SNN window
        r = [0] * 10
        for step in range(time_window):
            x = _input[:, step, :, :, :]

            x = self.conv1_custom(x)
            x, self.c_mem, self.c_spk = self.synapse0.mem_update(x, self.c_mem, self.c_spk)
            r[0] += x.nonzero().size(0) / x.numel()

            x = self.custom_pool1(x)
            x, self.c1_mem, self.c1_spk = self.synapse1.mem_update(x, self.c1_mem, self.c1_spk)
            r[1] += x.nonzero().size(0) / x.numel()

            x = self.custom_pool2(x)
            x, self.c2_mem, self.c2_spk = self.synapse2.mem_update(x, self.c2_mem, self.c2_spk)
            r[2] += x.nonzero().size(0) / x.numel()

            residual = x
            x = self.custom_pool22(x)
            x, self.c22_mem, self.c22_spk = self.synapse22.mem_update(x, self.c22_mem, self.c22_spk)
            r[3] += x.nonzero().size(0) / x.numel()

            #x = self.custom_pool23(x)
            #x += residual
            #x, self.c23_mem, self.c23_spk = self.synapse23.mem_update(x, self.c23_mem, self.c23_spk)
            #r[4] += x.nonzero().size(0) / x.numel()

            x = x.view(x.size(0), -1)

            x = self.fc_custom1(x)
            x, self.fc1_mem, self.fc1_spk = self.synapse3.mem_update(x, self.fc1_mem, self.fc1_spk)
            r[5] += x.nonzero().size(0) / x.numel()

            x = self.fc_custom2(x)
            x, self.fc2_mem, self.fc2_spk = self.synapse4.mem_update(x, self.fc2_mem, self.fc2_spk)
            r[6] += x.nonzero().size(0) / x.numel()

            self.fc_sumspike += x
        self.fc_sumspike = self.fc_sumspike / time_window
        return self.fc_sumspike, np.array(r)/time_window


def spiking_resnet_18(input_size, synapse_reset, threshs, nb_classes=101):
    model = SpikingResNet(SpikingBasicBlock, input_size, synapse_reset, threshs, nb_classes)
    return model

