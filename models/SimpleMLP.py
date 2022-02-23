import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .new_relu import *
from .spiking_layer import *
from .settings import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleMLP(nn.Module):
    def __init__(self, modify):
        super(SimpleMLP, self).__init__()
        height = 32
        width = 32
        channels = 3
        self.fc0 = nn.Linear(in_features=height*width*channels, out_features=1024) 
        self.fc1 = nn.Linear(in_features=1024, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=10)

        self.init_epoch = args.init_epoch
        self.relu = th_shift_ReLU(args.shift_relu, modify)

        self.max_active = [0] * 4

    def renew_max(self, x, y, epoch):
        # if epoch > self.init_epoch:
        x = max(x, y)
        return x

    def forward(self, x, epoch):
        output = x.view(x.size(0), -1)
        output = self.fc0(output)
        output = self.relu(output)
        self.max_active[0] = self.renew_max(self.max_active[0], output.max(), epoch)
        output = self.fc1(output)
        output = self.relu(output)
        self.max_active[1] = self.renew_max(self.max_active[1], output.max(), epoch)
        output = self.fc2(output)
        output = self.relu(output)
        self.max_active[2] = self.renew_max(self.max_active[2], output.max(), epoch)
        output = self.fc3(output)
        self.max_active[3] = self.renew_max(self.max_active[3], output.max(), epoch)
        return output

    def record(self):
        return np.array(torch.tensor(self.max_active).cpu())

    def load_max_active(self, mat):
        self.max_active = mat


class SimpleMLP_spiking(nn.Module):
    def __init__(self, thresh_list, model):
        super(SimpleMLP_spiking, self).__init__()

        height = 32
        width = 32
        channels = 3

        self.fc0 = SPIKE_layer(thresh_list[0], model.fc0)
        self.fc1 = SPIKE_layer(thresh_list[1], model.fc1)
        self.fc2 = SPIKE_layer(thresh_list[2], model.fc2)
        self.fc3 = SPIKE_layer(thresh_list[3], model.fc3)

        self.T = args.T

    def init_layer(self):
        self.fc0.init_mem()
        self.fc1.init_mem()
        self.fc2.init_mem()
        self.fc3.init_mem()

    def forward(self, x):
        self.init_layer()
        with torch.no_grad():
            out_spike_sum = 0
            result_list = []
            for time in range(self.T):
                spike_input = x
                output = spike_input.view(x.size(0), -1)
                output = self.fc0(output)
                output = self.fc1(output)
                output = self.fc2(output)
                output = self.fc3(output)
                out_spike_sum += output
                if (time + 1) % args.step == 0:
                    sub_result = out_spike_sum / (time + 1)
                    result_list.append(sub_result)
        return result_list
