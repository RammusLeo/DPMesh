import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def conv_init(conv):
    nn.init.kaiming_normal(conv.weight, mode='fan_out')
    nn.init.constant(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant(bn.weight, scale)
    nn.init.constant(bn.bias, 0)

class Shift_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3):
        super(Shift_gcn, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.Linear_weight = nn.Parameter(torch.zeros(in_channels, out_channels, requires_grad=True, device='cuda'), requires_grad=True)
        nn.init.normal_(self.Linear_weight, 0,math.sqrt(1.0/out_channels))

        self.Linear_bias = nn.Parameter(torch.zeros(1,1,out_channels,requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant(self.Linear_bias, 0)

        self.Feature_Mask = nn.Parameter(torch.ones(1,23,in_channels, requires_grad=True,device='cuda'),requires_grad=True)
        nn.init.constant(self.Feature_Mask, 0)

        self.bn = nn.BatchNorm1d(23*out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm1d):
                bn_init(m, 1)

        index_array = np.empty(23*in_channels).astype(np.int)
        for i in range(23):
            for j in range(in_channels):
                index_array[i*in_channels + j] = (i*in_channels + j + j*in_channels)%(in_channels*23)
        self.shift_in = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)

        index_array = np.empty(23*out_channels).astype(np.int)
        for i in range(23):
            for j in range(out_channels):
                index_array[i*out_channels + j] = (i*out_channels + j - j*out_channels)%(out_channels*23)
        self.shift_out = nn.Parameter(torch.from_numpy(index_array),requires_grad=False)
        

    def forward(self, x0):
        n, c, v = x0.size()
        x = x0.permute(0,2,1).contiguous()

        # shift1
        x = x.view(n,v*c)
        x = torch.index_select(x, 1, self.shift_in)
        x = x.view(n,v,c)
        x = x * (torch.tanh(self.Feature_Mask)+1)

        x = torch.einsum('nwc,cd->nwd', (x, self.Linear_weight)).contiguous() # nt,v,c
        x = x + self.Linear_bias

        # shift2
        x = x.view(n,-1) 
        x = torch.index_select(x, 1, self.shift_out)
        x = self.bn(x)
        x = x.view(n,v,self.out_channels).permute(0,2,1) # n,c,t,v
        x = x + self.down(x0)

        x = self.relu(x)
        return x

class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = Shift_gcn(in_channels, out_channels, A)
        self.relu = nn.ReLU()

        if not residual or (in_channels != out_channels):
            self.residual = lambda x: 0

        # elif (in_channels == out_channels) and (stride == 1)
        else:
            # self.residual = tcn(in_channels, out_channels, kernel_size=1, stride=stride)
            self.residual = lambda x: x

    def forward(self, x):
        x = self.gcn1(x) + self.residual(x)
        return self.relu(x)


class ShiftGCN_Encoder(nn.Module):
    def __init__(self, num_point=23, num_person=1, graph=None, graph_args=dict(), in_channels=3):
        super(ShiftGCN_Encoder, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            # Graph = import_class(graph)
            # self.graph = Graph(**graph_args)
            self.graph = graph

        A = self.graph.A
        # self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 64, A)
        self.l4 = TCN_GCN_unit(64, 64, A)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A)
        self.l7 = TCN_GCN_unit(128, 128, A)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A)
        self.l10 = TCN_GCN_unit(256, 256, A)

        # bn_init(self.data_bn, 1)

    def forward(self, x):
        # N, C, V = x.size()

        # x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, V * C, T)
        # x = self.data_bn(x)
        # x = x.view(N, V, C, T).permute(0, 2, 3,1).contiguous()
        # x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        # c_new = x.size(1)
        # x = x.view(N, M, c_new, -1)
        # x = x.mean(3).mean(1)
        # N, C, V
        return x


class ShiftGCN_Decoder(nn.Module):
    def __init__(self, num_point=23, num_person=1, graph=None, graph_args=dict(), in_channels=3):
        super(ShiftGCN_Decoder, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            # Graph = import_class(graph)
            # self.graph = Graph(**graph_args)
            self.graph = graph

        A = self.graph.A
        # self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # self.l1 = TCN_GCN_unit(3, 64, A, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A)
        self.l3 = TCN_GCN_unit(64, 64, A)
        self.l4 = TCN_GCN_unit(64, 32, A, stride=2)
        self.l5 = TCN_GCN_unit(32, 32, A)
        self.l6 = TCN_GCN_unit(32, 32, A)
        # self.l7 = TCN_GCN_unit(64, 128, A, stride=2)
        # self.l8 = TCN_GCN_unit(64, 64, A)
        # self.l9 = TCN_GCN_unit(64, 64, A)
        # self.l10 = TCN_GCN_unit(64, 64, A)


        # bn_init(self.data_bn, 1)

    def forward(self, x):
        # N, C, V = x.size()

        # x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, V * C, T)
        # x = self.data_bn(x)
        # x = x.view(N, V, C, T).permute(0, 2, 3,1).contiguous()
        # x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        # x = self.l7(x)
        # x = self.l8(x)
        # x = self.l9(x)
        # x = self.l10(x)

        # N*M,C,T,V
        # c_new = x.size(1)
        # x = x.view(N, M, c_new, -1)
        # x = x.mean(3).mean(1)
        # N, C, V
        return x