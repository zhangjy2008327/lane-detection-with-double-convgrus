#!/usr/bin/evn python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 - zihao.chen <zihao.chen@moji.com>
'''
Author: zihao.chen
Create Date: 2018-03-29
Modify Date: 2018-03-29
descirption: ""
'''
import torch
from torch import nn
import torch.nn.functional as f
import numpy as np
from torch.autograd import Variable


class ConvGRUCell(nn.Module):
    #input_size=(8, 16), hidden_size=512, kernel_size=(3,3)
    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.height, self.width = 8, 16
        self.dropout = nn.Dropout(p=0.5)
        # print('kernal-----', kernel_size[0], kernel_size[1])
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.ConvGates = nn.Conv2d(self.input_size + self.hidden_size,
                                   2 * self.hidden_size,
                                   self.kernel_size,
                                   padding=self.padding
                                   )
        self.Conv_ct = nn.Conv2d(self.input_size + self.hidden_size, self.hidden_size, self.kernel_size,
                                 padding=self.padding)
        # dtype = torch.FloatTensor

        #input = [5, 3, 512, 8, 16], hidden=
    def forward(self, input, hidden):
        if hidden is None:
            # print (input.data.size()[0])
            # print (self.hidden_size)
            # print (list(input.data.size()[2:]))
            size_h = [input.data.size()[0], self.hidden_size] + list(input.data.size()[2:])
            # print size_h
            hidden = Variable(torch.zeros(size_h).cuda())
        if input is None:
            # print (input.data.size()[0])
            # print (self.hidden_size)
            # print (list(input.data.size()[2:]))
            size_h = [hidden.data.size()[0], self.input_size] + list(hidden.data.size()[2:])
            # print size_h
            input = Variable(torch.zeros(size_h).cuda())
        # print input.size()
        # print hidden.size()
        c1 = self.ConvGates(torch.cat((input, hidden), 1))
        # print('c1-----', c1.shape)
        (rt, ut) = c1.chunk(2, 1)
        # print('rt-----', rt.shape)
        # print('ut-----', ut.shape)
        # reset_gate = self.dropout(f.sigmoid(rt))
        # update_gate = self.dropout(f.sigmoid(ut))

        reset_gate = self.dropout(torch.sigmoid(rt))
        update_gate = self.dropout(torch.sigmoid(ut))

        # print('reset_gate-----', reset_gate.shape)
        # print('update_gate-----', update_gate.shape)
        # print('hidden----', hidden.shape)
        gated_hidden = torch.mul(reset_gate, hidden)
        p1 = self.Conv_ct(torch.cat((input, gated_hidden), 1))
        # ct = f.tanh(p1)
        #ct = h(t-1)
        ct = torch.tanh(p1)
        #ht
        next_h = torch.mul(update_gate, hidden) + (1 - update_gate) * ct
        # print('next_h----', next_h.shape)
        return next_h, 0

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_size, self.height, self.width).cuda(),
                torch.zeros(batch_size, self.hidden_size, self.height, self.width).cuda())

class ConvGRU(nn.Module):
    # input_size=(8,16), input_dim=512, hidden_dim=[512, 512], kernel_size=(3,3), num_layers=2
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvGRU, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')
        # input_size = (8, 16) is a tuple, notate the useful method to sign values for height, and width
        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            # to prepare all parameters for every layer, and then construct a object to be called when needed/required
            cell_list.append(ConvGRUCell(input_size=self.input_dim,
                                          #input_dim=cur_input_dim,
                                         hidden_size= self.hidden_dim[i],
                                          #hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i])
                                          #bias=self.bias)
                                         )

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output

        -------
        here data = [5, 1, 512, 8, 16]= input_tensor is passed into ConvLSTM
        5 is number of times, each time is one batch, as 1,
        and each batch_size size has 512 feature maps, each feature map size is 8x16
        """
        # print('input_tensor---', input_tensor.shape)
        # input_tensor = [5, 3, 512, 8, 16]
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # input_tensor.shape = [3, 5, 512, 8, 16]
        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # input_tensor.size(0)=1
            # print('input_tensor.size(0)--', input_tensor.size(0))
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        # seq_len = 5, actually is 5 feature map, each size =
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        #initilize hidden_state only once
        h, c = hidden_state[layer_idx]
        for layer_idx in range(self.num_layers):

            # initilize hidden_state
            h, c = hidden_state[layer_idx]
            # h = c size = [3, 512, 8, 16]
            # print('h, c,---', h.shape, c.shape)
            output_inner = []
            output_inner1 = []
            final_out = []
            for t in range(seq_len):
                # change to run forward of ConvLSTMCell, after input_tensor=cur_layer_input[:, t, :, :, :]
                # input_tensor shape is [3, 512, 8, 16]
                # h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                #                                  cur_state=[h, c])
                # hl = torch.cat((h, c), dim=1)
                h, c = self.cell_list[layer_idx](input=cur_layer_input[:, t, :, :, :],
                                                 hidden=h)
                # h = h
                # h1, c1 = self.cell_list[layer_idx](input=cur_layer_input[:, t, :, :, :],
                #                                  hidden=h)

                # aa = torch.sum(h1)
                # h = h - h1 / abs(aa)
                # print('h------', aa)
                # print('h, c----', h.shape, c.shape)
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            # print('1--------', layer_output.shape)
            cur_layer_input = layer_output
            # print('2--------', cur_layer_input.shape)
            layer_output = layer_output.permute(1, 0, 2, 3, 4)
            # print('3-------', layer_output.shape)
            layer_output_list.append(layer_output)
            # print('4-------', layer_output_list)
            last_state_list.append([h, c])
            # print('5-------', last_state_list)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            # print('6-------', layer_output_list)
            last_state_list = last_state_list[-1:]
            # print('7-------', last_state_list)

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    """
    check the type of kernel_size to be the same

    """

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    """
    if kerner_size = (3,3), num_layers = 2, then _extend_for_multilayer() will make 
    kernal_size = [(3,3), (3,3)] for each layer
    """

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param