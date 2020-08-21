import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        # print('double_conv', x.shape)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        # print('inconv---', x.shape)
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

#
class Interpolate(nn.Module):
    def __init__(self, scaler_size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        # self.size = size
        self.scaler = scaler_size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scaler, mode=self.mode)
        return x



class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
            # self.up = Interpolate(scaler_size=2, mode='bilinear')
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        # print('x1,x2----------', x1.shape, x2.shape)
        x1 = self.up(x1)
        # print('110--------------', x1.shape)
        # exit(0)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        # print('x2--x1---', x2.shape, x1.shape)

        x = torch.cat([x2, x1], dim=1)
        # print('x2 and x1------combi-----', x.shape)
        x = self.conv(x)
        # print('x2 and x1------cov-----', x.shape)
        # print('\n')
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        batch_size: none
            because ConvLSTMCell is called by other blocks, so batch_size is passed into it when called by calling
            ConvLSTMCell object instances

        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        # print('h_cur----', h_cur.shape)
        # print('input_tensor---', input_tensor.shape)
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        # print('combined---', combined.shape)
        #combined shape = [1, 1024, 8, 16]
        combined_conv = self.conv(combined)
        # print('combined_conv---', combined_conv.shape)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        # print('cc_i---', cc_i.shape)
        # print('cc_f---', cc_f.shape)
        # print('cc_o---', cc_o.shape)
        # print('cc_g---', cc_g.shape)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_dim, self.height, self.width).cuda(),
                torch.zeros(batch_size, self.hidden_dim, self.height, self.width).cuda())


class ConvLSTM(nn.Module):
    #input_size=(8,16), input_dim=512, hidden_dim=512, kernel_size=(3,3), num_layers=2
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')
        #input_size = (8, 16) is a tuple, notate the useful method to sign values for height, and width
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
            #to prepare all parameters for every layer, and then construct a object to be called when needed/required
            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

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
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor=input_tensor.permute(1, 0, 2, 3, 4)

        #input_tensor.shape = [3, 5, 512, 8, 16]
        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            #input_tensor.size(0)=1
            # print('input_tensor.size(0)--', input_tensor.size(0))
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        #seq_len = 5, actually is 5 feature map, each size =
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            # initilize hidden_state
            h, c = hidden_state[layer_idx]
            #h = c size = [1, 512, 8, 16]
            # print('h, c,---', h.shape, c.shape)
            output_inner = []
            for t in range(seq_len):
                #change to run forward of ConvLSTMCell, after input_tensor=cur_layer_input[:, t, :, :, :]
                #input_tensor shape is [1, 512, 8, 16]
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                # print('h, c----', h.shape, c.shape)
                output_inner.append(h)
            #stack 5 [3, 512, 8, 16], so '3' as dim=1, in original list, '3' is dim=0
            layer_output = torch.stack(output_inner, dim=1)
            # print('layer_output----', layer_output.shape)
            cur_layer_input = layer_output

            layer_output = layer_output.permute(1, 0, 2, 3, 4)

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

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






