__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np


from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_layers import series_decomp
from layers.FBP_backbone import backbone_new_NLinear
import math
from torch.fft import rfft, irfft
class Model(nn.Module):
    def __init__(self, configs, verbose:bool=False):
        super().__init__()   
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        kernel_size = configs.kernel_size
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        individual = configs.individual
        head_dropout = configs.head_dropout
        # model
        self.decomposition = configs.decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.linear=nn.Linear(context_window,target_window)
            self.model_trend = backbone_new_NLinear(c_in=c_in, context_window = context_window, target_window=target_window,head_dropout=head_dropout, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose)
            self.model_res = backbone_new_NLinear(c_in=c_in, context_window = context_window, target_window=target_window,head_dropout=head_dropout, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose)
        else:
            self.model = backbone_new_NLinear(c_in=c_in, context_window = context_window, target_window=target_window,head_dropout=head_dropout, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose)
                                  

    def forward(self, x):           # x: [Batch, Input length, Channel]
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        return x

