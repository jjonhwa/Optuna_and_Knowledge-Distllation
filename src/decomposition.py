import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker
import numpy as np
import torch
import torch.nn as nn
from typing import *


def cp_decomposition_conv_layer(layer, rank):
    # rank = max(layer.weight.data.numpy().shape) // 3
    """Gets a conv layer and a target rank,
    returns a nn.Sequential object with the decomposition"""
    # Perform CP decomposition on the layer weight tensorly.
    last, first, vertical, horizontal = parafac(layer.weight.data, rank=rank, init="svd").factors

    pointwise_s_to_r_layer = torch.nn.Conv2d(
        in_channels=first.shape[0],
        out_channels=first.shape[1],
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=False,
    )

    depthwise_vertical_layer = torch.nn.Conv2d(
        in_channels=vertical.shape[1],
        out_channels=vertical.shape[1],
        kernel_size=(vertical.shape[0], 1),
        stride=1,
        padding=(layer.padding[0], 0),
        dilation=layer.dilation,
        groups=vertical.shape[1],
        bias=False,
    )

    depthwise_horizontal_layer = torch.nn.Conv2d(
        in_channels=horizontal.shape[1],
        out_channels=horizontal.shape[1],
        kernel_size=(1, horizontal.shape[0]),
        stride=layer.stride,
        padding=(0, layer.padding[0]),
        dilation=layer.dilation,
        groups=horizontal.shape[1],
        bias=False,
    )

    pointwise_r_to_t_layer = torch.nn.Conv2d(
        in_channels=last.shape[1],
        out_channels=last.shape[0],
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=True,
    )
    if layer.bias is not None:
        pointwise_r_to_t_layer.bias.data = layer.bias.data
    depthwise_horizontal_layer.weight.data = (
        torch.transpose(horizontal, 1, 0).unsqueeze(1).unsqueeze(1)
    )
    depthwise_vertical_layer.weight.data = (
        torch.transpose(vertical, 1, 0).unsqueeze(1).unsqueeze(-1)
    )
    pointwise_s_to_r_layer.weight.data = torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    pointwise_r_to_t_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)

    new_layers = [
        pointwise_s_to_r_layer,
        depthwise_vertical_layer,
        depthwise_horizontal_layer,
        pointwise_r_to_t_layer,
    ]

    return nn.Sequential(*new_layers)


def tucker_decomposition_conv_layer(
    layer: nn.Module,
    normed_rank: List[int] = [0.5, 0.5],
) -> nn.Module:
    """Gets a conv layer,
    returns a nn.Sequential object with the Tucker decomposition.
    The ranks are estimated with a Python implementation of VBMF
    https://github.com/CasvandenBogaard/VBMF
    """
    if hasattr(layer, "rank"):
        normed_rank = getattr(layer, "rank")
    rank = [
        int(r * layer.weight.shape[i]) for i, r in enumerate(normed_rank)
    ]  # output channel * normalized rank
    rank = [max(r, 2) for r in rank]

    core, [last, first] = partial_tucker(
        layer.weight.data,
        modes=[0, 1],
        n_iter_max=2000000,
        rank=rank,
        init="svd",
    )

    # A pointwise convolution that reduces the channels from S to R3
    first_layer = nn.Conv2d(
        in_channels=first.shape[0],
        out_channels=first.shape[1],
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=False,
    )

    # A regular 2D convolution layer with R3 input channels
    # and R3 output channels
    core_layer = nn.Conv2d(
        in_channels=core.shape[1],
        out_channels=core.shape[0],
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        bias=False,
    )

    # A pointwise convolution that increases the channels from R4 to T
    last_layer = nn.Conv2d(
        in_channels=last.shape[1],
        out_channels=last.shape[0],
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=layer.dilation,
        bias=True,
    )

    if hasattr(layer, "bias") and layer.bias is not None:
        last_layer.bias.data = layer.bias.data

    first_layer.weight.data = torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core

    new_layers = [first_layer, core_layer, last_layer]
    return nn.Sequential(*new_layers)


def cp_decompose_model(model, exclude_first_conv=True, passed_first_conv=False):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = cp_decompose_model(module, exclude_first_conv, passed_first_conv)
        elif type(module) == nn.Conv2d:
            if passed_first_conv is False:
                passed_first_conv = True
                if exclude_first_conv is True:
                    continue

            conv_layer = module
            rank = rank = max(conv_layer.weight.data.numpy().shape) // 3
            print(conv_layer, "CP Estimated rank", rank)

            decomposed = cp_decomposition_conv_layer(conv_layer, rank)

            model._modules[name] = decomposed
    return model


def tucker_decompose_model(model, exclude_first_conv=True, passed_first_conv=False):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = tucker_decompose_model(
                module, exclude_first_conv, passed_first_conv
            )
        elif type(module) == nn.Conv2d:
            if passed_first_conv is False:
                passed_first_conv = True
                if exclude_first_conv is True:
                    continue

            conv_layer = module
            rank = [0.5, 0.5]
            print(conv_layer, "Tucker Estimated ranks", rank)

            decomposed = tucker_decomposition_conv_layer(conv_layer, rank)

            model._modules[name] = decomposed
    return model
