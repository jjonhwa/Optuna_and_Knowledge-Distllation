import copy
import tensorly as tl
from tensorly.decomposition import parafac, partial_tucker
from typing import List

import torch
import torch.nn as nn

# switch to the PyTorch backend
tl.set_backend("pytorch")


def tucker_decomposition_conv_layer(
    layer: nn.Module, normed_rank: List[int] = [0.5, 0.5],
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
        layer.weight.data, modes=[0, 1], n_iter_max=2000000, rank=rank, init="svd",
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


def decompose(module: nn.Module):
    """Iterate model layers and decompose"""
    model_layers = list(module.children())
    if not model_layers:
        return None
    for i in range(len(model_layers)):
        if type(model_layers[i]) == nn.Sequential:
            decomposed_module = decompose(model_layers[i])
            if decomposed_module:
                model_layers[i] = decomposed_module
        if type(model_layers[i]) == nn.Conv2d:
            model_layers[i] = tucker_decomposition_conv_layer(model_layers[i])
    return nn.Sequential(*model_layers)

