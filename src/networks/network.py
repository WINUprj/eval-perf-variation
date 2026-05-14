from collections import defaultdict
from functools import partial
from typing import Sequence

import torch
from torch import nn

### Basic block of architectures ###
class FCReLUNetwork(nn.Sequential):
    """
    Non-Linear (ReLU activated) fully connected neural network.
    """

    def __init__(self, layer_sizes: Sequence[int], bias: bool=True) -> None:
        """
        Args:
            layer_sizes (Sequence[int]): Width of each layer.
            bias (bool): Specify whether to add bias terms.
        """
        super(FCReLUNetwork, self).__init__()
        if len(layer_sizes) < 2:
            raise ValueError("At least input and output size must be specified.")

        for l in range(len(layer_sizes) - 1):
            self.add_module(
                f"fc{l + 1}",
                nn.Linear(
                    in_features=layer_sizes[l],
                    out_features=layer_sizes[l + 1],
                    bias=bias,
                ),
            )

            if l < len(layer_sizes) - 2:
                self.add_module(f"act{l + 1}", nn.ReLU())

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()


class FCReLUNetworkWithHook(nn.Sequential):
    """
    Non-Linear (ReLU activated) fully connected neural network with forward hook
    counting the number of dormant neurons.
    """

    def __init__(self, layer_sizes: Sequence[int], bias: bool=True, activate_last: bool=False) -> None:
        """
        Args:
            layer_sizes (Sequence[int]): Width of each layer.
            bias (bool): Specify whether to add bias terms.
            activate_last (bool): Specify whether to activate final layer.
        """
        super(FCReLUNetworkWithHook, self).__init__()
        if len(layer_sizes) < 2:
            raise ValueError("At least input and output size must be specified.")

        self.n_neurons = 0
        offset = 1 if activate_last else 2
        for l in range(len(layer_sizes) - 1):
            self.add_module(
                f"fc{l + 1}",
                nn.Linear(
                    in_features=layer_sizes[l],
                    out_features=layer_sizes[l + 1],
                    bias=bias,
                ),
            )

            if l < len(layer_sizes) - offset:
                self.n_neurons += getattr(self, f"fc{l + 1}").out_features
                self.add_module(f"act{l + 1}", nn.ReLU())

        self.dormant_neurons = defaultdict()
        for name, layer in self.named_modules():
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            elif isinstance(layer, nn.ReLU):
                layer.register_forward_hook(partial(self.count_dormant_neurons, name))

    def count_dormant_neurons(self, name: str, module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
        """
        Count the dormant neuron for single layer.

        Args:
            name (str): Name of the layer.
            module (nn.Module): Torch linear layer module.
            inp (tuple): Input of the given module.
            out (torch.Tensor): Output of the given module.
        """
        self.dormant_neurons[name] = torch.sum(out == 0.0).item()


class LayernormReLUNetworkWithHook(nn.Sequential):
    """
    Non-Linear (ReLU activated) fully connected, layer-normed neural network with forward hook
    counting the number of dormant neurons.
    """

    def __init__(self, layer_sizes: Sequence[int], bias: bool=True, activate_last: bool=False) -> None:
        """
        Args:
            layer_sizes (Sequence[int]): Width of each layer.
            bias (bool): Specify whether to add bias terms.
            activate_last (bool): Specify whether to activate final layer.
        """
        super(LayernormReLUNetworkWithHook, self).__init__()
        if len(layer_sizes) < 2:
            raise ValueError("At least input and output size must be specified.")

        self.n_neurons = 0
        offset = 1 if activate_last else 2
        for l in range(len(layer_sizes) - 1):
            self.add_module(
                f"fc{l + 1}",
                nn.Linear(
                    in_features=layer_sizes[l],
                    out_features=layer_sizes[l + 1],
                    bias=bias,
                ),
            )
            self.add_module(f"ln{l + 1}", nn.LayerNorm(layer_sizes[l + 1]))

            if l < len(layer_sizes) - offset:
                self.n_neurons += getattr(self, f"fc{l + 1}").out_features
                self.add_module(f"act{l + 1}", nn.ReLU())

        self.dormant_neurons = defaultdict()
        for name, layer in self.named_modules():
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            elif isinstance(layer, nn.ReLU):
                layer.register_forward_hook(partial(self.count_dormant_neurons, name))

    def count_dormant_neurons(self, name: str, module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
        """
        Count the dormant neuron for single layer.

        Args:
            name (str): Name of the layer.
            module (nn.Module): Torch linear layer module.
            inp (tuple): Input of the given module.
            out (torch.Tensor): Output of the given module.
        """
        self.dormant_neurons[name] = torch.sum(out == 0.0).item()


class FCTanhNetwork(nn.Sequential):
    """
    Non-Linear (Tanh activated) fully connected neural network.
    """

    def __init__(self, layer_sizes: Sequence[int], bias: bool=True) -> None:
        """
        Args:
            layer_sizes (Sequence[int]): Width of each layer.
            bias (bool): Specify whether to add bias terms.
        """
        super(FCTanhNetwork, self).__init__()
        if len(layer_sizes) < 2:
            raise ValueError("At least input and output size must be specified.")

        for l in range(len(layer_sizes) - 1):
            self.add_module(
                f"fc{l + 1}",
                nn.Linear(
                    in_features=layer_sizes[l],
                    out_features=layer_sizes[l + 1],
                    bias=bias,
                ),
            )

            if l < len(layer_sizes) - 2:
                self.add_module(f"act{l + 1}", nn.Tanh())

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()


class FCTanhNetworkWithHook(nn.Sequential):
    """
    Non-Linear (Tanh activated) fully connected neural network with forward hook
    counting the number of dormant neurons.
    """

    def __init__(self, layer_sizes: Sequence[int], bias: bool=True, activate_last: bool=False) -> None:
        """
        Args:
            layer_sizes (Sequence[int]): Width of each layer.
            bias (bool): Specify whether to add bias terms.
            activate_last (bool): Specify whether to activate final layer.
        """
        super(FCTanhNetworkWithHook, self).__init__()
        if len(layer_sizes) < 2:
            raise ValueError("At least input and output size must be specified.")

        self.n_neurons = 0
        offset = 1 if activate_last else 2
        for l in range(len(layer_sizes) - 1):
            self.add_module(
                f"fc{l + 1}",
                nn.Linear(
                    in_features=layer_sizes[l],
                    out_features=layer_sizes[l + 1],
                    bias=bias,
                ),
            )

            if l < len(layer_sizes) - offset:
                self.n_neurons += getattr(self, f"fc{l + 1}").out_features
                self.add_module(f"act{l + 1}", nn.Tanh())

        self.dormant_neurons = defaultdict()
        for name, layer in self.named_modules():
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
            elif isinstance(layer, nn.Tanh):
                layer.register_forward_hook(partial(self.count_dormant_neurons, name))

    def count_dormant_neurons(self, name: str, module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
        """
        Count the dormant neuron for single layer.

        Args:
            name (str): Name of the layer.
            module (nn.Module): Torch linear layer module.
            inp (tuple): Input of the given module.
            out (torch.Tensor): Output of the given module.
        """
        self.dormant_neurons[name] = torch.sum(
            torch.abs(out) >= 0.95
        ).item()  # Following Elsayed et al. (2024)


class LayernormTanhNetworkWithHook(nn.Sequential):
    """
    Non-Linear (Tanh activated) fully connected, layer-normed neural network with forward hook
    counting the number of dormant neurons.
    """

    def __init__(self, layer_sizes: Sequence[int], bias: bool=True, activate_last: bool=False) -> None:
        """
        Args:
            layer_sizes (Sequence[int]): Width of each layer.
            bias (bool): Specify whether to add bias terms.
            activate_last (bool): Specify whether to activate final layer.
        """
        super(LayernormTanhNetworkWithHook, self).__init__()
        if len(layer_sizes) < 2:
            raise ValueError("At least input and output size must be specified.")

        self.n_neurons = 0
        offset = 1 if activate_last else 2
        for l in range(len(layer_sizes) - 1):
            self.add_module(
                f"fc{l + 1}",
                nn.Linear(
                    in_features=layer_sizes[l],
                    out_features=layer_sizes[l + 1],
                    bias=bias,
                ),
            )
            self.add_module(f"ln{l + 1}", nn.LayerNorm(layer_sizes[l + 1]))

            if l < len(layer_sizes) - offset:
                self.n_neurons += getattr(self, f"fc{l + 1}").out_features
                self.add_module(f"act{l + 1}", nn.Tanh())

        self.dormant_neurons = defaultdict()
        for name, layer in self.named_modules():
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
            elif isinstance(layer, nn.Tanh):
                layer.register_forward_hook(partial(self.count_dormant_neurons, name))

    def count_dormant_neurons(self, name: str, module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
        """
        Count the dormant neuron for single layer.

        Args:
            name (str): Name of the layer.
            module (nn.Module): Torch linear layer module.
            inp (tuple): Input of the given module.
            out (torch.Tensor): Output of the given module.
        """
        self.dormant_neurons[name] = torch.sum(
            torch.abs(out) >= 0.95
        ).item()  # Following Elsayed et al. (2024)
