"""
Weight Generator Modules - Specialized networks for generating different weight types.
Each generator produces specific parameter structures for ARCCompressor components.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Callable, Union


class LinearWeightGenerator(nn.Module):
    """
    Generates weights and biases for linear layers.

    Args:
        z_dim (int): Input embedding dimension
        output_features (int): Output feature dimension
        hidden_dim (int): Hidden layer dimension for generator MLP
    """

    def __init__(self, z_dim: int, output_features: int, hidden_dim: int = 128):
        super(LinearWeightGenerator, self).__init__()
        self.z_dim = z_dim
        self.output_features = output_features

        # Simple 2-layer MLP as in Ha et al.
        self.generator = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_features)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate weight chunk from embedding."""
        return self.generator(z)


class MLPWeightGenerator(nn.Module):
    """
    Generates weights for multi-layer perceptrons.
    Produces weight matrices and bias vectors for each layer.

    Args:
        z_dim (int): Input embedding dimension
        layer_dims (list): List of layer dimensions [in_dim, hidden_dims..., out_dim]
    """

    def __init__(self, z_dim: int, layer_dims: List[int]):
        super(MLPWeightGenerator, self).__init__()
        self.z_dim = z_dim
        self.layer_dims = layer_dims

        # Calculate total parameters needed
        self.total_params = 0
        for i in range(len(layer_dims) - 1):
            self.total_params += layer_dims[i] * layer_dims[i+1]  # Weights
            self.total_params += layer_dims[i+1]  # Biases

        # Generator network
        self.generator = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.total_params)
        )

    def forward(self, z: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        """
        Generate all MLP weights.

        Returns:
            dict: {'weights': [W1, W2, ...], 'biases': [b1, b2, ...]}
        """
        params = self.generator(z)

        weights = []
        biases = []
        offset = 0

        for i in range(len(self.layer_dims) - 1):
            in_dim = self.layer_dims[i]
            out_dim = self.layer_dims[i+1]

            # Extract weight matrix
            weight_size = in_dim * out_dim
            weight = params[offset:offset + weight_size].view(in_dim, out_dim)
            weights.append(weight)
            offset += weight_size

            # Extract bias vector
            bias = params[offset:offset + out_dim]
            biases.append(bias)
            offset += out_dim

        return {'weights': weights, 'biases': biases}


class ConvWeightGenerator(nn.Module):
    """
    Generates convolutional kernel weights.
    Based on Ha et al.'s hypernetwork architecture.

    Args:
        z_dim (int): Embedding dimension
        out_channels (int): Number of output channels
        in_channels (int): Number of input channels
        kernel_size (int): Kernel size (assumes square kernels)
    """

    def __init__(self, z_dim: int, out_channels: int, in_channels: int, kernel_size: int = 3):
        super(ConvWeightGenerator, self).__init__()
        self.z_dim = z_dim
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

        # Two-layer hypernetwork from paper
        self.w1 = nn.Parameter(torch.randn(z_dim, in_channels * z_dim))
        self.b1 = nn.Parameter(torch.zeros(in_channels * z_dim))
        self.w2 = nn.Parameter(torch.randn(z_dim, out_channels * kernel_size * kernel_size))
        self.b2 = nn.Parameter(torch.zeros(out_channels * kernel_size * kernel_size))

        # Initialize
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate convolutional kernel.

        Returns:
            torch.Tensor: Kernel [out_channels, in_channels, kernel_size, kernel_size]
        """
        # First layer: z -> hidden representation per input channel
        h_in = torch.matmul(z, self.w2) + self.b2
        h_in = h_in.view(self.in_channels, self.z_dim)

        # Second layer: hidden -> kernel slices
        h_final = torch.matmul(h_in, self.w1) + self.b1

        # Reshape to kernel format
        kernel = h_final.view(self.out_channels, self.in_channels,
                             self.kernel_size, self.kernel_size)

        return kernel


class ResidualWeightGenerator(nn.Module):
    """
    Generates weights for residual connections.
    Produces two linear transformations: input -> hidden, hidden -> output.

    Args:
        z_dim (int): Embedding dimension
        n_in (int): Input channel dimension
        n_out (int): Output channel dimension
    """

    def __init__(self, z_dim: int, n_in: int, n_out: int):
        super(ResidualWeightGenerator, self).__init__()
        self.z_dim = z_dim
        self.n_in = n_in
        self.n_out = n_out

        # Generate both transformations
        total_params = (n_in * n_out) * 2 + n_in + n_out  # Two weight matrices + biases

        self.generator = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, total_params)
        )

    def forward(self, z: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Generate residual weights.

        Returns:
            dict: {'linear1': {'weight': W1, 'bias': b1},
                   'linear2': {'weight': W2, 'bias': b2}}
        """
        params = self.generator(z)

        offset = 0

        # First linear: input -> hidden
        w1_size = self.n_in * self.n_out
        w1 = params[offset:offset + w1_size].view(self.n_in, self.n_out)
        offset += w1_size

        b1 = params[offset:offset + self.n_out]
        offset += self.n_out

        # Second linear: hidden -> output
        w2_size = self.n_out * self.n_in
        w2 = params[offset:offset + w2_size].view(self.n_out, self.n_in)
        offset += w2_size

        b2 = params[offset:offset + self.n_in]

        return {
            'linear1': {'weight': w1, 'bias': b1},
            'linear2': {'weight': w2, 'bias': b2}
        }


class MultiComponentGenerator(nn.Module):
    """
    Master generator that produces weights for all ARCCompressor components.
    Coordinates multiple specialized generators.

    Args:
        z_dim (int): Embedding dimension
        n_layers (int): Number of layers in ARCCompressor
        channel_dims (dict): Channel dimensions for each component type
    """

    def __init__(self, z_dim: int, n_layers: int, channel_dims: Dict):
        super(MultiComponentGenerator, self).__init__()
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.channel_dims = channel_dims

        # Extract dimensions
        share_up_dim = channel_dims['share_up_dim']
        share_down_dim = channel_dims['share_down_dim']
        decoding_dim = channel_dims['decoding_dim']
        softmax_dim = channel_dims['softmax_dim']
        cummax_dim = channel_dims['cummax_dim']
        shift_dim = channel_dims['shift_dim']
        nonlinear_dim = channel_dims['nonlinear_dim']

        # Decode weights generator
        self.decode_gen = LinearWeightGenerator(z_dim, decoding_dim * 16, hidden_dim=128)

        # Layer-specific generators (one per layer)
        self.share_up_gens = nn.ModuleList([
            ResidualWeightGenerator(z_dim, share_up_dim, share_up_dim)
            for _ in range(n_layers)
        ])

        self.share_down_gens = nn.ModuleList([
            ResidualWeightGenerator(z_dim, share_down_dim, share_down_dim)
            for _ in range(n_layers)
        ])

        self.softmax_gens = nn.ModuleList([
            ResidualWeightGenerator(z_dim, softmax_dim, softmax_dim)
            for _ in range(n_layers)
        ])

        self.cummax_gens = nn.ModuleList([
            ResidualWeightGenerator(z_dim, cummax_dim, cummax_dim)
            for _ in range(n_layers)
        ])

        self.shift_gens = nn.ModuleList([
            ResidualWeightGenerator(z_dim, shift_dim, shift_dim)
            for _ in range(n_layers)
        ])

        self.nonlinear_gens = nn.ModuleList([
            ResidualWeightGenerator(z_dim, nonlinear_dim, nonlinear_dim)
            for _ in range(n_layers)
        ])

        # Head weights generator
        self.head_gen = LinearWeightGenerator(z_dim, 16 * 11, hidden_dim=128)

        # Mask weights generator
        self.mask_gen = LinearWeightGenerator(z_dim, 16 * 2, hidden_dim=64)

    def forward(self, z: torch.Tensor) -> Dict[str, any]:
        """
        Generate all weights for ARCCompressor.

        Args:
            z (torch.Tensor): Task embedding [z_dim]

        Returns:
            dict: Complete weight dictionary matching ARCCompressor structure
        """
        weights = {}

        # Decode weights
        weights['decode_weights'] = self.decode_gen(z)

        # Layer-wise weights
        weights['share_up_weights'] = [gen(z) for gen in self.share_up_gens]
        weights['share_down_weights'] = [gen(z) for gen in self.share_down_gens]
        weights['softmax_weights'] = [gen(z) for gen in self.softmax_gens]
        weights['cummax_weights'] = [gen(z) for gen in self.cummax_gens]
        weights['shift_weights'] = [gen(z) for gen in self.shift_gens]
        weights['nonlinear_weights'] = [gen(z) for gen in self.nonlinear_gens]

        # Output heads
        weights['head_weights'] = self.head_gen(z)
        weights['mask_weights'] = self.mask_gen(z)

        return weights
