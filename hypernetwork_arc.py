import numpy as np
import torch
import torch.nn as nn
import multitensor_systems


class HyperNetworkARC(nn.Module):
    """
    A hypernetwork that generates weights for ARCCompressor dynamically based on task metadata.

    Architecture:
        - Input: Puzzle embedding (learnable vector) + normalized task metadata
        - Shared body: MLP that processes the input into a shared representation
        - Modular heads: Separate heads for each weight component type
        - Output: Complete weights_list matching Initializer structure

    Key features:
        - Dynamic weight generation based on task-specific dimensions
        - Conditional on metadata (n_examples, n_colors, n_x, n_y)
        - Modular design with separate generators for each weight type
        - Applies symmetrization post-generation for equivariance
    """

    def __init__(self, emb_dim=128, hidden_dim=512, max_chunk=16384):
        """
        Initialize the hypernetwork.

        Args:
            emb_dim: Dimension of puzzle embedding (default: 128)
            hidden_dim: Dimension of hidden layers in shared body (default: 512)
            max_chunk: Maximum size for weight generation chunks (default: 16384)
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.max_chunk = max_chunk
        self.metadata_dim = 4  # n_examples, n_colors, n_x, n_y

        # Shared body: processes embedding + metadata into shared representation
        self.body = nn.Sequential(
            nn.Linear(self.emb_dim + self.metadata_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Modular heads for different weight component types
        self.head_posterior = nn.Linear(hidden_dim, max_chunk)
        self.head_linear = nn.Linear(hidden_dim, max_chunk)
        self.head_direction_share = nn.Linear(hidden_dim, max_chunk)

    def forward(self, task, puzzle_emb):
        """
        Generate complete weights_list for the given task.

        Args:
            task: Task object with metadata (n_examples, n_colors, n_x, n_y)
            puzzle_emb: Learnable puzzle embedding tensor [emb_dim]

        Returns:
            weights_list: Flat list of all weight tensors
            Corresponding MultiTensor structures matching Initializer output
        """
        device = puzzle_emb.device
        multitensor_system = task.multitensor_system

        # Normalize metadata based on expected maxima
        metadata = torch.tensor([
            task.n_examples / 12.0,
            task.n_colors / 20.0,
            task.n_x / 30.0,
            task.n_y / 30.0
        ], dtype=torch.float32, device=device)

        # Process through shared body
        h = torch.cat([puzzle_emb, metadata])
        h = self.body(h)  # [hidden_dim]

        # Generate all weight components
        weights_list = []
        channel_dim_fn = lambda dims: 16 if dims[2] == 0 else 8

        # 1. Multiposteriors (for decoding layer)
        multiposteriors = self.generate_multiposterior(h, multitensor_system, channel_dim_fn, 4)
        weights_list.extend([w for mt in multiposteriors for w in self._flatten_multitensor(mt)])

        # 2. Decode weights (multilinear)
        decode_weights = self.generate_multilinear(h, multitensor_system, channel_dim_fn, [4, channel_dim_fn])
        weights_list.extend([w for mt in decode_weights for w in self._flatten_multitensor(mt)])

        # 3. Target capacities (multizeros)
        target_capacities = self.generate_multizeros(h, multitensor_system, [4])
        weights_list.extend([w for w in self._flatten_multitensor(target_capacities)])

        # 4-10. Layer weights (4 layers, each with 7 weight components)
        n_layers = 4
        share_up_weights_list = []
        share_down_weights_list = []
        softmax_weights_list = []
        cummax_weights_list = []
        shift_weights_list = []
        direction_share_weights_list = []
        nonlinear_weights_list = []

        for layer_idx in range(n_layers):
            # Condition on layer index
            layer_emb = torch.cat([h, torch.tensor([layer_idx / n_layers], device=device)])

            # Generate weights for this layer
            share_up = self.generate_multiresidual(layer_emb, multitensor_system, channel_dim_fn, 16, 16)
            share_down = self.generate_multiresidual(layer_emb, multitensor_system, channel_dim_fn, 8, 8)

            # Softmax with dynamic output scaling
            def softmax_output_fn(dims):
                return 2 * (2 ** (sum(dims[1:]) ) - 1)
            softmax = self.generate_multiresidual(layer_emb, multitensor_system, channel_dim_fn, 2, softmax_output_fn)

            cummax = self.generate_multiresidual(layer_emb, multitensor_system, channel_dim_fn, 4, 4)
            shift = self.generate_multiresidual(layer_emb, multitensor_system, channel_dim_fn, 4, 4)
            direction_share = self.generate_multidirection_share(layer_emb, multitensor_system, channel_dim_fn)
            nonlinear = self.generate_multiresidual(layer_emb, multitensor_system, channel_dim_fn, 16, 16)

            # Collect for symmetrization
            share_up_weights_list.append(share_up)
            share_down_weights_list.append(share_down)
            softmax_weights_list.append(softmax)
            cummax_weights_list.append(cummax)
            shift_weights_list.append(shift)
            direction_share_weights_list.append(direction_share)
            nonlinear_weights_list.append(nonlinear)

            # Add to flat weights_list
            for mt in [share_up, share_down, softmax, cummax, shift, direction_share, nonlinear]:
                weights_list.extend([w for w in self._flatten_multitensor(mt)])

        # 11. Head weights (linear head with symmetry)
        head_weights = self.generate_head(h, multitensor_system, channel_dim_fn)
        weights_list.extend([w for w in self._flatten_multitensor(head_weights)])

        # 12. Mask weights
        mask_weights = self.generate_multilinear(h, multitensor_system, channel_dim_fn,
                                                 [channel_dim_fn([1,0,0,1,0]), 2],
                                                 specific_dims=[1,0,0,1,0])
        weights_list.extend([w for w in self._flatten_multitensor(mask_weights)])

        # Apply symmetrization (post-generation)
        for weight_mt in [*share_up_weights_list, *share_down_weights_list, *softmax_weights_list,
                          *cummax_weights_list, *shift_weights_list, *nonlinear_weights_list]:
            self.symmetrize_xy(weight_mt, multitensor_system)

        for direction_share_mt in direction_share_weights_list:
            self.symmetrize_direction_sharing(direction_share_mt, multitensor_system)

        return weights_list

    def generate_multiposterior(self, h, multitensor_system, channel_dim_fn, decoding_dim):
        """Generate multiposterior (mean + local_capacity_adjustment) for decoding layer."""
        result = multitensor_system.make_multitensor()

        for dims in multitensor_system:
            shape = multitensor_system.shape(dims, decoding_dim)
            numel = int(np.prod(shape))

            # Generate mean
            chunk = self.head_posterior(h)
            mean = chunk[:numel].reshape(shape) * 0.01  # Small initialization
            mean = mean.detach().requires_grad_(True)

            # Local capacity adjustment (zeros)
            local_capacity_adj = torch.zeros_like(mean, requires_grad=True)

            result[dims] = [mean, local_capacity_adj]

        return result

    def generate_multilinear(self, h, multitensor_system, channel_dim_fn, shape_spec, specific_dims=None):
        """Generate multilinear weights (weight matrix + bias vector)."""
        result = multitensor_system.make_multitensor()

        dims_to_iterate = [specific_dims] if specific_dims else multitensor_system

        for dims in dims_to_iterate:
            # Compute n_in and n_out
            n_in_spec, n_out_spec = shape_spec
            n_in = n_in_spec(dims) if callable(n_in_spec) else n_in_spec
            n_out = n_out_spec(dims) if callable(n_out_spec) else n_out_spec

            numel_weight = n_in * n_out
            numel_bias = n_out

            # Generate weights with Xavier-like scaling
            chunk = self.head_linear(h)
            scale = 1.0 / np.sqrt(n_in)

            weight = chunk[:numel_weight].reshape(n_in, n_out) * scale
            bias = chunk[numel_weight:numel_weight + numel_bias].reshape(n_out) * scale

            weight = weight.detach().requires_grad_(True)
            bias = bias.detach().requires_grad_(True)

            result[dims] = [weight, bias]

        return result

    def generate_multizeros(self, h, multitensor_system, shape_spec):
        """Generate multizeros (small random initialization instead of hard zeros)."""
        result = multitensor_system.make_multitensor()

        for dims in multitensor_system:
            shape_fn = shape_spec[0] if isinstance(shape_spec, list) else shape_spec
            shape_val = shape_fn(dims) if callable(shape_fn) else shape_fn
            shape = multitensor_system.shape(dims, shape_val)

            numel = int(np.prod(shape))
            chunk = self.head_posterior(h)
            zeros = chunk[:numel].reshape(shape) * 0.01
            zeros = zeros.detach().requires_grad_(True)

            result[dims] = zeros

        return result

    def generate_multiresidual(self, h, multitensor_system, channel_dim_fn, n_in, n_out_fn):
        """Generate multiresidual (two linears for residual connection)."""
        # Two linear transformations: down-project and up-project
        linear1 = self.generate_multilinear(h, multitensor_system, channel_dim_fn, [channel_dim_fn, n_in])
        linear2 = self.generate_multilinear(h, multitensor_system, channel_dim_fn, [n_out_fn, channel_dim_fn])

        result = multitensor_system.make_multitensor()
        for dims in multitensor_system:
            result[dims] = [linear1[dims], linear2[dims]]

        return result

    def generate_multidirection_share(self, h, multitensor_system, channel_dim_fn):
        """Generate multidirection_share (8x8 grid of linear transformations)."""
        result = multitensor_system.make_multitensor()

        for dims in multitensor_system:
            direction_shares = []
            for _ in range(8):
                row = []
                for _ in range(8):
                    linear = self.generate_multilinear(h, multitensor_system, channel_dim_fn,
                                                      [channel_dim_fn, channel_dim_fn],
                                                      specific_dims=dims)
                    row.append(linear[dims])
                direction_shares.append(row)
            result[dims] = direction_shares

        return result

    def generate_head(self, h, multitensor_system, channel_dim_fn):
        """Generate head weights with symmetry."""
        dims = [1, 1, 0, 1, 1]
        head_weights = self.generate_multilinear(h, multitensor_system, channel_dim_fn,
                                                 [channel_dim_fn(dims), 2],
                                                 specific_dims=dims)

        # Apply symmetry: both output channels use the same weights
        weight, bias = head_weights[dims]
        weight_sym = torch.stack([weight[:, 0]] * 2, dim=-1)
        weight_sym = weight_sym.detach().requires_grad_(True)
        head_weights[dims] = [weight_sym, bias]

        return head_weights

    def symmetrize_xy(self, multiweights, multitensor_system):
        """Ensure xy swap symmetry by sharing weights."""
        for dims in multitensor_system:
            if dims[3] == 0 and dims[4] == 1:
                # Share weights between x and y dimensions
                multiweights[dims] = multiweights[tuple(dims[:3] + [1, 0])]

    def symmetrize_direction_sharing(self, multiweights, multitensor_system):
        """Ensure xy swap symmetry for directional communication layers."""
        for dims in multitensor_system:
            for dir1 in range(8):
                for dir2 in range(8):
                    from_dims = list(dims)
                    from_dir1, from_dir2 = dir1, dir2

                    if dims[3] + dims[4] == 1:
                        from_dims = dims[:3] + [1, 0]
                        if dims[4] == 1:
                            from_dir1 = (2 + from_dir1) % 8
                            from_dir2 = (2 + from_dir2) % 8

                        if from_dir1 > 4 or (from_dir1 in {0, 4} and from_dir2 > 4):
                            from_dir1 = (8 - from_dir1) % 8
                            from_dir2 = (8 - from_dir2) % 8

                        if 2 < from_dir1 < 6 or (from_dir1 in {2, 6} and 2 < from_dir2 < 6):
                            from_dir1 = (4 - from_dir1) % 8
                            from_dir2 = (4 - from_dir2) % 8
                    else:
                        rotation = (from_dir1 // 2) * 2
                        from_dir1 = (from_dir1 - rotation) % 8
                        from_dir2 = (from_dir2 - rotation) % 8

                        if (from_dir2 - from_dir1) % 8 > 4:
                            from_dir2 = (8 + 2 * from_dir1 - from_dir2) % 8

                    multiweights[tuple(dims)][dir1][dir2] = multiweights[tuple(from_dims)][from_dir1][from_dir2]

    def _flatten_multitensor(self, multitensor):
        """Helper to flatten MultiTensor into list of tensors."""
        if isinstance(multitensor, multitensor_systems.MultiTensor):
            result = []
            for dims in multitensor.multitensor_system:
                item = multitensor[dims]
                if isinstance(item, list):
                    result.extend(item)
                elif isinstance(item, torch.Tensor):
                    result.append(item)
                else:
                    # Handle nested structures (like direction_share)
                    result.extend(self._flatten_nested(item))
            return result
        elif isinstance(multitensor, list):
            result = []
            for item in multitensor:
                if isinstance(item, torch.Tensor):
                    result.append(item)
                else:
                    result.extend(self._flatten_nested(item))
            return result
        else:
            return [multitensor]

    def _flatten_nested(self, obj):
        """Recursively flatten nested structures."""
        result = []
        if isinstance(obj, torch.Tensor):
            return [obj]
        elif isinstance(obj, list):
            for item in obj:
                result.extend(self._flatten_nested(item))
        return result
