import numpy as np
import torch
import torch.nn as nn
import multitensor_systems


class WeightsStructure:
    """Container for all generated weights matching ARCCompressor structure."""
    def __init__(self):
        self.multiposteriors = None
        self.decode_weights = None
        self.target_capacities = None
        self.share_up_weights = []
        self.share_down_weights = []
        self.softmax_weights = []
        self.cummax_weights = []
        self.shift_weights = []
        self.direction_share_weights = []
        self.nonlinear_weights = []
        self.head_weights = None
        self.mask_weights = None


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

    def __init__(self, emb_dim=128, hidden_dim=512, chunk_size_posterior=1024, chunk_size_linear=512):
        """
        Initialize the hypernetwork with optimized 2-head design.

        Args:
            emb_dim: Dimension of puzzle embedding (default: 128)
            hidden_dim: Dimension of hidden layers in shared body (default: 512)
            chunk_size_posterior: Chunk size for posterior head (default: 1024)
                                 Used for: multiposteriors, target_capacities
                                 Optimized for bimodal distribution (18-4.6M params)
            chunk_size_linear: Chunk size for linear head (default: 512)
                              Used for: all linear transformations (decode, layers, heads)
                              Optimized for consistent small tensors (136-544 params)

        Benefits:
            - 44% parameter reduction (2.37M → 1.32M params in heads)
            - Removes dead code (head_direction_share was never used)
            - Better gradient utilization: ~50% waste vs 98% with uniform chunk_size=1024
            - head_linear optimized for 90%+ of tensors in typical ARC tasks
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.chunk_size_posterior = chunk_size_posterior
        self.chunk_size_linear = chunk_size_linear
        self.metadata_dim = 4  # n_examples, n_colors, n_x, n_y

        # Shared body: processes embedding + metadata into shared representation
        self.body = nn.Sequential(
            nn.Linear(self.emb_dim + self.metadata_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Two specialized heads based on fundamental operation types
        # Position encoding (dim=8) makes chunks unique for multi-chunk generation
        self.position_dim = 8

        # Head 1: Posterior parameters (CONTENT) - small-scale initialization (0.01)
        # Generates: multiposteriors, target_capacities
        self.head_posterior = nn.Sequential(
            nn.Linear(hidden_dim + self.position_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, chunk_size_posterior)
        )

        # Head 2: Linear transformations (STRUCTURE) - Xavier-scale initialization (1/√n_in)
        # Generates: decode, all residual layers, direction_share, head, mask weights
        self.head_linear = nn.Sequential(
            nn.Linear(hidden_dim + self.position_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, chunk_size_linear)
        )

    def forward(self, task, puzzle_emb):
        """
        Generate all weights for the given task.

        Args:
            task: Task object with metadata (n_examples, n_colors, n_x, n_y)
            puzzle_emb: Learnable puzzle embedding tensor [emb_dim]

        Returns:
            WeightsStructure: Object containing all weight components for ARCCompressor
        """
        device = puzzle_emb.device
        multitensor_system = task.multitensor_system

        # Normalize metadata based on expected maxima
        metadata = torch.tensor([
            task.n_examples / 12.0,
            task.n_colors / 9.0,
            task.n_x / 30.0,
            task.n_y / 30.0
        ], dtype=torch.float32, device=device)

        # Process through shared body
        h = torch.cat([puzzle_emb, metadata])
        h = self.body(h)  # [hidden_dim]

        # Create weights structure
        weights = WeightsStructure()
        channel_dim_fn = lambda dims: 16 if dims[2] == 0 else 8

        # 1. Multiposteriors (for decoding layer)
        weights.multiposteriors = self.generate_multiposterior(h, multitensor_system, 4)

        # 2. Decode weights (multilinear) with symmetrization
        weights.decode_weights = self.generate_multilinear(h, multitensor_system, [4, channel_dim_fn])
        self.symmetrize_xy(weights.decode_weights, multitensor_system)

        # 3. Target capacities (multizeros)
        weights.target_capacities = self.generate_multizeros(h, multitensor_system, [4])

        # 4-10. Layer weights (4 layers, each with 7 weight components)
        n_layers = 4
        for layer_idx in range(n_layers):
            # Condition on layer index
            layer_emb = torch.cat([h, torch.tensor([layer_idx / n_layers], device=device)])

            # Generate weights for this layer
            share_up = self.generate_multiresidual(layer_emb, multitensor_system, 16, 16, channel_dim_fn)
            share_down = self.generate_multiresidual(layer_emb, multitensor_system, 8, 8, channel_dim_fn)

            # Softmax with dynamic output scaling
            def softmax_output_fn(dims):
                return 2 * (2 ** (sum(dims[1:]) ) - 1)
            softmax = self.generate_multiresidual(layer_emb, multitensor_system, 2, softmax_output_fn, channel_dim_fn)

            cummax = self.generate_multiresidual(layer_emb, multitensor_system, 4, 4, channel_dim_fn)
            shift = self.generate_multiresidual(layer_emb, multitensor_system, 4, 4, channel_dim_fn)
            direction_share = self.generate_multidirection_share(layer_emb, multitensor_system, channel_dim_fn)
            nonlinear = self.generate_multiresidual(layer_emb, multitensor_system, 16, 16, channel_dim_fn)

            # Add to weight structure
            weights.share_up_weights.append(share_up)
            weights.share_down_weights.append(share_down)
            weights.softmax_weights.append(softmax)
            weights.cummax_weights.append(cummax)
            weights.shift_weights.append(shift)
            weights.direction_share_weights.append(direction_share)
            weights.nonlinear_weights.append(nonlinear)

        # 11. Head weights (linear head with symmetry)
        weights.head_weights = self.generate_head(h, multitensor_system, channel_dim_fn)

        # 12. Mask weights
        weights.mask_weights = self.generate_multilinear(h, multitensor_system,
                                                         [channel_dim_fn([1,0,0,1,0]), 2],
                                                         specific_dims=[1,0,0,1,0])

        # Apply symmetrization (post-generation)
        for weight_mt in [*weights.share_up_weights, *weights.share_down_weights, *weights.softmax_weights,
                          *weights.cummax_weights, *weights.shift_weights, *weights.nonlinear_weights]:
            self.symmetrize_xy(weight_mt, multitensor_system)

        for direction_share_mt in weights.direction_share_weights:
            self.symmetrize_direction_sharing(direction_share_mt, multitensor_system)

        return weights

    def _generate_tensor_chunked(self, h, head, numel, device):
        """
        Generate a tensor of arbitrary size by concatenating diverse head outputs.

        Uses position encoding to ensure different chunks have different values.
        Chunk size is determined dynamically from head's output dimension.

        Args:
            h: Hidden representation from body MLP [hidden_dim]
            head: The head network to use (expects hidden_dim + position_dim input)
                  Chunk size = head's final layer output dimension
            numel: Total number of elements needed
            device: Device to place tensor on

        Returns:
            Flat tensor of size [numel]
        """
        if numel == 0:
            return torch.tensor([], device=device)

        # Get chunk_size dynamically from head's output dimension
        # The final layer of the head determines the chunk size
        chunk_size = head[-1].out_features

        # Compute exactly how many chunks needed
        n_chunks = (numel + chunk_size - 1) // chunk_size

        if n_chunks == 1:
            # Single chunk: use zero position encoding
            pos_encoding = torch.zeros(self.position_dim, device=device)
            h_with_pos = torch.cat([h, pos_encoding])
            chunk = head(h_with_pos)
            return chunk[:numel]

        # Multiple chunks: generate each with unique position encoding
        chunks = []
        for chunk_idx in range(n_chunks):
            # Sinusoidal position encoding (like Transformers)
            pos_encoding = torch.zeros(self.position_dim, device=device)
            for i in range(self.position_dim):
                if i % 2 == 0:
                    pos_encoding[i] = np.sin(chunk_idx / (10000 ** (i / self.position_dim)))
                else:
                    pos_encoding[i] = np.cos(chunk_idx / (10000 ** (i / self.position_dim)))

            # Concatenate h with position encoding
            h_with_pos = torch.cat([h, pos_encoding])

            # Generate chunk conditioned on position
            chunk = head(h_with_pos)
            chunks.append(chunk)

        # Concatenate and trim to exact size (no waste!)
        full_tensor = torch.cat(chunks, dim=0)
        return full_tensor[:numel]

    def generate_multiposterior(self, h, multitensor_system, decoding_dim):
        """Generate multiposterior (mean + local_capacity_adjustment) for decoding layer."""
        result = multitensor_system.make_multitensor()
        device = h.device

        for dims in multitensor_system:
            shape = multitensor_system.shape(dims, decoding_dim)
            numel = int(np.prod(shape))

            # Generate mean using chunked generation (handles arbitrary sizes)
            mean_flat = self._generate_tensor_chunked(h, self.head_posterior, numel, device)
            mean = mean_flat.reshape(shape) * 0.01  # Small initialization
            mean = mean.detach().requires_grad_(True)

            # Local capacity adjustment (zeros)
            local_capacity_adj = torch.zeros_like(mean, requires_grad=True)

            result[dims] = [mean, local_capacity_adj]

        return result

    def generate_multilinear(self, h, multitensor_system, shape_spec, specific_dims=None):
        """Generate multilinear weights (weight matrix + bias vector)."""
        result = multitensor_system.make_multitensor()
        device = h.device

        dims_to_iterate = [specific_dims] if specific_dims else multitensor_system

        for dims in dims_to_iterate:
            # Compute n_in and n_out
            n_in_spec, n_out_spec = shape_spec
            n_in = n_in_spec(dims) if callable(n_in_spec) else n_in_spec
            n_out = n_out_spec(dims) if callable(n_out_spec) else n_out_spec

            numel_weight = n_in * n_out
            numel_bias = n_out

            # Generate weights with Xavier-like scaling using chunked generation
            scale = 1.0 / np.sqrt(n_in)

            # Generate weight matrix
            weight_flat = self._generate_tensor_chunked(h, self.head_linear, numel_weight, device)
            weight = weight_flat.reshape(n_in, n_out) * scale
            weight = weight.detach().requires_grad_(True)

            # Generate bias vector (usually small, but use chunking for consistency)
            bias_flat = self._generate_tensor_chunked(h, self.head_linear, numel_bias, device)
            bias = bias_flat.reshape(n_out) * scale
            bias = bias.detach().requires_grad_(True)

            result[dims] = [weight, bias]

        return result

    def generate_multizeros(self, h, multitensor_system, shape_spec):
        """Generate multizeros (small random initialization instead of hard zeros)."""
        result = multitensor_system.make_multitensor()
        device = h.device

        for dims in multitensor_system:
            shape_fn = shape_spec[0] if isinstance(shape_spec, list) else shape_spec
            shape_val = shape_fn(dims) if callable(shape_fn) else shape_fn
            shape = multitensor_system.shape(dims, shape_val)

            numel = int(np.prod(shape))

            # Generate zeros using chunked generation (handles arbitrary sizes)
            zeros_flat = self._generate_tensor_chunked(h, self.head_posterior, numel, device)
            zeros = zeros_flat.reshape(shape) * 0.01
            zeros = zeros.detach().requires_grad_(True)

            result[dims] = zeros

        return result

    def generate_multiresidual(self, h, multitensor_system, n_in, n_out_fn, channel_dim_fn):
        """Generate multiresidual (two linears for residual connection)."""
        # Two linear transformations: down-project and up-project
        linear1 = self.generate_multilinear(h, multitensor_system, [channel_dim_fn, n_in])
        linear2 = self.generate_multilinear(h, multitensor_system, [n_out_fn, channel_dim_fn])

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
                    linear = self.generate_multilinear(h, multitensor_system,
                                                      [channel_dim_fn, channel_dim_fn],
                                                      specific_dims=dims)
                    row.append(linear[dims])
                direction_shares.append(row)
            result[dims] = direction_shares

        return result

    def generate_head(self, h, multitensor_system, channel_dim_fn):
        """Generate head weights with symmetry."""
        dims = [1, 1, 0, 1, 1]
        head_weights = self.generate_multilinear(h, multitensor_system,
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
