import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from multitensor_systems import MultiTensorSystem, multitensor_systems  # Assuming from multitensor_systems.py
from pre_processing import Task  # For task metadata

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

class HyperNet(nn.Module):
    """
    A hypernetwork that generates the weights_list for ARCCompressor conditionally based on task metadata.
    - Takes a puzzle embedding (learnable, high-dim vector, e.g., 128-256 dim) and metadata (n_examples, n_colors, n_x, n_y).
    - Modular: Separate heads for each component in weights_list (e.g., multiposteriors, decode_weights).
    - Computes exact shapes using multitensor_system.shape(dims, channel_dim) where needed.
    - Outputs a list of tensors matching weights_list structure.
    - Handles lists (e.g., share_up_weights for n_layers=4) by generating per layer.
    - Post-generation: Applies symmetrization (symmetrize_xy, symmetrize_direction_sharing) as in initializer.
    
    Usage: In ARCCompressor.__init__, replace Initializer with HyperNet generation.
    """
    def __init__(self, multitensor_system: MultiTensorSystem, channel_dim_fn, emb_dim: int = 128, n_layers: int = 4):
        super().__init__()
        self.multitensor_system = multitensor_system
        self.channel_dim_fn = channel_dim_fn
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        
        # Metadata dim: Fixed 4 (n_examples, n_colors, n_x, n_y)
        self.metadata_dim = 4
        
        # Shared body: MLP to process embedding + normalized metadata
        self.body = nn.Sequential(
            nn.Linear(self.emb_dim + self.metadata_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        
        # Modular heads: Larger output (16384) to cover max tensor sizes (~10k elements typical)
        max_chunk = 16384
        self.head_multiposteriors = nn.Linear(512, max_chunk)  
        self.head_multilinear = nn.Linear(512, max_chunk)      
        self.head_multizeros = nn.Linear(512, max_chunk)        
        self.head_multiresidual = nn.Linear(512, max_chunk)    
        self.head_multidirection_share = nn.Linear(512, max_chunk)  
        self.head_head_weights = nn.Linear(512, max_chunk)      
        
        # Learnable puzzle embedding: Random init, as in Ha et al.
        self.puzzle_emb = nn.Parameter(torch.rand(emb_dim).uniform_(-0.01, 0.01))
        
    def forward(self, task: Task):
        """
        Generate weights_list conditionally on task metadata.
        - Normalize metadata [0,1] based on global maxes (from stats: n_examples<=12, n_x/n_y<=30, n_colors variable but small).
        - Concat to embedding, process through body.
        - Use heads to generate flattened chunks, reshape to exact shapes via multitensor_system.shape.
        - For lists: Loop over n_layers.
        - Return weights_list as list of tensors/groups.
        """
        device = self.puzzle_emb.device
        # Normalize metadata (based on global maxes from stats)
        metadata = torch.tensor([
            task.n_examples / 12.0,
            task.n_colors / 9.0,  # Safer max
            task.n_x / 30.0,
            task.n_y / 30.0
        ], dtype=torch.float32, device=device)
        
        # Input: Embedding + metadata
        h = torch.cat([self.puzzle_emb, metadata])
        h = self.body(h)  # Shared representation [512]
        
        weights_list = []
        
        # 1. multiposteriors: [mean, local_capacity_adjustment]
        # Use component-specific dims (e.g., for decoding; from initializer)
        multiposteriors = self._generate_multiposterior(h, task)
        weights_list.extend(multiposteriors)  # Includes mean in list
        
        # 2. decode_weights: Multilinear [decoding_dim=4, channel_dim_fn]
        # dims from original: [1,1,0,1,1] typical
        decode_weights = self._generate_multilinear(h, task, [4, self.channel_dim_fn], dims=[1,1,0,1,1])
        self.symmetrize_xy(decode_weights)  # As in original
        weights_list.extend(decode_weights)
        
        # 3. target_capacities: Multizeros [decoding_dim=4]
        target_capacities = self._generate_multizeros(h, task, [4], dims=[1,1,0,1,1])
        weights_list.extend(target_capacities)
        
        # 4. Layer lists (n_layers=4): e.g., share_up_weights (multiresiduals)
        share_up_weights = []
        share_down_weights = []
        softmax_weights = []
        cummax_weights = []
        shift_weights = []
        direction_share_weights = []
        nonlinear_weights = []
        
        for layer_num in range(self.n_layers):
            # Condition on layer (add layer index to h for differentiation)
            h_layer = torch.cat([h, torch.tensor([layer_num / self.n_layers], dtype=torch.float32, device=device)])
            
            # share_up_weights: Multiresidual (share_up_dim=16)
            share_up = self._generate_multiresidual(h_layer, task, 16, 16)
            share_up_weights.append(share_up)
            weights_list.extend(share_up)  # Flatten internals
            
            # share_down_weights: Similar (share_down_dim=8)
            share_down = self._generate_multiresidual(h_layer, task, 8, 8)
            share_down_weights.append(share_down)
            weights_list.extend(share_down)
            
            # softmax_weights: Multiresidual with scaling fn
            def output_scaling_fn(dims):
                return 2 * (2 ** (sum(dims[1:]) ) - 1)  # softmax_dim=2, adjusted for dims
            softmax = self._generate_multiresidual(h_layer, task, 2, output_scaling_fn)
            softmax_weights.append(softmax)
            weights_list.extend(softmax)
            
            # cummax_weights: Multiresidual (cummax_dim=4)
            cummax = self._generate_multiresidual(h_layer, task, 4, 4)
            cummax_weights.append(cummax)
            weights_list.extend(cummax)
            
            # shift_weights: Multiresidual (shift_dim=4)
            shift = self._generate_multiresidual(h_layer, task, 4, 4)
            shift_weights.append(shift)
            weights_list.extend(shift)
            
            # direction_share_weights: Multidirection_share (8x8 linears)
            direction_share = self._generate_multidirection_share(h_layer, task)
            direction_share_weights.append(direction_share)
            weights_list.extend(direction_share)  # Flatten 8x8
            
            # nonlinear_weights: Multiresidual (nonlinear_dim=16)
            nonlinear = self._generate_multiresidual(h_layer, task, 16, 16)
            nonlinear_weights.append(nonlinear)
            weights_list.extend(nonlinear)
        
        # 5. head_weights: Special linear head with symmetry
        head_weights = self._generate_head(h, task)
        weights_list.extend(head_weights)
        
        # 6. mask_weights: Linear [channel_dim_fn([1,0,0,1,0]), 2]
        mask_weights = self._generate_linear(h, task, [[1,0,0,1,0], [self.channel_dim_fn([1,0,0,1,0]), 2]], dims=[1,0,0,1,0])
        weights_list.extend(mask_weights)
        
        # Post-Generation Symmetrization (as in original)
        for weight_list in [share_up_weights, share_down_weights, softmax_weights, cummax_weights, shift_weights, nonlinear_weights]:
            for layer_num in range(self.n_layers):
                self.symmetrize_xy(weight_list[layer_num])
        
        for layer_num in range(self.n_layers):
            self.symmetrize_direction_sharing(direction_share_weights[layer_num])
        
        return weights_list  # Flat list as in original
    
    # Helper Methods (Modular Generation with Shaping)
    def _generate_multiposterior(self, h: torch.Tensor, task: Task):
        chunk = self.head_multiposteriors(h)  # Flattened base
        # Exact shape from multitensor_system (for mean/adjustment); use appropriate dims (from init: variable per tensor)
        dims = next(iter(self.multitensor_system))  # Example: first valid dims; in real, per multitensor
        shape = tuple(self.multitensor_system.shape(dims, 4))  # decoding_dim=4
        numel = int(np.prod(shape))
        mean = chunk[:numel].view(shape) * 0.01  # Scale as in init
        adjustment = torch.zeros_like(mean)  # Zeros-like
        return [mean, adjustment]
    
    def _generate_multilinear(self, h: torch.Tensor, task: Task, shape_fn, dims):
        chunk = self.head_multilinear(h)
        shape = shape_fn(dims) if callable(shape_fn) else shape_fn
        n_in, n_out = shape
        n_in = n_in(dims) if callable(n_in) else n_in
        n_out = n_out(dims) if callable(n_out) else n_out
        numel_weight = n_in * n_out
        numel_bias = n_out
        scale = 1 / np.sqrt(n_in)
        weight = chunk[:numel_weight].view(n_in, n_out) * scale
        bias = chunk[numel_weight:numel_weight + numel_bias].view(n_out) * scale
        return [weight, bias]
    
    def _generate_multizeros(self, h: torch.Tensor, task: Task, shape_fn, dims):
        chunk = self.head_multizeros(h)
        shape = shape_fn(dims) if callable(shape_fn) else shape_fn
        numel = int(np.prod(shape))
        zeros = chunk[:numel].view(shape) * 0.01  # Small scale instead of hard zeros
        return [zeros]
    
    def _generate_multiresidual(self, h: torch.Tensor, task: Task, n_in, n_out_fn):
        # Two linears: To/from residual
        linear1 = self._generate_multilinear(h, task, [self.channel_dim_fn, n_in], dims=[1,1,0,1,1])  # Adjust dims
        linear2 = self._generate_multilinear(h, task, [n_out_fn, self.channel_dim_fn], dims=[1,1,0,1,1])
        return [linear1, linear2]
    
    def _generate_multidirection_share(self, h: torch.Tensor, task: Task):
        # 8x8 linears [channel, channel]
        direction_shares = []
        for _ in range(8):
            row = []
            for _ in range(8):
                row.append(self._generate_multilinear(h, task, [self.channel_dim_fn, self.channel_dim_fn], dims=[1,1,1,1,1]))  # Directional dims
            direction_shares.append(row)
        return direction_shares
    
    def _generate_head(self, h: torch.Tensor, task: Task):
        dims = [1,1,0,1,1]
        head_weights = self._generate_multilinear(h, task, [self.channel_dim_fn(dims), 2], dims)
        # Symmetry as in init
        head_weights[0] = torch.stack([head_weights[0][..., 0]] * 2, dim=-1)
        return head_weights
    
    # Symmetry Helpers (Copied/Adapted from initializer.py)
    def symmetrize_xy(self, multiweights):
        for dims in self.multitensor_system:
            if dims[3] == 0 and dims[4] == 1:
                multiweights[dims] = multiweights[dims[:3] + [1, 0]]
    
    def symmetrize_direction_sharing(self, multiweights):
        # Full impl from initializer.py (as provided)
        for dims in self.multitensor_system:
            for dir1 in range(8):
                for dir2 in range(8):
                    from_dims = dims
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

                    multiweights[dims][dir1][dir2] = multiweights[from_dims][from_dir1][from_dir2]