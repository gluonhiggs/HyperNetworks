import torch
import torch.nn as nn
import numpy as np
import layers
from hypernetwork_arc import HyperNetworkARC
from pre_processing import Task


class PuzzleEmbedding(nn.Module):
    """
    Puzzle embedding module for ARC tasks.

    Mimics the interface of primary_net.py's Embedding class, but adapted for CompressARC:
    - Single z-vector per task (not multiple per layer)
    - forward() takes hypernetwork AND task (needs metadata)
    - Owns the embedding parameter

    Usage (similar to CIFAR-10):
        puzzle_emb = PuzzleEmbedding(emb_dim=128)
        weights_list = puzzle_emb(hypernetwork, task)
    """

    def __init__(self, emb_dim=128, init_std=0.01):
        """
        Initialize puzzle embedding.

        Args:
            emb_dim: Dimension of embedding vector
            init_std: Standard deviation for random initialization
        """
        super().__init__()
        self.emb_dim = emb_dim
        # Single embedding vector (analogous to z in primary_net.py)
        self.z = nn.Parameter(torch.randn(emb_dim) * init_std)

    def forward(self, hyper_net, task):
        """
        Generate weights by passing embedding through hypernetwork.

        This mimics the interface from primary_net.py:
            CIFAR-10: embedding.forward(hyper_net) → weights
            CompressARC: puzzle_embedding.forward(hyper_net, task) → weights

        Args:
            hyper_net: HyperNetworkARC instance
            task: Task object with metadata (n_examples, n_colors, n_x, n_y)

        Returns:
            weights_list: Flat list of all weight tensors for ARCCompressor
        """
        return hyper_net(task, self.z)


class ARCPrimaryNetwork(nn.Module):
    """
    Primary network for ARC-AGI hypernetwork training.

    Matches primary_net.py pattern: contains hypernetwork, embeddings, AND architecture.
    This is the complete end-to-end model like CIFAR-10's PrimaryNetwork.

    Analogous to CIFAR-10's PrimaryNetwork:
        - CIFAR: self.hope + self.zs + ResNet layers + forward(images)
        - ARC: self.hypernetwork + self.task_embeddings + ARCCompressor logic + forward(task, task_name)

    Usage (matching train_hyper.py pattern):
        net = ARCPrimaryNetwork(emb_dim=128)
        if args.resume:
            ckpt = torch.load('./hypernetworks_arc.pth')
            net.load_state_dict(ckpt['net'])
        net.cuda()

        # Training (like CIFAR-10's net(images))
        for task in tasks:
            output, x_mask, y_mask, KL_amounts, KL_names = net(task, task_name)
            # optimize net.parameters()

        # Save like CIFAR-10
        torch.save({'net': net.state_dict()}, './hypernetworks_arc.pth')
    """

    # Architecture constants (same as ARCCompressor)
    n_layers = 4
    share_up_dim = 16
    share_down_dim = 8
    decoding_dim = 4
    softmax_dim = 2
    cummax_dim = 4
    shift_dim = 4
    nonlinear_dim = 16

    def __init__(self, emb_dim=128):
        """
        Initialize ARC primary network (hypernetwork + embeddings + architecture).

        Args:
            emb_dim: Dimension of puzzle embeddings
        """
        super().__init__()
        self.emb_dim = emb_dim

        # Hypernetwork that generates all ARCCompressor weights from embeddings
        # (CIFAR-10 calls this self.hope, we use descriptive name)
        self.hypernetwork = HyperNetworkARC(emb_dim=emb_dim)

        # Task embeddings (analogous to self.zs in primary_net.py)
        # ModuleDict allows saving/loading with state_dict()
        self.task_embeddings = nn.ModuleDict()

    def get_or_create_embedding(self, task_name, init_std=0.01):
        """
        Get embedding if exists, otherwise create new one.

        Args:
            task_name: Name of the task
            init_std: Standard deviation for random initialization if creating

        Returns:
            embedding: PuzzleEmbedding instance
        """
        if task_name not in self.task_embeddings:
            self.task_embeddings[task_name] = PuzzleEmbedding(
                emb_dim=self.emb_dim, init_std=init_std
            )
        return self.task_embeddings[task_name]

    def channel_dim_fn(self, dims):
        """Return channel dimension based on presence of direction dimension."""
        return 16 if dims[2] == 0 else 8

    def forward(self, task: Task, task_name: str):
        """
        Complete forward pass for ARC task (matching CIFAR-10 pattern).

        This matches CIFAR-10's PrimaryNetwork.forward(images) but for ARC:
            - CIFAR: net(images) → class predictions
            - ARC: net(task, task_name) → pixel logits, masks, KL divergences

        Args:
            task: Task object with puzzle data and metadata
            task_name: Name of task (to get/create embedding)

        Returns:
            output: [example, color, x, y, channel] tensor of pixel logits
            x_mask: [example, x, channel] mask scores
            y_mask: [example, y, channel] mask scores
            KL_amounts: List of KL divergence contributions
            KL_names: List of names for KL components
        """
        # Get or create task-specific embedding
        puzzle_embedding = self.get_or_create_embedding(task_name)

        # Generate weights from hypernetwork (like CIFAR-10's w1 = self.zs[i](self.hope))
        device = puzzle_embedding.z.device
        multitensor_system = task.multitensor_system
        channel_dim_fn = self.channel_dim_fn

        metadata = torch.tensor([
            task.n_examples / 12.0,
            task.n_colors / 9.0,
            task.n_x / 30.0,
            task.n_y / 30.0
        ], dtype=torch.float32, device=device)

        h = torch.cat([puzzle_embedding.z, metadata])
        h = self.hypernetwork.body(h)

        # Generate all weight components
        multiposteriors = self.hypernetwork.generate_multiposterior(
            h, multitensor_system, channel_dim_fn, 4
        )

        decode_weights = self.hypernetwork.generate_multilinear(
            h, multitensor_system, channel_dim_fn, [4, channel_dim_fn]
        )

        target_capacities = self.hypernetwork.generate_multizeros(
            h, multitensor_system, [4]
        )

        # Layer weights
        share_up_weights = []
        share_down_weights = []
        softmax_weights = []
        cummax_weights = []
        shift_weights = []
        direction_share_weights = []
        nonlinear_weights = []

        for layer_idx in range(self.n_layers):
            layer_emb = torch.cat([h, torch.tensor([layer_idx / self.n_layers], device=device)])

            share_up_weights.append(
                self.hypernetwork.generate_multiresidual(layer_emb, multitensor_system, channel_dim_fn, 16, 16)
            )
            share_down_weights.append(
                self.hypernetwork.generate_multiresidual(layer_emb, multitensor_system, channel_dim_fn, 8, 8)
            )

            def softmax_output_fn(dims):
                return 2 * (2 ** (sum(dims[1:])) - 1)

            softmax_weights.append(
                self.hypernetwork.generate_multiresidual(layer_emb, multitensor_system, channel_dim_fn, 2, softmax_output_fn)
            )
            cummax_weights.append(
                self.hypernetwork.generate_multiresidual(layer_emb, multitensor_system, channel_dim_fn, 4, 4)
            )
            shift_weights.append(
                self.hypernetwork.generate_multiresidual(layer_emb, multitensor_system, channel_dim_fn, 4, 4)
            )
            direction_share_weights.append(
                self.hypernetwork.generate_multidirection_share(layer_emb, multitensor_system, channel_dim_fn)
            )
            nonlinear_weights.append(
                self.hypernetwork.generate_multiresidual(layer_emb, multitensor_system, channel_dim_fn, 16, 16)
            )

        head_weights = self.hypernetwork.generate_head(h, multitensor_system, channel_dim_fn)

        mask_weights = self.hypernetwork.generate_multilinear(
            h, multitensor_system, channel_dim_fn,
            [channel_dim_fn([1, 0, 0, 1, 0]), 2],
            specific_dims=[1, 0, 0, 1, 0]
        )

        # Apply symmetrization
        for weight_list in [share_up_weights, share_down_weights, softmax_weights,
                            cummax_weights, shift_weights, nonlinear_weights]:
            for weights in weight_list:
                self.hypernetwork.symmetrize_xy(weights, multitensor_system)

        for weights in direction_share_weights:
            self.hypernetwork.symmetrize_direction_sharing(weights, multitensor_system)

        # Forward pass through ARCCompressor architecture (like CIFAR-10's ResNet blocks)
        # Decoding layer
        x, KL_amounts, KL_names = layers.decode_latents(
            target_capacities, decode_weights, multiposteriors
        )

        # Transformer-like blocks
        for layer_num in range(self.n_layers):
            x = layers.share_up(x, share_up_weights[layer_num])
            x = layers.softmax(x, softmax_weights[layer_num], pre_norm=True, post_norm=False, use_bias=False)
            x = layers.cummax(x, cummax_weights[layer_num], multitensor_system.task.masks,
                            pre_norm=False, post_norm=True, use_bias=False)
            x = layers.shift(x, shift_weights[layer_num], multitensor_system.task.masks,
                           pre_norm=False, post_norm=True, use_bias=False)
            x = layers.direction_share(x, direction_share_weights[layer_num], pre_norm=True, use_bias=False)
            x = layers.nonlinear(x, nonlinear_weights[layer_num], pre_norm=True, post_norm=False, use_bias=False)
            x = layers.share_down(x, share_down_weights[layer_num])
            x = layers.normalize(x)

        # Linear heads
        output = (
            layers.affine(x[[1, 1, 0, 1, 1]], head_weights, use_bias=False)
            + 100 * head_weights[[1, 1, 0, 1, 1]][1]
        )
        x_mask = layers.affine(x[[1, 0, 0, 1, 0]], mask_weights, use_bias=True)
        y_mask = layers.affine(x[[1, 0, 0, 0, 1]], mask_weights, use_bias=True)

        # Postprocessing
        x_mask, y_mask = layers.postprocess_mask(multitensor_system.task, x_mask, y_mask)

        return output, x_mask, y_mask, KL_amounts, KL_names

    def __len__(self):
        """Return number of task embeddings."""
        return len(self.task_embeddings)

    def __contains__(self, task_name):
        """Check if task has an embedding."""
        return task_name in self.task_embeddings

    def __repr__(self):
        return f"ARCPrimaryNetwork(emb_dim={self.emb_dim}, n_tasks={len(self)})"
