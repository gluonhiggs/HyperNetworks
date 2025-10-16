import torch
import torch.nn as nn
import numpy as np
import layers
from hypernetwork_arc import HyperNetworkARC
from pre_processing import Task


class EmbeddingPredictor(nn.Module):
    """
    Predicts initial embedding from task metadata.

    Learns the mapping: metadata → embedding space
    Enables informed initialization for new tasks during meta-learning.
    """
    def __init__(self, emb_dim=128):
        super().__init__()
        self.metadata_dim = 4  # n_examples, n_colors, n_x, n_y

        # MLP: metadata → embedding
        self.predictor = nn.Sequential(
            nn.Linear(self.metadata_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, emb_dim)
        )

    def forward(self, task):
        """
        Predict embedding from task metadata.

        Args:
            task: Task object with metadata (n_examples, n_colors, n_x, n_y)

        Returns:
            predicted_embedding: [emb_dim] tensor
        """
        # Normalize metadata (same as hypernetwork)
        metadata = torch.tensor([
            task.n_examples / 12.0,
            task.n_colors / 9.0,
            task.n_x / 30.0,
            task.n_y / 30.0
        ], dtype=torch.float32, device=next(self.parameters()).device)

        return self.predictor(metadata)


class PuzzleEmbedding(nn.Module):
    """
    Puzzle embedding module for ARC tasks.

    Mimics the interface of primary_net.py's Embedding class, but adapted for CompressARC:
    - Single z-vector per task (not multiple per layer)
    - forward() takes hypernetwork AND task (needs metadata)
    - Owns the embedding parameter
    - IMPROVED: Can initialize from metadata predictor for better convergence

    Usage (similar to CIFAR-10):
        puzzle_emb = PuzzleEmbedding(emb_dim=128)
        weights_list = puzzle_emb(hypernetwork, task)
    """

    def __init__(self, emb_dim=128, init_std=0.01, initial_embedding=None):
        """
        Initialize puzzle embedding.

        Args:
            emb_dim: Dimension of embedding vector
            init_std: Standard deviation for random initialization (if initial_embedding=None)
            initial_embedding: Optional pre-computed embedding (from EmbeddingPredictor)
        """
        super().__init__()
        self.emb_dim = emb_dim

        # Initialize from predictor (informed) or randomly (fallback)
        if initial_embedding is not None:
            self.z = nn.Parameter(initial_embedding.detach().clone())
        else:
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

    def __init__(self, emb_dim=128, use_metadata_init=True):
        """
        Initialize ARC primary network (hypernetwork + embeddings + architecture).

        Args:
            emb_dim: Dimension of puzzle embeddings
            use_metadata_init: If True, use metadata predictor for embedding initialization
        """
        super().__init__()
        self.emb_dim = emb_dim
        self.use_metadata_init = use_metadata_init

        # Hypernetwork that generates all ARCCompressor weights from embeddings
        # (CIFAR-10 calls this self.hope, we use descriptive name)
        self.hypernetwork = HyperNetworkARC(emb_dim=emb_dim)

        # Metadata → embedding predictor (meta-learned)
        if use_metadata_init:
            self.embedding_predictor = EmbeddingPredictor(emb_dim=emb_dim)
        else:
            self.embedding_predictor = None

        # Task embeddings (analogous to self.zs in primary_net.py)
        # ModuleDict allows saving/loading with state_dict()
        self.task_embeddings = nn.ModuleDict()

    def get_or_create_embedding(self, task_name, task, init_std=0.01):
        """
        Get embedding if exists, otherwise create new one (metadata-informed if available).

        Args:
            task_name: Name of the task
            task: Task object (for metadata-based initialization)
            init_std: Standard deviation for random initialization (fallback)

        Returns:
            embedding: PuzzleEmbedding instance
        """
        if task_name not in self.task_embeddings:
            # Use metadata predictor for informed initialization
            if self.use_metadata_init and self.embedding_predictor is not None:
                initial_embedding = self.embedding_predictor(task)
                self.task_embeddings[task_name] = PuzzleEmbedding(
                    emb_dim=self.emb_dim,
                    initial_embedding=initial_embedding
                )
            else:
                # Fallback to random initialization
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
            - CIFAR: w1 = self.zs[i](self.hope) → net(images) → class predictions
            - ARC: weights = puzzle_emb(self.hypernetwork, task) → net(task) → pixel logits

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
        # Get or create task-specific embedding (metadata-informed initialization)
        puzzle_embedding = self.get_or_create_embedding(task_name, task)

        # Generate weights from hypernetwork (matching CIFAR-10: w1 = self.zs[i](self.hope))
        weights = puzzle_embedding(self.hypernetwork, task)

        # Forward pass through ARCCompressor architecture (like CIFAR-10's ResNet blocks)
        multitensor_system = task.multitensor_system

        # Decoding layer
        x, KL_amounts, KL_names = layers.decode_latents(
            weights.target_capacities, weights.decode_weights, weights.multiposteriors
        )

        # Transformer-like blocks
        for layer_num in range(self.n_layers):
            x = layers.share_up(x, weights.share_up_weights[layer_num])
            x = layers.softmax(x, weights.softmax_weights[layer_num], pre_norm=True, post_norm=False, use_bias=False)
            x = layers.cummax(x, weights.cummax_weights[layer_num], multitensor_system.task.masks,
                            pre_norm=False, post_norm=True, use_bias=False)
            x = layers.shift(x, weights.shift_weights[layer_num], multitensor_system.task.masks,
                           pre_norm=False, post_norm=True, use_bias=False)
            x = layers.direction_share(x, weights.direction_share_weights[layer_num], pre_norm=True, use_bias=False)
            x = layers.nonlinear(x, weights.nonlinear_weights[layer_num], pre_norm=True, post_norm=False, use_bias=False)
            x = layers.share_down(x, weights.share_down_weights[layer_num])
            x = layers.normalize(x)

        # Linear heads
        output = (
            layers.affine(x[[1, 1, 0, 1, 1]], weights.head_weights, use_bias=False)
            + 100 * weights.head_weights[1]  # head_weights is now [weight, bias]
        )
        x_mask = layers.affine(x[[1, 0, 0, 1, 0]], weights.mask_weights, use_bias=True)
        y_mask = layers.affine(x[[1, 0, 0, 0, 1]], weights.mask_weights, use_bias=True)

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
