"""
Dynamic HyperNetwork - Main module that generates task-specific weights for ARCCompressor.
Integrates puzzle embeddings and weight generators to produce conditional parameters.
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.puzzle_embedding import PuzzleEmbedding, HierarchicalPuzzleEmbedding
from src.weight_generators import MultiComponentGenerator


class DynamicHyperNetwork(nn.Module):
    """
    Dynamic HyperNetwork that generates task-specific weights for ARCCompressor.

    This module combines:
    1. Puzzle embeddings (task-specific latent codes)
    2. Weight generators (produce network parameters from embeddings)
    3. Integration with ARCCompressor architecture

    Args:
        num_puzzles (int): Number of unique ARC puzzles
        z_dim (int): Dimension of task embeddings
        n_layers (int): Number of transformer layers in ARCCompressor
        use_metadata (bool): Whether to condition on task metadata
        use_hierarchical (bool): Whether to use hierarchical embeddings
        channel_dims (dict): Channel dimensions for each component
    """

    def __init__(
        self,
        num_puzzles: int = 1000,
        z_dim: int = 64,
        n_layers: int = 4,
        use_metadata: bool = True,
        use_hierarchical: bool = False,
        channel_dims: Optional[Dict] = None
    ):
        super(DynamicHyperNetwork, self).__init__()

        self.num_puzzles = num_puzzles
        self.z_dim = z_dim
        self.n_layers = n_layers
        self.use_metadata = use_metadata
        self.use_hierarchical = use_hierarchical

        # Default channel dimensions from ARCCompressor
        if channel_dims is None:
            channel_dims = {
                'share_up_dim': 16,
                'share_down_dim': 8,
                'decoding_dim': 4,
                'softmax_dim': 2,
                'cummax_dim': 4,
                'shift_dim': 4,
                'nonlinear_dim': 16,
            }
        self.channel_dims = channel_dims

        # Puzzle embedding module
        if use_hierarchical:
            self.puzzle_embedding = HierarchicalPuzzleEmbedding(
                num_puzzles, z_dim, num_levels=3
            )
        else:
            self.puzzle_embedding = PuzzleEmbedding(
                num_puzzles, z_dim, use_meta_features=use_metadata
            )

        # Weight generator modules
        self.weight_generator = MultiComponentGenerator(
            z_dim, n_layers, channel_dims
        )

        # Cache for generated weights (optional optimization)
        self._weight_cache = {}
        self._cache_enabled = False

    def enable_cache(self):
        """Enable weight caching for inference speed."""
        self._cache_enabled = True
        self._weight_cache = {}

    def disable_cache(self):
        """Disable weight caching (required during training)."""
        self._cache_enabled = False
        self._weight_cache = {}

    def clear_cache(self):
        """Clear the weight cache."""
        self._weight_cache = {}

    def get_task_embedding(
        self,
        task_name: str,
        task_metadata: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Get task embedding for a specific puzzle.

        Args:
            task_name (str): Unique identifier for the ARC task
            task_metadata (dict): Optional metadata from Task object

        Returns:
            torch.Tensor: Task embedding [1, z_dim]
        """
        # Convert task name to consistent puzzle ID
        puzzle_id = hash(task_name) % self.num_puzzles
        puzzle_id_tensor = torch.tensor([puzzle_id], dtype=torch.long)
        puzzle_id_tensor = puzzle_id_tensor.to(next(self.parameters()).device)

        # Get embedding
        if self.use_metadata and task_metadata is not None:
            return self.puzzle_embedding(puzzle_id_tensor, task_metadata)
        else:
            return self.puzzle_embedding(puzzle_id_tensor)

    def generate_weights(
        self,
        task_name: str,
        task_metadata: Optional[Dict] = None,
        use_cache: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Generate all network weights for a specific task.

        Args:
            task_name (str): Task identifier
            task_metadata (dict): Optional task metadata
            use_cache (bool): Whether to use cached weights

        Returns:
            dict: Generated weights for all components
        """
        # Check cache if enabled
        if use_cache and self._cache_enabled and task_name in self._weight_cache:
            return self._weight_cache[task_name]

        # Get task embedding
        z = self.get_task_embedding(task_name, task_metadata)
        z = z.squeeze(0)  # Remove batch dimension

        # Generate weights
        weights = self.weight_generator(z)

        # Cache if enabled
        if self._cache_enabled:
            self._weight_cache[task_name] = weights

        return weights

    def forward(
        self,
        task_name: str,
        task_metadata: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass: generate task embedding and weights.

        Args:
            task_name (str): Task identifier
            task_metadata (dict): Optional task metadata

        Returns:
            embedding (torch.Tensor): Task embedding [1, z_dim]
            weights (dict): Generated weights for all components
        """
        # Get task embedding
        embedding = self.get_task_embedding(task_name, task_metadata)

        # Generate weights
        weights = self.generate_weights(task_name, task_metadata)

        return embedding, weights

    def get_num_parameters(self, generated_weights: Dict = None) -> int:
        """
        Calculate total number of parameters in the hypernetwork and generated weights.

        Args:
            generated_weights (dict): Optional pre-generated weights to count

        Returns:
            int: Total parameter count
        """
        # Count hypernetwork parameters
        hyper_params = sum(p.numel() for p in self.parameters())

        # Count generated parameters if provided
        generated_params = 0
        if generated_weights is not None:
            for key, weight in generated_weights.items():
                if isinstance(weight, torch.Tensor):
                    generated_params += weight.numel()
                elif isinstance(weight, dict):
                    for subkey, subweight in weight.items():
                        if isinstance(subweight, torch.Tensor):
                            generated_params += subweight.numel()

        return hyper_params + generated_params

    def inject_weights_into_model(
        self,
        model,
        generated_weights: Dict[str, torch.Tensor],
        task_name: str
    ):
        """
        Inject generated weights into an ARCCompressor model.

        Args:
            model: ARCCompressor instance
            generated_weights (dict): Generated weights from this hypernetwork
            task_name (str): Task identifier for logging
        """
        # This method would be implemented to properly inject weights
        # into the ARCCompressor architecture
        # For now, it's a placeholder for the integration interface

        # Example structure:
        # model.decode_weights = generated_weights['decode_weights']
        # for layer_idx in range(self.n_layers):
        #     layer_key = f"layer_{layer_idx}"
        #     model.share_up_weights[layer_idx] = generated_weights[layer_key]['share_up']
        #     ...

        raise NotImplementedError(
            "Weight injection requires modifications to ARCCompressor to accept external weights"
        )


class HyperNetworkTrainer:
    """
    Training utilities for the Dynamic HyperNetwork.
    Handles meta-learning across multiple ARC tasks.

    Args:
        hypernetwork (DynamicHyperNetwork): The hypernetwork to train
        base_model_class: ARCCompressor class (not instance)
        learning_rate (float): Learning rate for optimization
    """

    def __init__(
        self,
        hypernetwork: DynamicHyperNetwork,
        base_model_class,
        learning_rate: float = 1e-3
    ):
        self.hypernetwork = hypernetwork
        self.base_model_class = base_model_class
        self.learning_rate = learning_rate

        # Optimizer for hypernetwork parameters only
        self.optimizer = torch.optim.Adam(
            self.hypernetwork.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Training statistics
        self.train_losses = []
        self.val_losses = []

    def train_step(
        self,
        task_name: str,
        task_obj,
        task_metadata: Dict
    ) -> float:
        """
        Single training step on one task.

        Args:
            task_name (str): Task identifier
            task_obj: Task object from pre_processing
            task_metadata (dict): Task metadata

        Returns:
            float: Loss value
        """
        self.hypernetwork.train()
        self.optimizer.zero_grad()

        # Generate task-specific weights
        embedding, weights = self.hypernetwork(task_name, task_metadata)

        # Create task-specific model with generated weights
        # This is where you'd integrate with ARCCompressor
        # For now, this is a placeholder for the training interface

        # Placeholder loss computation
        # In practice, this would involve:
        # 1. Create ARCCompressor with generated weights
        # 2. Run forward pass on task data
        # 3. Compute reconstruction loss + KL divergence
        # 4. Backpropagate through hypernetwork

        loss = torch.tensor(0.0, requires_grad=True)  # Placeholder

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.hypernetwork.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def validate(self, val_tasks: list) -> float:
        """
        Validation on multiple tasks.

        Args:
            val_tasks (list): List of validation tasks

        Returns:
            float: Average validation loss
        """
        self.hypernetwork.eval()
        total_loss = 0.0

        with torch.no_grad():
            for task in val_tasks:
                # Validation logic here
                pass

        avg_loss = total_loss / len(val_tasks) if val_tasks else 0.0
        return avg_loss

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.hypernetwork.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> Dict:
        """Load training checkpoint."""
        checkpoint = torch.load(path)
        self.hypernetwork.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        return checkpoint
