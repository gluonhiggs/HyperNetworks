"""
ARCHyperCompressor - Integration layer between Dynamic HyperNetwork and ARCCompressor.
Provides factory methods and utilities for creating hypernetwork-driven models.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dynamic_hypernetwork import DynamicHyperNetwork
from pre_processing import Task


class HyperCompressorFactory:
    """
    Factory class for creating Dynamic HyperNetwork configurations.
    Provides common presets (lightweight, default, large).
    """

    @staticmethod
    def create_lightweight(device: str = 'cuda') -> DynamicHyperNetwork:
        """
        Lightweight configuration for faster training/inference.

        Args:
            device (str): Device to place model on

        Returns:
            DynamicHyperNetwork: Configured model
        """
        model = DynamicHyperNetwork(
            num_puzzles=500,
            z_dim=32,
            n_layers=4,
            use_metadata=True,
            use_hierarchical=False
        )
        return model.to(device)

    @staticmethod
    def create_default(device: str = 'cuda') -> DynamicHyperNetwork:
        """
        Default balanced configuration.

        Args:
            device (str): Device to place model on

        Returns:
            DynamicHyperNetwork: Configured model
        """
        model = DynamicHyperNetwork(
            num_puzzles=1000,
            z_dim=64,
            n_layers=4,
            use_metadata=True,
            use_hierarchical=False
        )
        return model.to(device)

    @staticmethod
    def create_large(device: str = 'cuda') -> DynamicHyperNetwork:
        """
        Large configuration for maximum expressiveness.

        Args:
            device (str): Device to place model on

        Returns:
            DynamicHyperNetwork: Configured model
        """
        model = DynamicHyperNetwork(
            num_puzzles=2000,
            z_dim=128,
            n_layers=6,
            use_metadata=True,
            use_hierarchical=True
        )
        return model.to(device)

    @staticmethod
    def create_custom(
        num_puzzles: int,
        z_dim: int,
        n_layers: int = 4,
        use_metadata: bool = True,
        use_hierarchical: bool = False,
        device: str = 'cuda'
    ) -> DynamicHyperNetwork:
        """
        Create custom configuration.

        Args:
            num_puzzles (int): Number of unique puzzles
            z_dim (int): Embedding dimension
            n_layers (int): Number of layers
            use_metadata (bool): Use task metadata
            use_hierarchical (bool): Use hierarchical embeddings
            device (str): Device to place model on

        Returns:
            DynamicHyperNetwork: Configured model
        """
        model = DynamicHyperNetwork(
            num_puzzles=num_puzzles,
            z_dim=z_dim,
            n_layers=n_layers,
            use_metadata=use_metadata,
            use_hierarchical=use_hierarchical
        )
        return model.to(device)


class TaskMetadataExtractor:
    """
    Utility class for extracting metadata from Task objects.
    Converts Task attributes to dictionary format for hypernetwork input.
    """

    @staticmethod
    def extract(task: Task) -> Dict:
        """
        Extract all relevant metadata from a Task object.

        Args:
            task (Task): Task object from pre_processing

        Returns:
            dict: Metadata dictionary
        """
        metadata = {
            'n_examples': task.n_examples,
            'n_colors': task.n_colors,
            'n_x': task.n_x,
            'n_y': task.n_y,
            'n_train': task.n_train,
            'n_test': task.n_test,
            'in_out_same_size': task.in_out_same_size,
            'all_in_same_size': task.all_in_same_size,
            'all_out_same_x': task.all_out_same_x,
            'all_out_same_y': task.all_out_same_y,
        }
        return metadata


class HyperCompressorCheckpoint:
    """
    Utilities for saving and loading hypernetwork models.
    """

    @staticmethod
    def save(
        model: DynamicHyperNetwork,
        path: str,
        metadata: Optional[Dict] = None
    ):
        """
        Save hypernetwork model.

        Args:
            model (DynamicHyperNetwork): Model to save
            path (str): Save path
            metadata (dict): Optional metadata to include
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'num_puzzles': model.num_puzzles,
            'z_dim': model.z_dim,
            'n_layers': model.n_layers,
            'use_metadata': model.use_metadata,
            'use_hierarchical': model.use_hierarchical,
            'channel_dims': model.channel_dims,
            'metadata': metadata or {}
        }
        torch.save(checkpoint, path)

    @staticmethod
    def load(path: str, device: str = 'cuda') -> DynamicHyperNetwork:
        """
        Load hypernetwork model.

        Args:
            path (str): Path to checkpoint
            device (str): Device to load onto

        Returns:
            DynamicHyperNetwork: Loaded model
        """
        checkpoint = torch.load(path, map_location=device)

        # Recreate model with saved configuration
        model = DynamicHyperNetwork(
            num_puzzles=checkpoint['num_puzzles'],
            z_dim=checkpoint['z_dim'],
            n_layers=checkpoint['n_layers'],
            use_metadata=checkpoint['use_metadata'],
            use_hierarchical=checkpoint['use_hierarchical'],
            channel_dims=checkpoint.get('channel_dims')
        )

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        return model


class WeightInjector:
    """
    Helper class for injecting generated weights into ARCCompressor.
    NOTE: Requires modifications to ARCCompressor to accept external weights.
    """

    @staticmethod
    def prepare_weights_for_arccompressor(
        generated_weights: Dict,
        task: Task
    ) -> Dict:
        """
        Convert hypernetwork-generated weights to ARCCompressor format.

        Args:
            generated_weights (dict): Raw weights from hypernetwork
            task (Task): Task object for dimension information

        Returns:
            dict: Formatted weights ready for ARCCompressor
        """
        # This would implement the conversion logic
        # from hypernetwork output format to ARCCompressor's expected format

        # For now, this is a placeholder
        # Real implementation would need to:
        # 1. Reshape tensors to match MultiTensor structure
        # 2. Apply symmetrization constraints
        # 3. Ensure proper device placement
        # 4. Match exact naming conventions

        formatted_weights = {}
        # ... conversion logic here ...

        return formatted_weights

    @staticmethod
    def inject_into_model(
        arc_model,
        generated_weights: Dict,
        task: Task
    ):
        """
        Inject weights into an ARCCompressor instance.

        Args:
            arc_model: ARCCompressor instance
            generated_weights (dict): Generated weights
            task (Task): Task object

        NOTE: This requires ARCCompressor to be modified to accept external weights.
        """
        formatted_weights = WeightInjector.prepare_weights_for_arccompressor(
            generated_weights, task
        )

        # Would inject weights here
        # arc_model.decode_weights = formatted_weights['decode_weights']
        # arc_model.share_up_weights = formatted_weights['share_up_weights']
        # etc.

        raise NotImplementedError(
            "ARCCompressor must be modified to accept external weights. "
            "Add an 'external_weights' parameter to ARCCompressor.__init__()."
        )
