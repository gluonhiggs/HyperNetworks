"""
Puzzle Embedding Module - Learnable task-specific embeddings for ARC puzzles.
Generates unique latent representations for each puzzle to condition weight generation.
"""
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class PuzzleEmbedding(nn.Module):
    """
    Learnable embedding module that produces task-specific latent vectors.
    Each puzzle gets a unique embedding vector that conditions the hypernetwork.

    Args:
        num_puzzles (int): Number of unique ARC puzzles
        embedding_dim (int): Dimension of the embedding vector (z_dim)
        use_meta_features (bool): Whether to concatenate metadata features
    """

    def __init__(self, num_puzzles: int, embedding_dim: int = 64, use_meta_features: bool = True):
        super(PuzzleEmbedding, self).__init__()
        self.num_puzzles = num_puzzles
        self.embedding_dim = embedding_dim
        self.use_meta_features = use_meta_features

        # Learnable embedding lookup table
        self.embeddings = nn.Embedding(num_puzzles, embedding_dim)

        # Initialize with normal distribution
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.02)

        # Optional: Linear projection for metadata features
        if use_meta_features:
            # Metadata: n_examples, n_colors, n_x, n_y, in_out_same_size, etc.
            self.meta_dim = 8
            self.meta_projection = nn.Linear(self.meta_dim, embedding_dim)
            nn.init.xavier_uniform_(self.meta_projection.weight)
            nn.init.zeros_(self.meta_projection.bias)

    def forward(self, puzzle_ids: torch.Tensor, metadata: dict = None):
        """
        Generate task-specific embeddings.

        Args:
            puzzle_ids (torch.Tensor): Tensor of puzzle indices [batch_size]
            metadata (dict): Optional metadata features for the puzzle
                - n_examples, n_colors, n_x, n_y, in_out_same_size, etc.

        Returns:
            torch.Tensor: Task embeddings [batch_size, embedding_dim]
        """
        # Get base embeddings from lookup table
        base_embeddings = self.embeddings(puzzle_ids)

        if self.use_meta_features and metadata is not None:
            # Extract metadata features
            meta_features = self._extract_metadata_features(metadata)
            meta_embeddings = self.meta_projection(meta_features)

            # Combine base embeddings with metadata
            combined_embeddings = base_embeddings + meta_embeddings
            return combined_embeddings

        return base_embeddings

    def _extract_metadata_features(self, metadata: dict) -> torch.Tensor:
        """
        Extract and normalize metadata features from task.

        Args:
            metadata (dict): Task metadata from pre_processing.Task

        Returns:
            torch.Tensor: Normalized metadata features [batch_size, meta_dim]
        """
        features = []

        # Normalize numerical features
        features.append(metadata.get('n_examples', 0) / 10.0)  # Typically 2-10 examples
        features.append(metadata.get('n_colors', 0) / 10.0)    # Typically 1-10 colors
        features.append(metadata.get('n_x', 0) / 30.0)         # Max grid dimension ~30
        features.append(metadata.get('n_y', 0) / 30.0)
        features.append(metadata.get('n_train', 0) / 10.0)
        features.append(metadata.get('n_test', 0) / 10.0)

        # Binary features
        features.append(float(metadata.get('in_out_same_size', False)))
        features.append(float(metadata.get('all_in_same_size', False)))

        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)

        # Add batch dimension if needed
        if features_tensor.dim() == 1:
            features_tensor = features_tensor.unsqueeze(0)

        return features_tensor.to(next(self.parameters()).device)

    def get_embedding_for_task(self, task_name: str, task_metadata: dict = None):
        """
        Convenience method to get embedding for a specific task by name.

        Args:
            task_name (str): Task identifier
            task_metadata (dict): Optional metadata from Task object

        Returns:
            torch.Tensor: Task embedding [1, embedding_dim]
        """
        # In practice, you'd maintain a task_name -> puzzle_id mapping
        # For now, we'll hash the task name to get a consistent ID
        puzzle_id = hash(task_name) % self.num_puzzles
        puzzle_id_tensor = torch.tensor([puzzle_id], dtype=torch.long)
        puzzle_id_tensor = puzzle_id_tensor.to(next(self.parameters()).device)

        return self.forward(puzzle_id_tensor, task_metadata)


class HierarchicalPuzzleEmbedding(nn.Module):
    """
    Hierarchical embedding that captures multiple levels of task abstraction.
    Useful for transfer learning across similar puzzles.

    Args:
        num_puzzles (int): Number of unique puzzles
        embedding_dim (int): Total embedding dimension
        num_levels (int): Number of hierarchy levels (e.g., 3 for task/type/difficulty)
    """

    def __init__(self, num_puzzles: int, embedding_dim: int = 64, num_levels: int = 3):
        super(HierarchicalPuzzleEmbedding, self).__init__()
        self.num_puzzles = num_puzzles
        self.embedding_dim = embedding_dim
        self.num_levels = num_levels

        # Divide embedding dimension across levels
        self.level_dim = embedding_dim // num_levels

        # Create embeddings for each level
        self.level_embeddings = nn.ModuleList([
            nn.Embedding(num_puzzles, self.level_dim)
            for _ in range(num_levels)
        ])

        # Initialize all embeddings
        for embedding in self.level_embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)

        # Optional: Non-linear combination
        self.combiner = nn.Sequential(
            nn.Linear(self.level_dim * num_levels, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU()
        )

    def forward(self, puzzle_ids: torch.Tensor):
        """
        Generate hierarchical embeddings.

        Args:
            puzzle_ids (torch.Tensor): Puzzle indices [batch_size]

        Returns:
            torch.Tensor: Combined embeddings [batch_size, embedding_dim]
        """
        # Get embeddings from each level
        level_outputs = [
            level_emb(puzzle_ids)
            for level_emb in self.level_embeddings
        ]

        # Concatenate across levels
        combined = torch.cat(level_outputs, dim=-1)

        # Non-linear combination
        output = self.combiner(combined)

        return output
