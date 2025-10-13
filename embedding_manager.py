import os
import torch
import json


class EmbeddingManager:
    """
    Manages puzzle embeddings for multiple ARC-AGI tasks.

    This class handles:
        - Creating new embeddings for tasks
        - Loading/saving embeddings from/to disk
        - Tracking which tasks have embeddings
        - Batch operations on embeddings

    Usage:
        manager = EmbeddingManager(emb_dim=128, save_dir='./embeddings')
        emb = manager.get_or_create_embedding('task_001')
        manager.save_embedding('task_001', emb)
        manager.load_all_embeddings()
    """

    def __init__(self, emb_dim=128, save_dir='./embeddings', device='cuda'):
        """
        Initialize embedding manager.

        Args:
            emb_dim: Dimension of puzzle embeddings
            save_dir: Directory to save/load embeddings
            device: Device to store embeddings on
        """
        self.emb_dim = emb_dim
        self.save_dir = save_dir
        self.device = device
        self.embeddings = {}  # task_name -> embedding tensor

        os.makedirs(save_dir, exist_ok=True)

    def create_embedding(self, task_name, init_std=0.01):
        """
        Create a new random embedding for a task.

        Args:
            task_name: Name of the task
            init_std: Standard deviation for random initialization

        Returns:
            embedding: nn.Parameter of shape [emb_dim]
        """
        embedding = torch.nn.Parameter(
            torch.randn(self.emb_dim, device=self.device) * init_std
        )
        self.embeddings[task_name] = embedding
        return embedding

    def get_embedding(self, task_name):
        """
        Get embedding for a task (returns None if not exists).

        Args:
            task_name: Name of the task

        Returns:
            embedding: Embedding tensor or None
        """
        return self.embeddings.get(task_name)

    def get_or_create_embedding(self, task_name, init_std=0.01):
        """
        Get embedding if exists, otherwise create new one.

        Args:
            task_name: Name of the task
            init_std: Standard deviation for random initialization if creating

        Returns:
            embedding: Embedding tensor
        """
        if task_name in self.embeddings:
            return self.embeddings[task_name]
        else:
            return self.create_embedding(task_name, init_std)

    def save_embedding(self, task_name, embedding=None):
        """
        Save embedding for a task to disk.

        Args:
            task_name: Name of the task
            embedding: Embedding tensor (if None, uses self.embeddings[task_name])
        """
        if embedding is None:
            embedding = self.embeddings[task_name]

        save_path = os.path.join(self.save_dir, f'{task_name}.pth')
        torch.save({
            'embedding': embedding.detach().cpu(),
            'task_name': task_name,
            'emb_dim': self.emb_dim
        }, save_path)

    def load_embedding(self, task_name):
        """
        Load embedding for a task from disk.

        Args:
            task_name: Name of the task

        Returns:
            embedding: Loaded embedding tensor
        """
        load_path = os.path.join(self.save_dir, f'{task_name}.pth')
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No embedding found for task {task_name} at {load_path}")

        checkpoint = torch.load(load_path)
        embedding = torch.nn.Parameter(checkpoint['embedding'].to(self.device))
        self.embeddings[task_name] = embedding
        return embedding

    def save_all_embeddings(self):
        """Save all embeddings in memory to disk."""
        for task_name, embedding in self.embeddings.items():
            self.save_embedding(task_name, embedding)
        print(f"Saved {len(self.embeddings)} embeddings to {self.save_dir}")

    def load_all_embeddings(self):
        """Load all embeddings from save directory."""
        embedding_files = [f for f in os.listdir(self.save_dir) if f.endswith('.pth')]
        for filename in embedding_files:
            task_name = filename[:-4]  # Remove .pth extension
            try:
                self.load_embedding(task_name)
            except Exception as e:
                print(f"Error loading {task_name}: {e}")

        print(f"Loaded {len(self.embeddings)} embeddings from {self.save_dir}")

    def get_all_embeddings_as_batch(self, task_names=None):
        """
        Get embeddings for multiple tasks as a batch tensor.

        Args:
            task_names: List of task names (if None, uses all tasks)

        Returns:
            embeddings_batch: Tensor of shape [n_tasks, emb_dim]
            task_names: List of corresponding task names
        """
        if task_names is None:
            task_names = list(self.embeddings.keys())

        embeddings_list = [self.embeddings[name] for name in task_names]
        embeddings_batch = torch.stack(embeddings_list, dim=0)

        return embeddings_batch, task_names

    def save_metadata(self):
        """Save metadata about all embeddings."""
        metadata = {
            'emb_dim': self.emb_dim,
            'n_embeddings': len(self.embeddings),
            'task_names': list(self.embeddings.keys())
        }
        metadata_path = os.path.join(self.save_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")

    def __len__(self):
        """Return number of embeddings stored."""
        return len(self.embeddings)

    def __contains__(self, task_name):
        """Check if task has an embedding."""
        return task_name in self.embeddings

    def __repr__(self):
        return f"EmbeddingManager(emb_dim={self.emb_dim}, n_tasks={len(self)}, save_dir={self.save_dir})"
