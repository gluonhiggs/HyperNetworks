"""
Training script for Dynamic HyperNetwork on ARC tasks.
Implements meta-learning across multiple puzzles.
"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import argparse
import os
import sys
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dynamic_hypernetwork import DynamicHyperNetwork
from src.arc_hypercompressor import (
    HyperCompressorFactory,
    TaskMetadataExtractor,
    HyperCompressorCheckpoint
)
from pre_processing import preprocess_tasks


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Dynamic HyperNetwork for ARC')

    # Model configuration
    parser.add_argument('--num_puzzles', type=int, default=1000,
                        help='Number of unique puzzles')
    parser.add_argument('--z_dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Number of layers')
    parser.add_argument('--use_hierarchical', action='store_true',
                        help='Use hierarchical embeddings')

    # Training configuration
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (number of tasks per batch)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping norm')

    # Data configuration
    parser.add_argument('--data_dir', type=str, default='dataset/',
                        help='Directory containing ARC dataset')
    parser.add_argument('--split', type=str, default='training',
                        choices=['training', 'evaluation'],
                        help='Dataset split to use')
    parser.add_argument('--num_tasks', type=int, default=None,
                        help='Number of tasks to train on (None = all)')
    parser.add_argument('--task_names', type=str, nargs='+', default=None,
                        help='Specific task names to train on')

    # Optimization
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to train on')

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Logging
    parser.add_argument('--log_every', type=int, default=10,
                        help='Log progress every N steps')

    return parser.parse_args()


class HyperNetworkMetaTrainer:
    """
    Meta-trainer for Dynamic HyperNetwork across ARC tasks.
    """

    def __init__(
        self,
        model: DynamicHyperNetwork,
        args,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.args = args
        self.device = device

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Mixed precision scaler
        self.scaler = GradScaler() if args.use_amp else None

        # Training statistics
        self.epoch = 0
        self.global_step = 0
        self.train_losses = []
        self.best_loss = float('inf')

    def train_on_task(self, task, task_name: str):
        """
        Train hypernetwork on a single task.

        Args:
            task: Task object from pre_processing
            task_name (str): Task identifier

        Returns:
            float: Loss value
        """
        self.model.train()

        # Extract task metadata
        task_metadata = TaskMetadataExtractor.extract(task)

        # Generate task-specific weights
        if self.args.use_amp:
            with autocast():
                embedding, weights = self.model(task_name, task_metadata)
                # Placeholder loss - in real implementation:
                # 1. Create ARCCompressor with generated weights
                # 2. Forward pass on task data
                # 3. Compute ELBO loss
                loss = embedding.abs().mean()  # Dummy loss
        else:
            embedding, weights = self.model(task_name, task_metadata)
            loss = embedding.abs().mean()  # Dummy loss

        return loss

    def train_epoch(self, tasks):
        """
        Train for one epoch across all tasks.

        Args:
            tasks (list): List of Task objects

        Returns:
            float: Average epoch loss
        """
        epoch_loss = 0.0
        num_tasks = len(tasks)

        pbar = tqdm(tasks, desc=f'Epoch {self.epoch + 1}/{self.args.num_epochs}')

        for task in pbar:
            self.optimizer.zero_grad()

            # Forward pass
            loss = self.train_on_task(task, task.task_name)

            # Backward pass
            if self.args.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.args.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.args.grad_clip
                )
                self.optimizer.step()

            # Update statistics
            epoch_loss += loss.item()
            self.global_step += 1

            # Logging
            if self.global_step % self.args.log_every == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / num_tasks
        return avg_loss

    def train(self, tasks):
        """
        Main training loop.

        Args:
            tasks (list): List of Task objects to train on
        """
        print(f"Starting training on {len(tasks)} tasks...")
        print(f"Model has {sum(p.numel() for p in self.model.parameters())} parameters")

        for epoch in range(self.args.num_epochs):
            self.epoch = epoch

            # Train epoch
            avg_loss = self.train_epoch(tasks)
            self.train_losses.append(avg_loss)

            # Learning rate scheduling
            self.scheduler.step(avg_loss)

            # Logging
            print(f"Epoch {epoch + 1}/{self.args.num_epochs} - "
                  f"Avg Loss: {avg_loss:.4f} - "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Save checkpoint
            if (epoch + 1) % self.args.save_every == 0:
                self.save_checkpoint(f'hypernet_epoch{epoch + 1}.pt')

            # Save best model
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint('hypernet_best.pt')
                print(f"  ✓ New best model saved (loss: {avg_loss:.4f})")

        print("Training complete!")
        print(f"Best loss: {self.best_loss:.4f}")

    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.args.checkpoint_dir, filename)

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'best_loss': self.best_loss,
            'args': vars(self.args)
        }

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.best_loss = checkpoint['best_loss']

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"Checkpoint loaded from: {path}")
        print(f"  Resuming from epoch {self.epoch + 1}")


def main():
    """Main training function."""
    args = parse_args()

    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load tasks
    print(f"Loading tasks from {args.data_dir}...")
    if args.task_names:
        tasks = preprocess_tasks(args.split, args.task_names)
    elif args.num_tasks:
        tasks = preprocess_tasks(args.split, list(range(args.num_tasks)))
    else:
        tasks = preprocess_tasks(args.split, list(range(800)))  # All training tasks

    print(f"Loaded {len(tasks)} tasks")

    # Create model
    print("Creating hypernetwork...")
    model = DynamicHyperNetwork(
        num_puzzles=args.num_puzzles,
        z_dim=args.z_dim,
        n_layers=args.n_layers,
        use_metadata=True,
        use_hierarchical=args.use_hierarchical
    )

    # Create trainer
    trainer = HyperNetworkMetaTrainer(model, args, device)

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(tasks)


if __name__ == '__main__':
    main()
