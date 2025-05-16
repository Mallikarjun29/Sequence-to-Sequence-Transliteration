import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from torch.utils.data import DataLoader
import wandb

class Trainer:
    """Trainer for sequence-to-sequence model."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        self.stats = []
        
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (source, target) in enumerate(dataloader):
            source, target = source.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = self.model(source, target, self.config.teacher_forcing_ratio)
            
            # Reshape for loss calculation
            batch_size = outputs.shape[0]
            seq_len = outputs.shape[1]
            outputs = outputs.reshape(-1, outputs.shape[2])
            target = target.reshape(-1)
            
            # Calculate loss
            loss = self.criterion(outputs, target)
            
            # Backpropagation
            loss.backward()
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            
            # Print progress
            if batch_idx == 0 or (batch_idx + 1) % 100 == 0:
                print(f"Batch {batch_idx + 1} Loss {loss.item():.4f}")
                
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        """Validate model."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for source, target in dataloader:
                source, target = source.to(self.device), target.to(self.device)
                
                # Forward pass
                outputs, _ = self.model(source, target, teacher_forcing_ratio=0)
                
                # Reshape for loss calculation
                outputs = outputs.reshape(-1, outputs.shape[2])
                target = target.reshape(-1)
                
                # Calculate loss
                loss = self.criterion(outputs, target)
                total_loss += loss.item()
                
        return total_loss / len(dataloader)
    
    def fit(self, train_dataset, val_dataset):
        """Train the model."""
        print(f"Training on {self.device}")
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            collate_fn=collate_fn
        )
        
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Initialize WandB if enabled
        if self.config.use_wandb:
            wandb.init(project=self.config.wandb_project)
            wandb.config.update(self.config.__dict__)
        
        # Training loop
        print("-" * 100)
        for epoch in range(1, self.config.epochs + 1):
            print(f"EPOCH {epoch}\n")
            
            # Training
            print("Training...\n")
            start_time = time.time()
            train_loss = self.train_epoch(train_dataloader)
            
            # Validation
            print("\nValidating...")
            val_loss = self.validate(val_dataloader)
            
            # Calculate training time
            time_taken = time.time() - start_time
            
            # Store statistics
            epoch_stats = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "training_time": time_taken
            }
            self.stats.append(epoch_stats)
            
            # Print statistics
            print(f"\nTrain Loss: {train_loss:.4f} Validation Loss: {val_loss:.4f}")
            print(f"\nTime taken for the epoch: {time_taken:.4f}")
            print("-" * 100)
            
            # Log to WandB if enabled
            if self.config.use_wandb:
                wandb.log(epoch_stats)
        
        print("\nModel trained successfully!")
        
    def evaluate(self, test_dataset):
        """Evaluate the model on test data."""
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        test_loss = self.validate(test_dataloader)
        print(f"Test Loss: {test_loss:.4f}")
        
        return test_loss
