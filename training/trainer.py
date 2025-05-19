import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from torch.utils.data import DataLoader
import wandb
# Assuming collate_fn is defined and imported correctly in your project setup.
from data.dataset import collate_fn # Example import

# Removed imports for BeamSearch and other metrics as they are no longer used for per-epoch evaluation
from data.tokenizer import CharTokenizer # Import CharTokenizer to access special token methods

class Trainer:
    """Trainer for sequence-to-sequence model with per-epoch evaluation."""

    def __init__(self, model, config, train_dataset, val_dataset): # Added datasets to init
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
         # Ignore padding tokens (assuming 0 is PAD ID)
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        self.stats = []

        # Store datasets and tokenizers for evaluation (needed for pad_id)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.source_tokenizer = train_dataset.source_tokenizer # Access tokenizers from dataset
        self.target_tokenizer = train_dataset.target_tokenizer
        self.pad_token_id = self.target_tokenizer.pad_id() # Get padding ID
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token_id) 
        # BeamSearch is no longer initialized here as it's not used for per-epoch evaluation metrics


    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        for batch_idx, (source, target) in enumerate(dataloader):
            source, target = source.to(self.device), target.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            # The model's forward method handles teacher forcing
            outputs, _ = self.model(source, target, self.config.teacher_forcing_ratio)

            # Reshape for loss calculation
            # outputs shape: [batch_size, target_len, target_vocab_size]
            # target shape: [batch_size, target_len]
            batch_size = outputs.shape[0]
            seq_len = outputs.shape[1]
            outputs = outputs.reshape(-1, outputs.shape[2]) # Flatten batch and sequence dimensions
            target = target.reshape(-1) # Flatten batch and sequence dimensions

            # Calculate loss
            # criterion expects outputs [N, C] and target [N]
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

    def evaluate_loss_and_accuracy(self, dataloader):
        """Calculate loss and token accuracy.""" # Renamed for clarity
        self.model.eval()
        total_loss = 0
        correct_tokens = 0
        total_tokens = 0

        with torch.no_grad():
            for source, target in dataloader:
                source, target = source.to(self.device), target.to(self.device)

                # Forward pass without teacher forcing (teacher_forcing_ratio=0)
                outputs, _ = self.model(source, target, teacher_forcing_ratio=0)

                # Reshape for loss calculation
                outputs_flat = outputs.reshape(-1, outputs.shape[2]) # [N, C]
                target_flat = target.reshape(-1) # [N]

                # Calculate loss
                loss = self.criterion(outputs_flat, target_flat)
                total_loss += loss.item()

                # Calculate token accuracy
                # Get predicted token IDs
                predicted_tokens = outputs_flat.argmax(dim=1) # [N]

                # Create a mask to ignore padding tokens in the target
                non_pad_mask = (target_flat != self.pad_token_id)

                # Apply mask to both predicted and target tokens
                predicted_tokens_masked = predicted_tokens[non_pad_mask]
                target_flat_masked = target_flat[non_pad_mask]

                # Count correct predictions among non-padding tokens
                correct_tokens += (predicted_tokens_masked == target_flat_masked).sum().item()
                total_tokens += non_pad_mask.sum().item()


        avg_loss = total_loss / len(dataloader)
        token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0

        return avg_loss, token_accuracy # Return both loss and accuracy


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

        # Create a dataloader for validation loss and token accuracy calculation
        val_dataloader = DataLoader( # Renamed dataloader
            val_dataset,
            batch_size=self.config.batch_size, # Use batch size from config
            shuffle=False,
            collate_fn=collate_fn
        )

        # Removed eval_metrics_dataloader as it's no longer needed


        # Training loop
        print("-" * 100)
        for epoch in range(1, self.config.epochs + 1):
            print(f"EPOCH {epoch}\n")

            # Training
            print("Training...\n")
            start_time = time.time()
            train_loss = self.train_epoch(train_dataloader)

            # Validation Loss and Token Accuracy
            print("\nCalculating Validation Loss and Token Accuracy...")
            # Use the single validation dataloader for both loss and accuracy
            val_loss, val_token_accuracy = self.evaluate_loss_and_accuracy(val_dataloader) # Use the new method


            # Calculate training time
            time_taken = time.time() - start_time

            # Removed call to evaluate_metrics

            # Store statistics
            epoch_stats = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_token_accuracy": val_token_accuracy, # Add token accuracy to stats
                # Removed char/word accuracy and BLEU from stats
                "training_time": time_taken
            }
            self.stats.append(epoch_stats)

            # Print statistics
            print(f"\nTrain Loss: {train_loss:.4f} Validation Loss: {val_loss:.4f} Validation Token Accuracy: {val_token_accuracy:.4f}") # Print token accuracy
            # Removed printing of char/word accuracy and BLEU
            print(f"\nTime taken for the epoch: {time_taken:.4f}")
            print("-" * 100)

            # Log to WandB if enabled
            if self.config.use_wandb:
                print("Logging epoch stats to WandB...")
                wandb.log(epoch_stats)
                print("Epoch stats logged.")


        print("\nModel trained successfully!")

    def evaluate(self, test_dataset):
        """Evaluate the model on test data (using loss and token accuracy).""" # Updated docstring
        print("\nEvaluating on test set...")
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size, # Use batch size from config
            shuffle=False,
            collate_fn=collate_fn
        )

        # Use the evaluate_loss_and_accuracy method for test set evaluation
        test_loss, test_token_accuracy = self.evaluate_loss_and_accuracy(test_dataloader) # Get both loss and accuracy

        print(f"Test Loss: {test_loss:.4f} Test Token Accuracy: {test_token_accuracy:.4f}") # Print test loss and accuracy

        # Log test metrics to WandB if the run is active
        if wandb.run:
             print("Logging test metrics to WandB...")
             wandb.log({
                 "test_loss": test_loss, # Log test loss
                 "test_token_accuracy": test_token_accuracy, # Log test token accuracy
                 # Removed logging of test char/word accuracy and BLEU
             })
             print("Test metrics logged.")

        return test_loss, test_token_accuracy # Return test loss and token accuracy

