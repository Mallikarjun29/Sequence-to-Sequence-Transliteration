import torch
import argparse
import os
import wandb
import random
import numpy as np

# Import necessary components from your project structure
from data.dataset import load_data # Assuming load_data is in data/dataset.py
# Assuming CharTokenizer is used by load_data or needed elsewhere
from data.tokenizer import CharTokenizer # Assuming CharTokenizer is in data/tokenizer.py
from model.seq2seq import Seq2SeqModel # Assuming Seq2SeqModel is in model/seq2seq.py
from training.trainer import Trainer # Assuming Trainer is in training/trainer.py
# BeamSearch and metrics are for evaluation
from training.beam_search import BeamSearch # Assuming BeamSearch is in training/beam_search.py
from utils.metrics import character_accuracy, word_accuracy, bleu_score # Assuming metrics are in utils/metrics.py
from config import Config # Import your Config class
from torch.utils.data import DataLoader

# Import collate_fn if it's used by your DataLoader in trainer.py
# Assuming collate_fn is defined and imported correctly in your project setup.
# If it's in data.dataset.py alongside load_data, you can import it like this:
# from data.dataset import load_data, collate_fn
# If it's elsewhere, adjust the import path.
# For this example, I will assume it's defined or imported correctly in your environment.
# If you copied collate_fn into the sweep script directly before, keep that.
# Assuming collate_fn exists in the scope where DataLoader is used.
# from data.dataset import collate_fn # Example import

# Add device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Define the function that will be executed by the sweep agent
def sweep_train():

    # This function will be called once per sweep configuration
    def train():
        # Initialize a new W&B run for this configuration
        # The project and entity are set by wandb.agent based on the sweep definition
        with wandb.init() as run:
            # Get the current configuration from wandb (contains sweep parameters and fixed values)
            config = wandb.config

            print(f"Starting run with config: {config}")

            # Set random seed for reproducibility within the run
            set_seed(42) # Using a fixed seed for each run in the sweep

            # --- Load data ---
            data_path = getattr(config, 'data_path', './data')
            language = getattr(config, 'language', 'hi')

            # Create a default Config instance to get default fixed parameters
            default_config = Config()
            sos_token = default_config.sos_token
            eos_token = default_config.eos_token

            print(f"Loading data from {data_path} for language {language} with SOS='{sos_token}', EOS='{eos_token}'...")
            try:
                # Use the tokens obtained from the default config instance
                train_dataset, val_dataset, test_dataset = load_data(
                    data_path, language,
                    sos_token=sos_token,
                    eos_token=eos_token
                )
                # Ensure tokenizers are available (load_data should provide them)
                if not hasattr(train_dataset, 'source_tokenizer') or not hasattr(train_dataset, 'target_tokenizer'):
                     raise AttributeError("load_data did not return datasets with tokenizers")
                print("Data loaded successfully.")
            except Exception as e:
                print(f"Error loading data: {e}")
                if wandb.run:
                    wandb.log({"error": f"Data loading failed: {e}"})
                    # wandb.run.finish(exit_code=1) # Optional: Mark run as failed
                return # Exit the train function
            # --- End Load data ---


            # --- Create model ---
            print("Creating model...")
            try:
                model = Seq2SeqModel(
                    source_vocab_size=len(train_dataset.source_tokenizer),
                    target_vocab_size=len(train_dataset.target_tokenizer),
                    embedding_dim=config.embedding_dim,
                    hidden_size=config.hidden_size,
                    # Note: main.py uses num_layers for both encoder/decoder.
                    # We tune a single 'num_rnn_layers' sweep parameter.
                    num_layers=config.num_rnn_layers,
                    dropout=config.dropout,
                    rnn_type=config.rnn_type,
                    attention=config.attention
                )
                model.to(device) # Ensure model is on the correct device
                print("Model created successfully.")
            except Exception as e:
                print(f"Error creating model: {e}")
                if wandb.run:
                    wandb.log({"error": f"Model creation failed: {e}"})
                    # wandb.run.finish(exit_code=1) # Optional: Mark run as failed
                return # Exit the train function
            # --- End Create model ---


            # --- Create and populate Config object for Trainer ---
            # Create a new Config instance to pass to the Trainer
            run_config = Config()
            # Populate it with device (determined by script)
            run_config.device = str(device)

            # Populate with sweep parameters from wandb.config
            run_config.language = language # From sweep config (fixed)
            run_config.embedding_dim = config.embedding_dim
            run_config.hidden_size = config.hidden_size
            # Use the same for encoder/decoder layers based on main.py structure
            run_config.encoder_layers = config.num_rnn_layers
            run_config.decoder_layers = config.num_rnn_layers
            run_config.dropout = config.dropout
            run_config.rnn_type = config.rnn_type
            run_config.attention = config.attention
            run_config.batch_size = config.batch_size
            run_config.epochs = config.epochs
            run_config.learning_rate = config.learning_rate # Added learning_rate to run_config

            # Populate with fixed parameters from the default Config instance
            # These are not in the sweep config but needed by Trainer/Model config
            run_config.sos_token = default_config.sos_token
            run_config.eos_token = default_config.eos_token
            run_config.teacher_forcing_ratio = default_config.teacher_forcing_ratio
            run_config.beam_width = config.beam_width
            run_config.wandb_project = run.project # Get project name from the W&B run
            run_config.use_wandb = True # Enable wandb logging in Trainer

            # --- End Create Config object for Trainer ---

            # Initialize the Trainer
            print("Initializing Trainer...")
            try:
                # Trainer automatically moves model to device in __init__ and sets up criterion/optimizer
                trainer = Trainer(model, run_config, train_dataset, val_dataset)
                # Watch model parameters and gradients if needed (optional, can add overhead)
                # if wandb.run:
                #     wandb.watch(model, log="all")
                print("Trainer initialized.")
            except Exception as e:
                print(f"Error initializing Trainer: {e}")
                if wandb.run:
                    wandb.log({"error": f"Trainer initialization failed: {e}"})
                    # wandb.run.finish(exit_code=1) # Optional: Mark run as failed
                return # Exit the train function


            # --- Train model ---
            print("Training model...")
            try:
                # The Trainer handles the training loop and logging to wandb (train_loss, val_loss)
                trainer.fit(train_dataset, val_dataset)
                print("Training finished by trainer.fit().") # Debug print after trainer.fit

            except Exception as e:
                print(f"Error during training: {e}")
                if wandb.run:
                    wandb.log({"error": f"Training failed: {e}"})
                return # Exit the train function
            # --- End Train model ---

            # --- Evaluate on VALIDATION set with accuracy and BLEU ---
            print("\n--- Starting Validation Evaluation Block ---") # Debug print BEFORE evaluation
            print("Evaluating on validation set...")
            try:
                model.eval() # Set model to evaluation mode
                beam_searcher = BeamSearch(model, beam_width=config.beam_width)

                # Use val_dataset for evaluation
                val_dataloader = DataLoader(
                    val_dataset, # <--- Changed from test_dataset to val_dataset
                    batch_size=run_config.batch_size, # Use batch size from config
                    shuffle=False, # No need to shuffle validation data
                    collate_fn=collate_fn # Use your collate_fn
                )

                total_char_accuracy = 0.0
                total_word_accuracy = 0.0
                num_examples = 0

                all_target_texts = []
                all_predicted_texts = []

                # Get tokenizers from the training dataset (they are the same for val)
                source_tokenizer = train_dataset.source_tokenizer
                target_tokenizer = train_dataset.target_tokenizer

                print(f"Starting validation evaluation loop over {len(val_dataloader)} batches...") # Debug print
                with torch.no_grad(): # Disable gradient calculation for evaluation
                    # Iterate through the validation dataloader
                    for batch_idx, (source_tensors, target_tensors) in enumerate(val_dataloader): # <--- Changed from test_dataloader to val_dataloader

                        # Process each example in the batch individually for beam search
                        for i in range(source_tensors.size(0)):
                            source_tensor = source_tensors[i].unsqueeze(0).to(device) # Add batch dim and move to device
                            target_tensor = target_tensors[i].to(device) # Move target to device

                            # Decode source and target tensors to strings (removing padding)
                            # Assuming padding_value=0, and SOS/EOS are handled by tokenizer
                            source_text = source_tokenizer.decode([token.item() for token in source_tensor.squeeze(0) if token.item() > 0])
                            target_text = target_tokenizer.decode([token.item() for token in target_tensor if token.item() > 0])

                            # Skip empty source or target texts if any
                            if not source_text or not target_text or source_text == source_tokenizer.sos_token or target_text == target_tokenizer.eos_token:
                                # print(f"Skipping example {i} in batch {batch_idx}: Empty source or target.") # Debug print
                                continue

                            # Perform beam search translation
                            # beam_searcher.translate expects string input
                            results = beam_searcher.translate(
                                source_text,
                                source_tokenizer, # Pass source tokenizer
                                target_tokenizer, # Pass target tokenizer
                                device=device
                            )

                            # Get the best predicted sequence (highest score)
                            if results:
                                predicted_text = results[0][0] # results is list of (text, score) tuples
                            else:
                                predicted_text = "" # Handle case where beam search yields no results

                            # Calculate character and word accuracy for this example
                            char_acc = character_accuracy(predicted_text, target_text)
                            word_acc = word_accuracy(predicted_text, target_text)

                            total_char_accuracy += char_acc
                            total_word_accuracy += word_acc
                            num_examples += 1

                            # Store for BLEU calculation
                            all_target_texts.append(target_text)
                            all_predicted_texts.append(predicted_text)

                            # Debug print for a few examples
                            # if num_examples <= 5: # Print details for the first 5 successful examples
                            #      print(f"  Example {num_examples}: Source='{source_text}', Target='{target_text}', Predicted='{predicted_text}', CharAcc={char_acc:.4f}, WordAcc={word_acc:.4f}")


                        # Optional: Print progress during evaluation
                        if (batch_idx + 1) % 10 == 0:
                            print(f"Evaluation Batch {batch_idx + 1}/{len(val_dataloader)} processed. Examples processed so far: {num_examples}") # <--- Updated print


                # Calculate average accuracies
                print(f"Total examples processed for evaluation: {num_examples}") # Debug print
                avg_char_accuracy = total_char_accuracy / num_examples if num_examples > 0 else 0.0
                avg_word_accuracy = total_word_accuracy / num_examples if num_examples > 0 else 0.0

                # Calculate BLEU score
                try:
                     # Note: BLEU score calculation can be sensitive to tokenization.
                     # The character-based tokenization might not be ideal for standard word-based BLEU.
                     # You might need to split by spaces for word BLEU if your metric function expects words.
                     # Assuming your bleu_score function takes lists of strings.
                     # Ensure all_target_texts are lists of words for BLEU calculation
                     target_references = [[word for word in target.split()] for target in all_target_texts]
                     predicted_candidates = [predicted.split() for predicted in all_predicted_texts]
                        # inside your validation loop
                     print("SRC:", source_text)
                     print("TGT:", target_text)
                     print("PRED:", predicted_text)
                     # Check if there are any valid candidates/references for BLEU calculation
                     if predicted_candidates and target_references:
                         bleu = bleu_score(predicted_candidates, target_references) # BLEU usually uses tokenized sentences
                     else:
                         print("No valid examples for BLEU calculation.") # Debug print
                         bleu = 0.0

                except Exception as bleu_e:
                     print(f"Error calculating BLEU score: {bleu_e}")
                     bleu = 0.0 # Set BLEU to 0 if calculation fails


                print(f"Validation Character Accuracy: {avg_char_accuracy:.4f}") # <--- Updated print
                print(f"Validation Word Accuracy: {avg_word_accuracy:.4f}") # <--- Updated print
                print(f"Validation BLEU Score: {bleu:.4f}")

                # Check if wandb run is active before logging
                if wandb.run:
                     print("Logging validation metrics to WandB...") # Debug print
                     wandb.log({
                         "val_char_accuracy": avg_char_accuracy, # <--- Updated log key
                         "val_word_accuracy": avg_word_accuracy, # <--- Updated log key
                         "val_bleu_score": bleu                  # <--- Updated log key
                     })
                     print("Validation metrics logged.") # Debug print
                else:
                    print("WandB run not active, skipping metric logging.") # Debug print


                print("Validation evaluation finished.") # <--- Updated print

            except Exception as e:
                print(f"An unexpected Error occurred during validation evaluation: {e}") # <--- Updated print with more detail
                import traceback
                traceback.print_exc() # Print the full traceback for debugging
                if wandb.run:
                    wandb.log({"error": f"Validation evaluation failed: {e}"}) # <--- Updated log key
                # No return here, training was successful, evaluation just failed


            # --- End Evaluate on VALIDATION set ---


            # Set run name for wandb based on hyperparameters
            run.name = f"emb{config.embedding_dim}_hid{config.hidden_size}_lyr{config.num_rnn_layers}_{config.rnn_type}_dr{config.dropout:.3f}_att{config.attention}_lr{config.learning_rate:.5f}_ep{config.epochs}_bs{config.batch_size}_beam{config.beam_width}"
            print(f"Wandb run name: {run.name}")

    # Return the inner function
    return train

# Assuming collate_fn is defined globally or imported here if needed by DataLoader
# You must ensure your actual collate_fn is available here.
# from data.dataset import collate_fn # Example import if in data.dataset
def collate_fn(batch):
    # Batch is a list of (source_tensor, target_tensor) tuples from the Dataset
    sources, targets = zip(*batch)

    # Pad sequences - assuming padding token index is 0
    sources_padded = torch.nn.utils.rnn.pad_sequence(sources, batch_first=True, padding_value=0)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)

    return sources_padded, targets_padded


if __name__ == "__main__":
    """
    Main entry point for the script.

    Sets up the sweep configuration and starts the wandb agent to perform hyperparameter optimization.
    """

    parser = argparse.ArgumentParser(description="W&B Sweep for Seq2Seq Transliteration")
    parser.add_argument("--data_path", type=str, default="",
                        help="Path to data directory")
    parser.add_argument("--language", type=str, default="hi",
                        help="Language code (e.g., 'hi' for Hindi)")
    parser.add_argument("--sweep_count", type=int, default=50,
                        help="Number of sweep runs to perform")
    parser.add_argument("--entity", type=str, default="da24s009-indiam-institute-of-technology-madras",
                        help="W&B entity name")
    parser.add_argument("--project", type=str, default="da6401_assignment_3_seq2seq_attention",
                        help="W&B project name")

    args = parser.parse_args()

    # Define the sweep configuration
    sweep_config = {
        'method': 'bayes', # Using bayesian optimization
        'metric': {
            # Based on trainer.py logging, optimize validation loss
            'name': 'val_loss', # <--- Keeping val_loss as the primary optimization metric
            'goal': 'minimize'
        },
        'parameters': {
            # Hyperparameters to tune
            'embedding_dim': {'values': [16, 32, 64, 128, 256]},
            'hidden_size': {'values': [16, 32, 64, 128, 256]},
            # Tuning a single layer count used for both encoder/decoder as per main.py model
            'num_rnn_layers': {'values': [1, 2, 3]},
            'rnn_type': {'values': ['rnn', 'gru', 'lstm']},
            # Tuning dropout rate between 20% and 30%
            'dropout': {'distribution': 'uniform', 'min': 0.1, 'max': 0.5},
            'attention': {'values': [True]}, # Whether to use attention
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-3},
            # Adjusted epoch range - feel free to increase for more thorough search if time permits
            'epochs': {'distribution': 'int_uniform', 'min': 2, 'max': 25},
            'batch_size': {'values': [64, 128, 256]}, # Tuning batch size
            'beam_width': {'values': [1, 2, 3]},
            # Fixed parameters passed to the training function via wandb.config
            # These are not tuned but made available in the config object
            'data_path': {'value': args.data_path},
            'language': {'value': args.language},
            # Note: sos_token, eos_token, teacher_forcing_ratio, beam_width, wandb_project
            # are now taken from a *default* Config instance inside the train function,
            # as they are instance attributes.
        }
    }

    print("Weights & Biases Sweep Configuration:")
    import yaml
    print(yaml.dump(sweep_config, default_flow_style=False))

    # Initialize sweep
    print(f"Initializing W&B sweep for project '{args.project}' under entity '{args.entity}'...")
    try:
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            entity=args.entity,
            project=args.project
        )
        print(f"Sweep ID: {sweep_id}")
    except Exception as e:
        print(f"Error initializing W&B sweep: {e}")
        print("Please check your W&B API key, entity, and project name.")
        exit()

    # Run sweep
    print(f"Starting W&B agent for sweep ID '{sweep_id}' for {args.sweep_count} runs...")
    try:
        # The function passed to wandb.agent is the inner 'train' function
        wandb.agent(sweep_id, function=sweep_train(), count=args.sweep_count)
        print("Sweep finished.")
    except Exception as e:
        print(f"Error running W&B agent: {e}")
        print("Ensure the 'function' passed to wandb.agent is callable and your training script runs without errors.")
