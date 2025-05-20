import torch
import argparse
import os
import wandb
import random
import numpy as np

from data.dataset import load_data
from data.tokenizer import CharTokenizer
from model.seq2seq import Seq2SeqModel
from training.trainer import Trainer
from training.beam_search import BeamSearch
from utils.metrics import character_accuracy, word_accuracy, bleu_score
from config import Config
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def set_seed(seed):
    """Set random seed for reproducibility.
    
    Args:
        seed (int): Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def sweep_train():
    """Create and return a function for W&B sweep training.
    
    Returns:
        function: Training function for W&B sweep agent.
    """

    def train():
        """Training function to be used by W&B sweep agent.
        
        Handles the complete training workflow including:
        - Loading data
        - Creating the model
        - Setting up the trainer
        - Training the model
        - Evaluating on validation set
        - Logging metrics to W&B
        """
        with wandb.init() as run:
            config = wandb.config

            print(f"Starting run with config: {config}")

            set_seed(42)

            data_path = getattr(config, 'data_path', './data')
            language = getattr(config, 'language', 'hi')

            default_config = Config()
            sos_token = default_config.sos_token
            eos_token = default_config.eos_token

            print(f"Loading data from {data_path} for language {language} with SOS='{sos_token}', EOS='{eos_token}'...")
            try:
                train_dataset, val_dataset, test_dataset = load_data(
                    data_path, language,
                    sos_token=sos_token,
                    eos_token=eos_token
                )
                if not hasattr(train_dataset, 'source_tokenizer') or not hasattr(train_dataset, 'target_tokenizer'):
                     raise AttributeError("load_data did not return datasets with tokenizers")
                print("Data loaded successfully.")
            except Exception as e:
                print(f"Error loading data: {e}")
                if wandb.run:
                    wandb.log({"error": f"Data loading failed: {e}"})
                return

            print("Creating model...")
            try:
                model = Seq2SeqModel(
                    source_vocab_size=len(train_dataset.source_tokenizer),
                    target_vocab_size=len(train_dataset.target_tokenizer),
                    embedding_dim=config.embedding_dim,
                    hidden_size=config.hidden_size,
                    num_layers=config.num_rnn_layers,
                    dropout=config.dropout,
                    rnn_type=config.rnn_type,
                    attention=config.attention
                )
                model.to(device)
                print("Model created successfully.")
            except Exception as e:
                print(f"Error creating model: {e}")
                if wandb.run:
                    wandb.log({"error": f"Model creation failed: {e}"})
                return

            run_config = Config()
            run_config.device = str(device)

            run_config.language = language
            run_config.embedding_dim = config.embedding_dim
            run_config.hidden_size = config.hidden_size
            run_config.encoder_layers = config.num_rnn_layers
            run_config.decoder_layers = config.num_rnn_layers
            run_config.dropout = config.dropout
            run_config.rnn_type = config.rnn_type
            run_config.attention = config.attention
            run_config.batch_size = config.batch_size
            run_config.epochs = config.epochs
            run_config.learning_rate = config.learning_rate

            run_config.sos_token = default_config.sos_token
            run_config.eos_token = default_config.eos_token
            run_config.teacher_forcing_ratio = default_config.teacher_forcing_ratio
            run_config.beam_width = config.beam_width
            run_config.wandb_project = run.project
            run_config.use_wandb = True

            print("Initializing Trainer...")
            try:
                trainer = Trainer(model, run_config, train_dataset, val_dataset)
                print("Trainer initialized.")
            except Exception as e:
                print(f"Error initializing Trainer: {e}")
                if wandb.run:
                    wandb.log({"error": f"Trainer initialization failed: {e}"})
                return

            print("Training model...")
            try:
                trainer.fit(train_dataset, val_dataset)
                print("Training finished by trainer.fit().")

            except Exception as e:
                print(f"Error during training: {e}")
                if wandb.run:
                    wandb.log({"error": f"Training failed: {e}"})
                return

            print("\n--- Starting Validation Evaluation Block ---")
            print("Evaluating on validation set...")
            try:
                model.eval()
                beam_searcher = BeamSearch(model, beam_width=config.beam_width)

                val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=run_config.batch_size,
                    shuffle=False,
                    collate_fn=collate_fn
                )

                total_char_accuracy = 0.0
                total_word_accuracy = 0.0
                num_examples = 0

                all_target_texts = []
                all_predicted_texts = []

                source_tokenizer = train_dataset.source_tokenizer
                target_tokenizer = train_dataset.target_tokenizer

                print(f"Starting validation evaluation loop over {len(val_dataloader)} batches...")
                with torch.no_grad():
                    for batch_idx, (source_tensors, target_tensors) in enumerate(val_dataloader):

                        for i in range(source_tensors.size(0)):
                            source_tensor = source_tensors[i].unsqueeze(0).to(device)
                            target_tensor = target_tensors[i].to(device)

                            source_text = source_tokenizer.decode([token.item() for token in source_tensor.squeeze(0) if token.item() > 0])
                            target_text = target_tokenizer.decode([token.item() for token in target_tensor if token.item() > 0])

                            if not source_text or not target_text or source_text == source_tokenizer.sos_token or target_text == target_tokenizer.eos_token:
                                continue

                            results = beam_searcher.translate(
                                source_text,
                                source_tokenizer,
                                target_tokenizer,
                                device=device
                            )

                            if results:
                                predicted_text = results[0][0]
                            else:
                                predicted_text = ""

                            char_acc = character_accuracy(predicted_text, target_text)
                            word_acc = word_accuracy(predicted_text, target_text)

                            total_char_accuracy += char_acc
                            total_word_accuracy += word_acc
                            num_examples += 1

                            all_target_texts.append(target_text)
                            all_predicted_texts.append(predicted_text)

                        if (batch_idx + 1) % 10 == 0:
                            print(f"Evaluation Batch {batch_idx + 1}/{len(val_dataloader)} processed. Examples processed so far: {num_examples}")


                print(f"Total examples processed for evaluation: {num_examples}")
                avg_char_accuracy = total_char_accuracy / num_examples if num_examples > 0 else 0.0
                avg_word_accuracy = total_word_accuracy / num_examples if num_examples > 0 else 0.0

                try:
                     target_references = [[word for word in target.split()] for target in all_target_texts]
                     predicted_candidates = [predicted.split() for predicted in all_predicted_texts]
                     print("SRC:", source_text)
                     print("TGT:", target_text)
                     print("PRED:", predicted_text)
                     if predicted_candidates and target_references:
                         bleu = bleu_score(predicted_candidates, target_references)
                     else:
                         print("No valid examples for BLEU calculation.")
                         bleu = 0.0

                except Exception as bleu_e:
                     print(f"Error calculating BLEU score: {bleu_e}")
                     bleu = 0.0


                print(f"Validation Character Accuracy: {avg_char_accuracy:.4f}")
                print(f"Validation Word Accuracy: {avg_word_accuracy:.4f}")
                print(f"Validation BLEU Score: {bleu:.4f}")

                if wandb.run:
                     print("Logging validation metrics to WandB...")
                     wandb.log({
                         "val_char_accuracy": avg_char_accuracy,
                         "val_word_accuracy": avg_word_accuracy,
                         "val_bleu_score": bleu
                     })
                     print("Validation metrics logged.")
                else:
                    print("WandB run not active, skipping metric logging.")


                print("Validation evaluation finished.")

            except Exception as e:
                print(f"An unexpected Error occurred during validation evaluation: {e}")
                import traceback
                traceback.print_exc()
                if wandb.run:
                    wandb.log({"error": f"Validation evaluation failed: {e}"})

            run.name = f"emb{config.embedding_dim}_hid{config.hidden_size}_lyr{config.num_rnn_layers}_{config.rnn_type}_dr{config.dropout:.3f}_att{config.attention}_lr{config.learning_rate:.5f}_ep{config.epochs}_bs{config.batch_size}_beam{config.beam_width}"
            print(f"Wandb run name: {run.name}")

    return train

def collate_fn(batch):
    """Collate function for DataLoader to handle variable length sequences.
    
    Args:
        batch (list): List of (source_tensor, target_tensor) tuples from the Dataset.
        
    Returns:
        tuple: (padded_sources, padded_targets) with consistent shapes.
    """
    sources, targets = zip(*batch)

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

    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'embedding_dim': {'values': [16, 32, 64, 128, 256]},
            'hidden_size': {'values': [16, 32, 64, 128, 256]},
            'num_rnn_layers': {'values': [1, 2, 3]},
            'rnn_type': {'values': ['rnn', 'gru', 'lstm']},
            'dropout': {'distribution': 'uniform', 'min': 0.1, 'max': 0.5},
            'attention': {'values': [True]},
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-3},
            'epochs': {'distribution': 'int_uniform', 'min': 2, 'max': 25},
            'batch_size': {'values': [64, 128, 256]},
            'beam_width': {'values': [1, 2, 3]},
            'data_path': {'value': args.data_path},
            'language': {'value': args.language},
        }
    }

    print("Weights & Biases Sweep Configuration:")
    import yaml
    print(yaml.dump(sweep_config, default_flow_style=False))

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

    print(f"Starting W&B agent for sweep ID '{sweep_id}' for {args.sweep_count} runs...")
    try:
        wandb.agent(sweep_id, function=sweep_train(), count=args.sweep_count)
        print("Sweep finished.")
    except Exception as e:
        print(f"Error running W&B agent: {e}")
        print("Ensure the 'function' passed to wandb.agent is callable and your training script runs without errors.")
