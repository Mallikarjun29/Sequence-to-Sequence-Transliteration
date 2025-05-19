"""
Compare vanilla and attention-based seq2seq models:
1. Load vanilla and attention model predictions
2. Compare performance and find error corrections
3. Generate attention heatmaps for selected examples
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import argparse
import seaborn as sns
from matplotlib.gridspec import GridSpec
from tabulate import tabulate
from tqdm import tqdm
from datetime import datetime
import wandb

# Import project modules
from model.seq2seq import Seq2SeqModel
from data.dataset import load_data, collate_fn
from training.beam_search import BeamSearch
from utils.metrics import character_accuracy, word_accuracy
from config import Config


def load_model_predictions(predictions_dir, model_type):
    """Load model predictions from CSV file"""
    # Find the most recent predictions file for this model type
    files = [f for f in os.listdir(predictions_dir) if f.startswith(f"test_predictions_{model_type}")]
    if not files:
        raise FileNotFoundError(f"No prediction files found for {model_type} model.")
    
    # Sort by timestamp (assuming format: test_predictions_TYPE_TIMESTAMP.csv)
    files.sort(key=lambda x: x.split('_')[-1].split('.')[0], reverse=True)
    
    # Load the most recent file
    csv_path = os.path.join(predictions_dir, files[0])
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} predictions from {csv_path}")
    return df


def compare_models(vanilla_df, attention_df, vanilla_config, attention_config):
    """Compare performance between vanilla and attention models"""
    # Calculate overall metrics
    vanilla_char_acc = vanilla_df['char_acc'].mean()
    vanilla_word_acc = vanilla_df['word_acc'].mean()
    attention_char_acc = attention_df['char_acc'].mean()
    attention_word_acc = attention_df['word_acc'].mean()
    
    # Calculate perfect predictions
    vanilla_perfect = sum(vanilla_df['word_acc'] == 1.0)
    attention_perfect = sum(attention_df['word_acc'] == 1.0)
    
    # Print comparison
    print("\n" + "="*50)
    print("MODEL ARCHITECTURE COMPARISON")
    print("="*50)
    
    architecture_table = [
        ["Parameter", "Vanilla Model", "Attention Model"],
        ["RNN Type", f"{vanilla_config.rnn_type.upper()}", f"{attention_config.rnn_type.upper()}"],
        ["Hidden Size", f"{vanilla_config.hidden_size}", f"{attention_config.hidden_size}"],
        ["Embedding Dim", f"{vanilla_config.embedding_dim}", f"{attention_config.embedding_dim}"],
        ["Encoder Layers", f"{vanilla_config.encoder_layers}", f"{attention_config.encoder_layers}"],
        ["Dropout", f"{vanilla_config.dropout}", f"{attention_config.dropout}"],
        ["Batch Size", f"{vanilla_config.batch_size}", f"{attention_config.batch_size}"],
        ["Learning Rate", f"{vanilla_config.learning_rate}", f"{attention_config.learning_rate}"],
        ["Attention", "No", "Yes"]
    ]
    
    print(tabulate(architecture_table, headers="firstrow", tablefmt="fancy_grid"))
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*50)
    
    comparison_table = [
        ["Metric", "Vanilla Model", "Attention Model", "Improvement"],
        ["Character Accuracy", f"{vanilla_char_acc:.4f}", f"{attention_char_acc:.4f}", f"{attention_char_acc - vanilla_char_acc:+.4f}"],
        ["Word Accuracy", f"{vanilla_word_acc:.4f}", f"{attention_word_acc:.4f}", f"{attention_word_acc - vanilla_word_acc:+.4f}"],
        ["Perfect Predictions", f"{vanilla_perfect} ({vanilla_perfect/len(vanilla_df):.2%})", 
         f"{attention_perfect} ({attention_perfect/len(attention_df):.2%})", 
         f"{attention_perfect - vanilla_perfect:+d} ({(attention_perfect - vanilla_perfect)/len(vanilla_df):+.2%})"]
    ]
    
    print(tabulate(comparison_table, headers="firstrow", tablefmt="fancy_grid"))
    
    return vanilla_char_acc, vanilla_word_acc, attention_char_acc, attention_word_acc


def find_error_corrections(vanilla_df, attention_df):
    """Find examples where attention model corrected errors made by vanilla model"""
    # Merge dataframes on source and target
    merged_df = pd.merge(vanilla_df, attention_df, 
                         on=['source', 'target'], 
                         suffixes=('_vanilla', '_attention'))
    
    # Find instances where vanilla was wrong and attention was right
    corrections = merged_df[(merged_df['word_acc_vanilla'] < 1.0) & (merged_df['word_acc_attention'] == 1.0)]
    
    # Find instances where attention significantly improved over vanilla
    improvements = merged_df[(merged_df['word_acc_attention'] - merged_df['word_acc_vanilla'] >= 0.5)]
    
    # Sort by improvement magnitude
    improvements = improvements.sort_values(by='word_acc_vanilla', ascending=True)
    
    print(f"\nFound {len(corrections)} complete error corrections (vanilla wrong, attention correct)")
    print(f"Found {len(improvements)} significant improvements (word_acc diff ≥ 0.5)")
    
    # Print selected examples with largest improvements
    print("\n" + "="*50)
    print("EXAMPLES OF ERROR CORRECTIONS")
    print("="*50)
    
    # Combine and take at most 10 examples - using dataframe operations instead of Python list operations
    examples_df = pd.concat([corrections.head(5), improvements.head(5)]).drop_duplicates()
    examples_df = examples_df.head(10)  # Limit to 10 examples
    
    # Convert to list of dictionaries for easier handling
    examples = examples_df.to_dict('records')
    
    for i, row in enumerate(examples):
        print(f"\nExample {i+1}:")
        print(f"Source:           '{row['source']}'")
        print(f"Target:           '{row['target']}'")
        print(f"Vanilla output:   '{row['predicted_vanilla']}' (Word Acc: {row['word_acc_vanilla']:.2f})")
        print(f"Attention output: '{row['predicted_attention']}' (Word Acc: {row['word_acc_attention']:.2f})")
        print(f"Improvement:      +{row['word_acc_attention'] - row['word_acc_vanilla']:.2f}")
    
    return corrections, improvements, examples


def load_model_and_generate_attention_maps(config, examples, output_dir):
    """Load attention model and generate attention heatmaps for selected examples"""
    # Create directory for attention maps
    attn_dir = os.path.join(output_dir, "attention_visualizations")
    os.makedirs(attn_dir, exist_ok=True)
    
    print("\n" + "="*50)
    print(f"GENERATING ATTENTION HEATMAPS USING {config.rnn_type.upper()} MODEL")
    print("="*50)
    
    # Load datasets to get tokenizers
    try:
        train_dataset, _, _ = load_data(
            config.data_path, config.language,
            sos_token=config.sos_token,
            eos_token=config.eos_token
        )
        source_tokenizer = train_dataset.source_tokenizer
        target_tokenizer = train_dataset.target_tokenizer
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Create model with attention and specific RNN type
    model = Seq2SeqModel(
        source_vocab_size=len(source_tokenizer),
        target_vocab_size=len(target_tokenizer),
        embedding_dim=config.embedding_dim,
        hidden_size=config.hidden_size,
        num_layers=config.encoder_layers,  # Using encoder_layers for both encoder and decoder
        dropout=config.dropout,
        rnn_type=config.rnn_type,
        attention=True
    )
    
    print(f"Model created with: {model.encoder.num_layers} encoder layers and {model.decoder.num_layers} decoder layers")
    print(f"Hidden size: {config.hidden_size}, RNN type: {config.rnn_type}")
    
    # Load best model weights
    model_path = os.path.join(output_dir, "models", "best_model_attention.pt")
    if not os.path.exists(model_path):
        print(f"Error: Attention model checkpoint not found at {model_path}")
        return
    
    try:
        # Try loading with weights_only=True first
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"Loaded attention model from {model_path} with weights_only=True")
    except Exception as e1:
        try:
            # If that fails, try the default loading method
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded attention model from {model_path}")
        except Exception as e2:
            print(f"Error loading model: {e2}")
            return
    
    # Set model to evaluation mode and move to device
    device = torch.device(config.device)
    model.eval()
    model.to(device)
    
    # Create the attention visualization grid with larger figure size for better readability
    examples_to_plot = examples[:9]  # Only take first 9 for 3x3 grid
    if len(examples_to_plot) < 9:
        print(f"Warning: Only {len(examples_to_plot)} examples available for visualization.")
        rows = cols = int(np.ceil(np.sqrt(len(examples_to_plot))))
    else:
        rows, cols = 3, 3
    
    # Use larger figure size for better readability
    fig = plt.figure(figsize=(6*cols, 5*rows))
    gs = GridSpec(rows, cols, figure=fig)
    
    # WORKAROUND: Set font that works well with Roman characters
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    for i, example in enumerate(examples_to_plot):
        if i >= rows*cols:
            break
            
        row, col = i // cols, i % cols
        ax = fig.add_subplot(gs[row, col])
        
        # Generate attention weights for this example
        source_text = example['source']
        target_text = example['target'] 
        
        # Get source tokens
        source_tokens = source_tokenizer.encode(source_text)
        source_tensor = torch.tensor(source_tokens).unsqueeze(0).to(device)
        
        # Forward pass with attention capture
        attention_weights = generate_attention_map(model, source_tensor, target_tokenizer, device)
        
        # Skip if attention weights could not be generated
        if attention_weights is None:
            continue
        
        # WORKAROUND: Use position indices instead of actual characters to avoid font issues
        source_tokens_display = [f"S{i}" for i in range(len(source_text))]
        target_tokens_display = [f"T{i}" for i in range(len(example['predicted_attention']))]
        
        # Ensure attention weights match the token lengths
        attention_to_plot = attention_weights[:len(target_tokens_display), :len(source_tokens_display)]
        
        # Plot heatmap
        sns.heatmap(attention_to_plot, annot=False, cmap="viridis", ax=ax,
                   xticklabels=source_tokens_display, yticklabels=target_tokens_display)
        
        # Include more details in the title - use position references instead of actual text
        title = f"Example {i+1}\n"
        title += f"Source length: {len(source_text)}, Target length: {len(target_text)}\n"
        title += f"Prediction length: {len(example['predicted_attention'])}\n"
        title += f"Word Acc: {example['word_acc_attention']:.2f} (Vanilla: {example['word_acc_vanilla']:.2f})"
        
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Source character positions")
        ax.set_ylabel("Target character positions")
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add a suptitle explaining what attention does
    plt.suptitle("Attention Mechanism Visualizations: How the model focuses on different source positions\n" +
                "when generating each target character", fontsize=16)
    
    # WORKAROUND: Add text box with example legend outside the plots
    legend_text = "Examples Key (Position number → character):\n"
    for i, example in enumerate(examples_to_plot):
        if i >= 3:  # Just show first 3 examples in the legend
            break
        source_map = ", ".join([f"S{j}→{c}" for j, c in enumerate(example['source'][:5])])
        target_map = ", ".join([f"T{j}→{c}" for j, c in enumerate(example['predicted_attention'][:5])])
        legend_text += f"Ex{i+1}: {source_map}..., {target_map}...\n"
    
    # Add text box with mappings
    plt.figtext(0.5, 0.01, legend_text, ha='center', va='bottom', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
                fontsize=12)
    
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.08, 1, 0.97])
    
    # Save at high resolution
    save_path = os.path.join(attn_dir, "attention_heatmaps_grid.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved attention heatmap grid to {save_path}")
    plt.close()


def generate_attention_map(model, source_tensor, target_tokenizer, device):
    """Generate attention weights for visualization using beam search"""
    model.eval()
    
    with torch.no_grad():
        try:
            # Get encoder outputs
            encoder_outputs, encoder_hidden = model.encoder(source_tensor)
            
            # Initialize decoder hidden state - fix hidden size mismatch
            # For bidirectional encoder or different layer counts, we need to process the hidden state
            if isinstance(encoder_hidden, tuple):  # LSTM returns tuple (hidden, cell)
                # Handle LSTM hidden state
                hidden, cell = encoder_hidden
                
                # For bidirectional encoder or different layer counts
                if hidden.size(0) != model.decoder.num_layers:
                    # Adjust hidden state: take last layers or duplicate as needed
                    decoder_hidden_h = hidden[-model.decoder.num_layers:]
                    decoder_hidden_c = cell[-model.decoder.num_layers:]
                    decoder_hidden = (decoder_hidden_h, decoder_hidden_c)
                else:
                    decoder_hidden = encoder_hidden
            else:
                # Handle GRU/RNN hidden state
                if encoder_hidden.size(0) != model.decoder.num_layers:
                    # Take the last n layers where n is decoder.num_layers
                    decoder_hidden = encoder_hidden[-model.decoder.num_layers:]
                else:
                    decoder_hidden = encoder_hidden
            
            # Start with SOS token 
            if hasattr(target_tokenizer, 'token_to_id'):
                # Use token_to_id if available
                sos_token_id = target_tokenizer.token_to_id[target_tokenizer.sos_token]
            else:
                # Otherwise, encode the SOS token
                sos_token_id = target_tokenizer.encode(target_tokenizer.sos_token)[0]
            
            decoder_input = torch.tensor([[sos_token_id]]).to(device)
            
            # Collect attention weights for each step
            attention_weights = []
            output_tokens = []
            
            # Maximum prediction length
            max_length = 50
            
            for _ in range(max_length):
                # Forward through decoder
                decoder_output, decoder_hidden, attention_weight = model.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                
                # Get prediction
                top_token = decoder_output.argmax(-1)
                output_tokens.append(top_token.item())
                
                # Store attention weights
                attention_weights.append(attention_weight.squeeze().cpu().numpy())
                
                # Update input for next step
                decoder_input = top_token.unsqueeze(0)
                
                # Check for EOS token
                if hasattr(target_tokenizer, 'token_to_id'):
                    eos_token_id = target_tokenizer.token_to_id[target_tokenizer.eos_token]
                else:
                    eos_token_id = target_tokenizer.encode(target_tokenizer.eos_token)[0]
                
                # Stop if EOS
                if top_token.item() == eos_token_id:
                    break
            
            # Convert to numpy array
            return np.array(attention_weights)
            
        except Exception as e:
            print(f"Error generating attention map: {e}")
            return None


def upload_results_to_wandb(vanilla_config, attention_config, vanilla_df, attention_df, 
                           architecture_table, comparison_table, examples, attn_heatmap_path=None):
    """Upload all results to Weights & Biases for better visualization and tracking"""
    
    # Initialize wandb
    run = wandb.init(
        project="da6401_assignment_3_seq2seq_attention",
        entity="da24s009-indiam-institute-of-technology-madras",
        name=f"compare_{vanilla_config.rnn_type}_vs_{attention_config.rnn_type}_attention",
        config={
            "vanilla_model": {
                "rnn_type": vanilla_config.rnn_type,
                "hidden_size": vanilla_config.hidden_size,
                "embedding_dim": vanilla_config.embedding_dim,
                "encoder_layers": vanilla_config.encoder_layers,
                "dropout": vanilla_config.dropout,
                "batch_size": vanilla_config.batch_size,
                "learning_rate": vanilla_config.learning_rate
            },
            "attention_model": {
                "rnn_type": attention_config.rnn_type,
                "hidden_size": attention_config.hidden_size, 
                "embedding_dim": attention_config.embedding_dim,
                "encoder_layers": attention_config.encoder_layers,
                "dropout": attention_config.dropout,
                "batch_size": attention_config.batch_size,
                "learning_rate": attention_config.learning_rate
            },
            "language": vanilla_config.language
        }
    )
    
    # 1. Log model architecture comparison as a table
    arch_table = wandb.Table(columns=["Parameter", "Vanilla Model", "Attention Model"])
    for row in architecture_table[1:]:  # Skip the header row
        arch_table.add_data(*row)
    
    wandb.log({"model_architecture_comparison": arch_table})
    
    # 2. Log performance comparison as a table
    perf_table = wandb.Table(columns=["Metric", "Vanilla Model", "Attention Model", "Improvement"])
    for row in comparison_table[1:]:  # Skip the header row
        perf_table.add_data(*row)
    
    wandb.log({"model_performance_comparison": perf_table})
    
    # 3. Create aggregate metrics
    vanilla_metrics = {
        "vanilla_char_accuracy": vanilla_df['char_acc'].mean(),
        "vanilla_word_accuracy": vanilla_df['word_acc'].mean(),
        "vanilla_perfect_predictions": sum(vanilla_df['word_acc'] == 1.0) / len(vanilla_df)
    }
    
    attention_metrics = {
        "attention_char_accuracy": attention_df['char_acc'].mean(),
        "attention_word_accuracy": attention_df['word_acc'].mean(),
        "attention_perfect_predictions": sum(attention_df['word_acc'] == 1.0) / len(attention_df)
    }
    
    improvement_metrics = {
        "char_accuracy_improvement": attention_metrics["attention_char_accuracy"] - vanilla_metrics["vanilla_char_accuracy"],
        "word_accuracy_improvement": attention_metrics["attention_word_accuracy"] - vanilla_metrics["vanilla_word_accuracy"],
        "perfect_predictions_improvement": attention_metrics["attention_perfect_predictions"] - vanilla_metrics["vanilla_perfect_predictions"]
    }
    
    # Log all metrics
    wandb.log({**vanilla_metrics, **attention_metrics, **improvement_metrics})
    
    # 4. Create a table for error corrections summary
    corrections_table = wandb.Table(columns=[
        "Example #", "Source", "Target", 
        "Vanilla Prediction", "Vanilla Word Acc",
        "Attention Prediction", "Attention Word Acc", 
        "Improvement"
    ])
    
    for i, ex in enumerate(examples):
        corrections_table.add_data(
            i+1,
            ex['source'],
            ex['target'],
            ex['predicted_vanilla'],
            ex['word_acc_vanilla'],
            ex['predicted_attention'],
            ex['word_acc_attention'],
            ex['word_acc_attention'] - ex['word_acc_vanilla']
        )
    
    wandb.log({"all_error_corrections": corrections_table})
    
    # 5. Upload each example as an individual table for detailed comparison
    for i, ex in enumerate(examples):
        # Create a detailed table for this example
        example_table = wandb.Table(columns=["Field", "Value"])
        
        # Add source and target
        example_table.add_data("Example ID", i+1)
        example_table.add_data("Source Text", ex['source'])
        example_table.add_data("Target Text", ex['target'])
        
        # Add vanilla model results
        example_table.add_data("Vanilla Model Prediction", ex['predicted_vanilla'])
        example_table.add_data("Vanilla Character Accuracy", f"{ex['char_acc_vanilla']:.4f}")
        example_table.add_data("Vanilla Word Accuracy", f"{ex['word_acc_vanilla']:.4f}")
        
        # Add attention model results
        example_table.add_data("Attention Model Prediction", ex['predicted_attention'])
        example_table.add_data("Attention Character Accuracy", f"{ex['char_acc_attention']:.4f}")
        example_table.add_data("Attention Word Accuracy", f"{ex['word_acc_attention']:.4f}")
        
        # Add improvement metrics
        example_table.add_data("Character Accuracy Improvement", 
                              f"{ex['char_acc_attention'] - ex['char_acc_vanilla']:+.4f}")
        example_table.add_data("Word Accuracy Improvement", 
                              f"{ex['word_acc_attention'] - ex['word_acc_vanilla']:+.4f}")
        
        # Log this example table
        wandb.log({f"example_{i+1}_details": example_table})
        
        # Alternative view: Create a comparison table for this example
        comparison_format = wandb.Table(columns=["Metric", "Vanilla Model", "Attention Model", "Improvement"])
        
        # Add rows for this example comparison
        comparison_format.add_data(
            "Prediction", 
            ex['predicted_vanilla'], 
            ex['predicted_attention'],
            "N/A"
        )
        comparison_format.add_data(
            "Character Accuracy",
            f"{ex['char_acc_vanilla']:.4f}",
            f"{ex['char_acc_attention']:.4f}",
            f"{ex['char_acc_attention'] - ex['char_acc_vanilla']:+.4f}"
        )
        comparison_format.add_data(
            "Word Accuracy",
            f"{ex['word_acc_vanilla']:.4f}",
            f"{ex['word_acc_attention']:.4f}",
            f"{ex['word_acc_attention'] - ex['word_acc_vanilla']:+.4f}"
        )
        
        # Log the comparison format table
        wandb.log({f"example_{i+1}_comparison": comparison_format})
    
    # 6. Log the attention heatmap if available
    if attn_heatmap_path and os.path.exists(attn_heatmap_path):
        wandb.log({"attention_heatmap_grid": wandb.Image(attn_heatmap_path)})
    
    # 7. Create character-level accuracy distribution
    char_acc_data = [[v, k] for k, v in zip(['Vanilla']*len(vanilla_df) + ['Attention']*len(attention_df),
                                         list(vanilla_df['char_acc']) + list(attention_df['char_acc']))]
    char_table = wandb.Table(data=char_acc_data, columns=["Model", "Character Accuracy"])
    
    wandb.log({"character_accuracy_distribution": wandb.plot.histogram(
        char_table, "Character Accuracy", title="Character Accuracy Distribution")
    })
    
    # 8. Create word-level accuracy distribution
    word_acc_data = [[v, k] for k, v in zip(['Vanilla']*len(vanilla_df) + ['Attention']*len(attention_df),
                                         list(vanilla_df['word_acc']) + list(attention_df['word_acc']))]
    word_table = wandb.Table(data=word_acc_data, columns=["Model", "Word Accuracy"])
    
    wandb.log({"word_accuracy_distribution": wandb.plot.histogram(
        word_table, "Word Accuracy", title="Word Accuracy Distribution")
    })
    
    # 9. Generate confusion matrix of perfect vs imperfect predictions
    # Create joint dataframe for easier comparison
    merged_df = pd.merge(vanilla_df, attention_df, 
                        on=['source', 'target'], 
                        suffixes=('_vanilla', '_attention'))
    
    # Create a 2x2 "confusion matrix" of perfect predictions
    vanilla_perfect = (merged_df['word_acc_vanilla'] == 1.0).astype(int)
    attention_perfect = (merged_df['word_acc_attention'] == 1.0).astype(int)
    
    # Count occurrences of each combination
    both_perfect = sum((vanilla_perfect == 1) & (attention_perfect == 1))
    only_vanilla_perfect = sum((vanilla_perfect == 1) & (attention_perfect == 0))
    only_attention_perfect = sum((vanilla_perfect == 0) & (attention_perfect == 1))
    neither_perfect = sum((vanilla_perfect == 0) & (attention_perfect == 0))
    
    # Create table for the confusion matrix
    matrix_table = wandb.Table(columns=["", "Attention Perfect", "Attention Imperfect"])
    matrix_table.add_data("Vanilla Perfect", both_perfect, only_vanilla_perfect)
    matrix_table.add_data("Vanilla Imperfect", only_attention_perfect, neither_perfect)
    
    wandb.log({"perfect_prediction_comparison": matrix_table})
    
    return run


def main():
    parser = argparse.ArgumentParser(description="Compare seq2seq models with and without attention")
    # Data paths
    parser.add_argument("--data_path", type=str, default="",
                        help="Path to the data directory")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory containing models and predictions")
    parser.add_argument("--language", type=str, default="hi",
                        help="Language code for the dataset")
    parser.add_argument("--wandb", action="store_true",
                        help="Upload results to Weights & Biases")
    
    args = parser.parse_args()
    
    # Create config objects for both vanilla and attention models based on provided hyperparameters
    vanilla_config = Config()
    vanilla_config.rnn_type = "gru"
    vanilla_config.hidden_size = 256
    vanilla_config.embedding_dim = 256
    vanilla_config.encoder_layers = 3
    vanilla_config.decoder_layers = 3
    vanilla_config.dropout = 0.3
    vanilla_config.data_path = args.data_path
    vanilla_config.language = args.language
    vanilla_config.device = "cuda" if torch.cuda.is_available() else "cpu"
    vanilla_config.sos_token = "<sos>"
    vanilla_config.eos_token = "<eos>"
    vanilla_config.output_dir = args.output_dir
    vanilla_config.batch_size = 64
    vanilla_config.epochs = 21
    vanilla_config.learning_rate = 0.0003
    vanilla_config.attention = False
    
    attention_config = Config()
    attention_config.rnn_type = "rnn"
    attention_config.hidden_size = 256
    attention_config.embedding_dim = 256
    attention_config.encoder_layers = 3
    attention_config.decoder_layers = 3
    attention_config.dropout = 0.354
    attention_config.data_path = args.data_path
    attention_config.language = args.language
    attention_config.device = "cuda" if torch.cuda.is_available() else "cpu"
    attention_config.sos_token = "<sos>"
    attention_config.eos_token = "<eos>"
    attention_config.output_dir = args.output_dir
    attention_config.batch_size = 128
    attention_config.epochs = 18
    attention_config.learning_rate = 0.00059
    attention_config.attention = True
    
    print("Vanilla Model Configuration:")
    for key, value in vars(vanilla_config).items():
        print(f"  {key}: {value}")
        
    print("\nAttention Model Configuration:")
    for key, value in vars(attention_config).items():
        print(f"  {key}: {value}")
    
    # Load predictions
    try:
        vanilla_df = load_model_predictions(os.path.join(args.output_dir, "predictions_vanilla"), "vanilla")
        attention_df = load_model_predictions(os.path.join(args.output_dir, "predictions_attention"), "attention")
    except Exception as e:
        print(f"Error loading predictions: {e}")
        sys.exit(1)
    
    # Compare models
    vanilla_char_acc, vanilla_word_acc, attention_char_acc, attention_word_acc = compare_models(
        vanilla_df, attention_df, vanilla_config, attention_config)
    
    # Store the tables for WandB
    architecture_table = [
        ["Parameter", "Vanilla Model", "Attention Model"],
        ["RNN Type", f"{vanilla_config.rnn_type.upper()}", f"{attention_config.rnn_type.upper()}"],
        ["Hidden Size", f"{vanilla_config.hidden_size}", f"{attention_config.hidden_size}"],
        ["Embedding Dim", f"{vanilla_config.embedding_dim}", f"{attention_config.embedding_dim}"],
        ["Encoder Layers", f"{vanilla_config.encoder_layers}", f"{attention_config.encoder_layers}"],
        ["Dropout", f"{vanilla_config.dropout}", f"{attention_config.dropout}"],
        ["Batch Size", f"{vanilla_config.batch_size}", f"{attention_config.batch_size}"],
        ["Learning Rate", f"{vanilla_config.learning_rate}", f"{attention_config.learning_rate}"],
        ["Attention", "No", "Yes"]
    ]
    
    comparison_table = [
        ["Metric", "Vanilla Model", "Attention Model", "Improvement"],
        ["Character Accuracy", f"{vanilla_char_acc:.4f}", f"{attention_char_acc:.4f}", 
         f"{attention_char_acc - vanilla_char_acc:+.4f}"],
        ["Word Accuracy", f"{vanilla_word_acc:.4f}", f"{attention_word_acc:.4f}", 
         f"{attention_word_acc - vanilla_word_acc:+.4f}"],
        ["Perfect Predictions", 
         f"{sum(vanilla_df['word_acc'] == 1.0)} ({sum(vanilla_df['word_acc'] == 1.0)/len(vanilla_df):.2%})", 
         f"{sum(attention_df['word_acc'] == 1.0)} ({sum(attention_df['word_acc'] == 1.0)/len(attention_df):.2%})", 
         f"{sum(attention_df['word_acc'] == 1.0) - sum(vanilla_df['word_acc'] == 1.0):+d} "
         f"({(sum(attention_df['word_acc'] == 1.0) - sum(vanilla_df['word_acc'] == 1.0))/len(vanilla_df):+.2%})"]
    ]
    
    # Find error corrections
    _, _, examples = find_error_corrections(vanilla_df, attention_df)
    
    # Generate attention heatmaps using the attention-specific configuration
    load_model_and_generate_attention_maps(attention_config, examples, args.output_dir)
    
    # Get the path to the generated attention heatmap
    attn_heatmap_path = os.path.join(args.output_dir, "attention_visualizations", "attention_heatmaps_grid.png") 
    
    # Upload to WandB if enabled
    if args.wandb:
        print("\nUploading results to Weights & Biases...")
        run = upload_results_to_wandb(
            vanilla_config, attention_config, 
            vanilla_df, attention_df,
            architecture_table, comparison_table, 
            examples, attn_heatmap_path
        )
        print(f"Results uploaded to W&B: {run.name}")


if __name__ == "__main__":
    main()