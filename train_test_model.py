"""
Train a seq2seq model and evaluate it on the test dataset.
This script:
1. Trains a seq2seq model with specified hyperparameters
2. Evaluates on the test dataset with detailed metrics
3. Visualizes sample predictions using beam search
4. Saves all test set predictions to a file
"""

import os
import sys
import torch
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from tabulate import tabulate
from tqdm import tqdm
import wandb
from wandb import Table

from model.seq2seq import Seq2SeqModel
from data.dataset import load_data, collate_fn
from training.trainer import Trainer
from training.beam_search import BeamSearch
from utils.metrics import character_accuracy, word_accuracy, bleu_score
from config import Config
from torch.utils.data import DataLoader


def main():
    """Main entry point - parses arguments, loads data, trains and evaluates the model."""
    parser = argparse.ArgumentParser(description="Train and test a seq2seq model")
    parser.add_argument("--data_path", type=str, default="",
                        help="Path to the data directory")
    parser.add_argument("--language", type=str, default="hi",
                        help="Language code for the dataset")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory to save models, predictions, and visualizations")
    
    parser.add_argument("--rnn_type", type=str, default="lstm", choices=["lstm", "gru", "rnn"],
                        help="Type of RNN cell to use")
    parser.add_argument("--hidden_size", type=int, default=256,
                        help="Hidden size for RNN layers")
    parser.add_argument("--embedding_dim", type=int, default=128,
                        help="Embedding dimension")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of RNN layers in encoder/decoder")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout probability")
    parser.add_argument("--attention", action="store_true", default=False,
                        help="Whether to use attention mechanism")
    
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training and evaluation")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Initial learning rate")
    parser.add_argument("--beam_width", type=int, default=3,
                        help="Beam width for beam search during inference")
    parser.add_argument("--teacher_forcing_ratio", type=float, default=1.0,
                        help="Teacher forcing ratio during training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training and testing")
    
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Whether to log results to W&B")
    parser.add_argument("--wandb_entity", type=str, 
                        default="da24s009-indiam-institute-of-technology-madras",
                        help="W&B entity name")
    parser.add_argument("--wandb_project", type=str, 
                        default="da6401_assignment_3_seq2seq_attention",
                        help="W&B project name")
    
    args = parser.parse_args()
    
    if args.wandb:
        run_name = f"{args.rnn_type}_h{args.hidden_size}_e{args.embedding_dim}_l{args.num_layers}_{'att' if args.attention else 'vanilla'}"
        
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name,
            config={
                "rnn_type": args.rnn_type,
                "hidden_size": args.hidden_size,
                "embedding_dim": args.embedding_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "attention": args.attention,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "beam_width": args.beam_width,
                "language": args.language,
                "teacher_forcing_ratio": args.teacher_forcing_ratio
            }
        )
        print(f"W&B initialized with run name: {run_name}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "predictions_vanilla"), exist_ok=True)
    
    config = Config()
    
    config.rnn_type = args.rnn_type
    config.hidden_size = args.hidden_size
    config.embedding_dim = args.embedding_dim
    config.encoder_layers = args.num_layers
    config.decoder_layers = args.num_layers
    config.dropout = args.dropout
    config.attention = args.attention
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.learning_rate = args.learning_rate
    config.beam_width = args.beam_width
    config.teacher_forcing_ratio = args.teacher_forcing_ratio
    config.device = args.device
    config.use_wandb = False
    
    config.sos_token = "<sos>"
    config.eos_token = "<eos>"
    
    print("Configuration:")
    for key, value in vars(config).items():
        print(f"  {key}: {value}")
    
    try:
        train_dataset, val_dataset, test_dataset = load_data(
            args.data_path, args.language,
            sos_token=config.sos_token,
            eos_token=config.eos_token
        )
        print(f"Loaded datasets: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test examples")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    model = Seq2SeqModel(
        source_vocab_size=len(train_dataset.source_tokenizer),
        target_vocab_size=len(train_dataset.target_tokenizer),
        embedding_dim=config.embedding_dim,
        hidden_size=config.hidden_size,
        num_layers=config.encoder_layers,
        dropout=config.dropout,
        rnn_type=config.rnn_type,
        attention=config.attention
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    trainer = Trainer(model, config, train_dataset, val_dataset)
    
    print("\nTraining model...")
    trainer.fit(train_dataset, val_dataset)
    
    suffix = "attention" if config.attention else "vanilla"
    
    model_dir = os.path.join(args.output_dir, "models")
    predictions_dir = os.path.join(args.output_dir, f"predictions_{suffix}")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, f"best_model_{suffix}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    print("\nEvaluating model on test dataset using token accuracy...")
    test_loss, test_token_accuracy = trainer.evaluate(test_dataset)
    print(f"Test Loss: {test_loss:.4f}, Test Token Accuracy: {test_token_accuracy:.4f}")
    
    print("\nPerforming detailed evaluation with beam search...")
    results = evaluate_with_beam_search(model, test_dataset, config)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(predictions_dir, f"test_predictions_{suffix}_{timestamp}.csv")
    
    df = pd.DataFrame([{
        'source': item['source'],
        'target': item['target'],
        'predicted': item['predicted'],
        'char_acc': item['char_acc'],
        'word_acc': item['word_acc'],
    } for item in results])
    
    df.to_csv(csv_path, index=False)
    print(f"Predictions saved to {csv_path}")
    
    char_accuracies = [item['char_acc'] for item in results]
    word_accuracies = [item['word_acc'] for item in results]
    
    avg_char_accuracy = np.mean(char_accuracies)
    avg_word_accuracy = np.mean(word_accuracies)
    
    print(f"\nOverall Character Accuracy: {avg_char_accuracy:.4f}")
    print(f"Overall Word Accuracy: {avg_word_accuracy:.4f}")
    
    metrics_path = os.path.join(predictions_dir, f"metrics_summary_{suffix}_{timestamp}.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"TEST SET EVALUATION RESULTS\n")
        f.write(f"Character Accuracy: {avg_char_accuracy:.4f}\n")
        f.write(f"Word Accuracy: {avg_word_accuracy:.4f}\n")
        f.write(f"Token Accuracy: {test_token_accuracy:.4f}\n")
        f.write(f"Total examples: {len(results)}\n")
        f.write(f"\nModel: {config.rnn_type.upper()}, Layers: {config.encoder_layers}, ")
        f.write(f"Hidden: {config.hidden_size}, Embedding: {config.embedding_dim}, ")
        f.write(f"Attention: {config.attention}\n")
    
    sorted_results, suffix = visualize_results(results, config, args.output_dir)
    
    if args.wandb:
        upload_wandb_results(sorted_results, config, suffix, test_loss, test_token_accuracy)
    
    print(f"\nTest completed. Word Accuracy: {avg_word_accuracy:.4f}")
    
    return avg_word_accuracy


def evaluate_with_beam_search(model, test_dataset, config):
    """Evaluate model on test dataset using beam search for prediction.
    
    Args:
        model: The sequence-to-sequence model to evaluate
        test_dataset: Dataset containing test examples
        config: Configuration object with model and evaluation parameters
        
    Returns:
        list: List of dictionaries containing evaluation results for each example
    """
    device = torch.device(config.device)
    model.to(device)
    model.eval()
    
    beam_searcher = BeamSearch(model, beam_width=config.beam_width)
    
    source_tokenizer = test_dataset.source_tokenizer
    target_tokenizer = test_dataset.target_tokenizer
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    results = []
    
    with torch.no_grad():
        for batch_idx, (source_tensors, target_tensors) in enumerate(tqdm(test_dataloader, desc="Evaluating")):
            for i in range(source_tensors.size(0)):
                source_tensor = source_tensors[i].unsqueeze(0).to(device)
                target_tensor = target_tensors[i].to(device)
                
                source_text = source_tokenizer.decode([token.item() for token in source_tensor.squeeze(0) if token.item() > 0])
                target_text = target_tokenizer.decode([token.item() for token in target_tensor if token.item() > 0])
                
                if not source_text or not target_text:
                    continue
                
                beam_results = beam_searcher.translate(
                    source_text,
                    source_tokenizer,
                    target_tokenizer,
                    device=device
                )
                
                if beam_results:
                    predicted_text, score = beam_results[0]
                    candidates = [text for text, _ in beam_results]
                else:
                    predicted_text = ""
                    score = 0.0
                    candidates = []
                
                char_acc = character_accuracy(predicted_text, target_text)
                word_acc = word_accuracy(predicted_text, target_text)
                
                results.append({
                    'source': source_text,
                    'target': target_text,
                    'predicted': predicted_text,
                    'char_acc': char_acc,
                    'word_acc': word_acc,
                    'score': score,
                    'candidates': candidates
                })
    
    return results


def visualize_results(results, config, output_dir):
    """Create visualizations of model results.
    
    Args:
        results: List of dictionaries containing evaluation results
        config: Configuration object with model parameters
        output_dir: Directory to save visualizations
        
    Returns:
        tuple: (sorted_results, suffix)
            sorted_results: Results sorted by word accuracy
            suffix: String suffix used for filenames (e.g., "attention" or "vanilla")
    """
    import matplotlib.font_manager as fm
    
    try:
        font_paths = [
            "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
            "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
        ]
        
        font_found = False
        for font_path in font_paths:
            if os.path.exists(font_path):
                fm.fontManager.addfont(font_path)
                plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
                print(f"Using font: {font_path}")
                font_found = True
                break
                
        if not font_found:
            print("Warning: No suitable Devanagari font found. Text may not display correctly.")
    except Exception as e:
        print(f"Error setting up fonts: {e}")
    
    suffix = "attention" if config.attention else "vanilla"
    print(f"Using output suffix: {suffix}")
    
    viz_dir = os.path.join(output_dir, f"visualizations_{suffix}")
    os.makedirs(viz_dir, exist_ok=True)
    
    sorted_results = sorted(results, key=lambda x: x['word_acc'], reverse=True)
    
    best_examples = sorted_results[:5]
    worst_examples = sorted_results[-5:]
    
    print("\nBEST TRANSLATIONS:")
    print("-" * 80)
    for i, ex in enumerate(best_examples):
        print(f"{i+1}. Source: '{ex['source']}'")
        print(f"   Target: '{ex['target']}'")
        print(f"   Predicted: '{ex['predicted']}' (Word Acc: {ex['word_acc']:.2f})")
    
    print("\nWORST TRANSLATIONS:")
    print("-" * 80)
    for i, ex in enumerate(worst_examples):
        print(f"{i+1}. Source: '{ex['source']}'")
        print(f"   Target: '{ex['target']}'")
        print(f"   Predicted: '{ex['predicted']}' (Word Acc: {ex['word_acc']:.2f})")
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    word_accs = [r['word_acc'] for r in results]
    char_accs = [r['char_acc'] for r in results]
    
    plt.hist(word_accs, bins=20, alpha=0.7, color='blue')
    plt.axvline(np.mean(word_accs), color='red', linestyle='dashed', linewidth=2, 
                label=f'Mean: {np.mean(word_accs):.4f}')
    plt.axvline(np.median(word_accs), color='green', linestyle='dotted', linewidth=2,
                label=f'Median: {np.median(word_accs):.4f}')
    plt.title(f'Distribution of Word Accuracy ({suffix})', fontsize=14)
    plt.xlabel('Word Accuracy', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'word_accuracy_distribution_{suffix}.png'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    plt.scatter(char_accs, word_accs, alpha=0.5, s=50)
    
    z = np.polyfit(char_accs, word_accs, 1)
    p = np.poly1d(z)
    plt.plot(sorted(char_accs), p(sorted(char_accs)), "r--", 
             label=f"Trend: y={z[0]:.4f}x{z[1]:+.4f}")
    
    plt.xlabel('Character Accuracy', fontsize=12)
    plt.ylabel('Word Accuracy', fontsize=12)
    plt.title(f'Character vs Word Accuracy ({suffix})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'char_vs_word_accuracy_{suffix}.png'), dpi=300)
    plt.close()
    
    length_groups = {}
    for r in results:
        src_len = len(r['source'])
        if src_len not in length_groups:
            length_groups[src_len] = []
        length_groups[src_len].append(r['word_acc'])
    
    lengths = sorted(length_groups.keys())
    mean_accs = [np.mean(length_groups[l]) for l in lengths]
    
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    plt.bar(lengths, mean_accs, alpha=0.7)
    plt.title(f'Word Accuracy by Source Word Length ({suffix})', fontsize=14)
    plt.xlabel('Source Word Length', fontsize=12)
    plt.ylabel('Mean Word Accuracy', fontsize=12)
    plt.xticks(lengths)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'accuracy_by_word_length_{suffix}.png'), dpi=300)
    plt.close()
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    def plot_examples_table(examples, ax, title):
        ax.axis('tight')
        ax.axis('off')
        ax.set_title(title, fontsize=14)
        
        table_data = []
        for i, ex in enumerate(examples):
            src_len = len(ex['source'])
            tgt_len = len(ex['target'])
            pred_len = len(ex['predicted'])
            table_data.append([
                f"Example {i+1}",
                f"Length: {src_len}",
                f"Length: {tgt_len}",
                f"Length: {pred_len}",
                f"{ex['word_acc']:.2f}"
            ])
        
        colLabels = ['#', 'Source', 'Target', 'Predicted', 'Word Acc']
        table = ax.table(cellText=table_data, colLabels=colLabels, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
    
    plot_examples_table(best_examples, axs[0], f"Top 5 Most Accurate Translations ({suffix})")
    plot_examples_table(worst_examples, axs[1], f"5 Least Accurate Translations ({suffix})")
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'best_worst_examples_{suffix}.png'), dpi=300)
    plt.close()
    
    example_good = best_examples[0] if best_examples else None
    example_med = [r for r in results if 0.4 <= r['word_acc'] < 0.6]
    example_med = example_med[0] if example_med else None
    example_poor = worst_examples[0] if worst_examples else None
    
    examples_to_visualize = []
    if example_good:
        examples_to_visualize.append((example_good, "High Accuracy Example"))
    if example_med:
        examples_to_visualize.append((example_med, "Medium Accuracy Example"))
    if example_poor:
        examples_to_visualize.append((example_poor, "Low Accuracy Example"))
    
    fig, axs = plt.subplots(len(examples_to_visualize), 1, figsize=(12, 4 * len(examples_to_visualize)))
    if len(examples_to_visualize) == 1:
        axs = [axs]
    
    for i, (example, title) in enumerate(examples_to_visualize):
        target = example['target']
        predicted = example['predicted']
        
        matches = []
        for j, char in enumerate(predicted):
            if j < len(target) and char == target[j]:
                matches.append(1)
            else:
                matches.append(0)
        
        if not matches:
            continue
            
        x_positions = np.arange(len(matches))
        colors = ['green' if m == 1 else 'red' for m in matches]
        axs[i].bar(x_positions, matches, color=colors, width=0.7)
        
        for j, match in enumerate(matches):
            axs[i].text(j, 0.5, "1" if match == 1 else "0", ha='center', 
                      va='center', color='white', fontweight='bold', fontsize=14)
        
        word_acc = example['word_acc']
        char_acc = example['char_acc']
        axs[i].set_title(f"{title} (Word Acc: {word_acc:.2f}, Char Acc: {char_acc:.2f})\n"
                        f"Source length: {len(example['source'])}, "
                        f"Target length: {len(target)}, "
                        f"Predicted length: {len(predicted)}", fontsize=12)
        axs[i].set_ylim(0, 1.5)
        axs[i].set_xlim(-0.5, len(matches) - 0.5)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_frame_on(False)
        
        axs[i].bar(0, 0, color='green', label='Match (1)')
        axs[i].bar(0, 0, color='red', label='Mismatch (0)')
        axs[i].legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'character_analysis_{suffix}.png'), dpi=300)
    plt.close()
    
    examples_with_beams = [r for r in results if len(r.get('candidates', [])) > 1]
    if examples_with_beams:
        example = next((r for r in examples_with_beams if 0.3 <= r['word_acc'] <= 0.7), examples_with_beams[0])
        candidates = example.get('candidates', [])[:5]
        
        plt.figure(figsize=(12, 6))
        sns.set_style("whitegrid")
        
        y_positions = np.arange(len(candidates))
        candidate_lengths = [len(cand) for cand in candidates]
        
        similarities = []
        for i, cand in enumerate(candidates):
            if i == 0:
                sim = example['word_acc']
            else:
                common = 0
                for j in range(min(len(cand), len(candidates[0]))):
                    if cand[j] == candidates[0][j]:
                        common += 1
                sim = common / max(len(cand), len(candidates[0])) if max(len(cand), len(candidates[0])) > 0 else 0
            similarities.append(sim)
        
        plt.barh(y_positions, candidate_lengths, alpha=0.4, color='blue', label='Length')
        plt.barh(y_positions, [sim * max(candidate_lengths) for sim in similarities], 
                alpha=0.6, color='green', label='Similarity')
        
        for i, (length, sim) in enumerate(zip(candidate_lengths, similarities)):
            plt.text(0.5, i, f"Length: {length}, Sim: {sim:.2f}", va='center', fontsize=10)
            if i == 0:
                plt.text(length - 2, i, "(Selected)", va='center', fontsize=10)
        
        plt.yticks(y_positions, [f"Candidate {i+1}" for i in range(len(candidates))])
        plt.title(f"Beam Search Candidates ({suffix})", fontsize=14)
        plt.xlabel('Properties', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'beam_search_candidates_{suffix}.png'), dpi=300)
        plt.close()
    
    avg_word_acc = np.mean(word_accs)
    avg_char_acc = np.mean(char_accs)
    perfect_count = sum(1 for acc in word_accs if acc == 1.0)
    perfect_percent = perfect_count / len(results) * 100 if results else 0
    
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['Word Accuracy', 'Character Accuracy', 'Perfect Predictions (%)']
    values = [avg_word_acc, avg_char_acc, perfect_percent / 100]
    
    bars = ax.bar(metrics, values, color=['blue', 'green', 'purple'], alpha=0.7)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2%}' if height < 1 else f'{height*100:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontweight='bold')
    
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7)
    
    ax.set_title(f'Model Performance Summary - {suffix.capitalize()}\n'
                f'{config.rnn_type.upper()}, {config.encoder_layers} layers, Hidden Size {config.hidden_size}',
                fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Accuracy / Percentage', fontsize=12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.grid(axis='y', alpha=0.3)
    
    model_config_text = (
        f"Embedding: {config.embedding_dim}, Attention: {config.attention}\n"
        f"Beam Width: {config.beam_width}, Total examples: {len(results)}"
    )
    plt.figtext(0.5, 0.01, model_config_text, ha='center', fontsize=10, 
               bbox={"facecolor":"lightgrey", "alpha":0.5, "pad":5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(os.path.join(viz_dir, f'performance_summary_{suffix}.png'), dpi=300)
    plt.close()
    
    print(f"\nAll visualizations saved to: {viz_dir}")
    
    return sorted_results, suffix


def upload_wandb_results(results, config, suffix, test_loss, test_token_accuracy):
    """Upload results and visualizations to Weights & Biases.
    
    Args:
        results: List of dictionaries containing evaluation results
        config: Configuration object with model parameters
        suffix: String suffix used in filenames ("attention" or "vanilla")
        test_loss: Test loss value
        test_token_accuracy: Token accuracy on test set
    """
    word_accs = [r['word_acc'] for r in results]
    char_accs = [r['char_acc'] for r in results]
    avg_word_acc = np.mean(word_accs)
    avg_char_acc = np.mean(char_accs)
    perfect_count = sum(1 for acc in word_accs if acc == 1.0)
    perfect_percent = perfect_count / len(results) * 100 if results else 0
    
    wandb.log({
        "test_loss": test_loss,
        "test_token_accuracy": test_token_accuracy,
        "test_char_accuracy": avg_char_acc,
        "test_word_accuracy": avg_word_acc,
        "perfect_predictions_percent": perfect_percent
    })
    
    sorted_results = sorted(results, key=lambda x: x['word_acc'], reverse=True)
    best_examples = sorted_results[:10]
    
    best_table = wandb.Table(columns=["Source", "Target", "Predicted", "Character Accuracy", "Word Accuracy"])
    for ex in best_examples:
        best_table.add_data(
            ex['source'],
            ex['target'],
            ex['predicted'],
            ex['char_acc'],
            ex['word_acc']
        )
    
    worst_examples = sorted_results[-10:]
    
    worst_table = wandb.Table(columns=["Source", "Target", "Predicted", "Character Accuracy", "Word Accuracy"])
    for ex in worst_examples:
        worst_table.add_data(
            ex['source'],
            ex['target'],
            ex['predicted'],
            ex['char_acc'],
            ex['word_acc']
        )
    
    viz_dir = os.path.join(config.output_dir if hasattr(config, 'output_dir') else "outputs", 
                          f"visualizations_{suffix}")
    
    if os.path.exists(viz_dir):
        for file in os.listdir(viz_dir):
            if file.endswith((".png", ".jpg", ".jpeg", ".svg", ".pdf")):
                file_path = os.path.join(viz_dir, file)
                wandb.log({f"visualization/{file}": wandb.Image(file_path)})
    
    wandb.log({
        f"best_examples_{suffix}": best_table,
        f"worst_examples_{suffix}": worst_table
    })
    
    print("Results and visualizations uploaded to W&B")


if __name__ == "__main__":
    main()