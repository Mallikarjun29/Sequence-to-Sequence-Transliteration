import os
import sys
import torch
import numpy as np
import json
from pathlib import Path
import argparse
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from comparison import generate_attention_map
except ImportError:
    print("Error: Could not import from comparison.py. Make sure the file exists in the project.")
    sys.exit(1)

try:
    from model.seq2seq import Seq2SeqModel
    from data.dataset import load_data, collate_fn
except ImportError:
    print("Error: Could not import project modules. Check your project structure.")
    sys.exit(1)


def create_html_visualization(examples_results, output_path):
    """Create HTML visualization of attention weights.
    
    Args:
        examples_results (list): List of examples with attention weights.
        output_path (str): Path to save the HTML visualization file.
    """
    
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Attention Visualization</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
            }
            h1, h2 {
                text-align: center;
            }
            .example-container {
                margin-bottom: 30px;
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 15px;
            }
            .attention-row {
                display: flex;
                margin-bottom: 5px;
                align-items: center;
            }
            .output-char {
                font-size: 18px;
                width: 30px;
                text-align: center;
                font-weight: bold;
            }
            .attention-cells {
                display: flex;
                flex-grow: 1;
            }
            .attention-cell {
                width: 30px;
                height: 30px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin: 1px;
                font-size: 16px;
            }
            .comparison-table {
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }
            .comparison-table th {
                background-color: #f2f2f2;
                padding: 10px;
                text-align: center;
                border: 1px solid #ddd;
            }
            .comparison-table td {
                padding: 10px;
                text-align: center;
                border: 1px solid #ddd;
                vertical-align: middle;
            }
            .attention-display {
                display: flex;
                justify-content: center;
            }
        </style>
    </head>
    <body>
        <h1>Attention Visualization</h1>
        
        <h2>Individual Example Visualizations</h2>
        <div id="examples-container">
            <!-- Examples will be inserted here by JavaScript -->
        </div>
        
        <h2>Character Focus at Index 4 Comparison</h2>
        <table class="comparison-table">
            <thead>
                <tr>
                    <th>Character in Prediction Focussed</th>
                    <th>Attention Visualization</th>
                </tr>
            </thead>
            <tbody id="comparison-body">
                <!-- Comparison rows will be inserted here by JavaScript -->
            </tbody>
        </table>
        
        <script>
            // Data for all examples
            const examples = EXAMPLES_JSON;
            
            // Function to get a color based on attention weight
            function getColorFromAttention(weight, maxWeight) {
                // Normalize weight to be between 0 and 1
                const normalizedWeight = weight / maxWeight;
                
                if (normalizedWeight < 0.3) {
                    return '#f7fbff'; // Very light blue
                } else if (normalizedWeight < 0.5) {
                    return '#deebf7'; // Light blue
                } else if (normalizedWeight < 0.7) {
                    return '#c6dbef'; // Medium light blue
                } else if (normalizedWeight < 0.9) {
                    return '#9ecae1'; // Medium blue
                } else {
                    return '#4292c6'; // Darker blue
                }
            }
            
            // Function to get the darkest color for highest attention
            function getHighlightColor() {
                return '#08519c'; // Dark blue for max attention
            }
            
            // Render individual examples
            const examplesContainer = document.getElementById('examples-container');
            
            examples.forEach(example => {
                const div = document.createElement('div');
                div.className = 'example-container';
                
                const header = document.createElement('h3');
                header.textContent = `Source: "${example.source}" → Predicted: "${example.predicted}"`;
                div.appendChild(header);
                
                // For each output character
                example.attention_weights.forEach((weights, i) => {
                    if (i >= example.predicted.length) return;
                    
                    const row = document.createElement('div');
                    row.className = 'attention-row';
                    
                    const outputChar = document.createElement('div');
                    outputChar.className = 'output-char';
                    outputChar.textContent = example.predicted[i];
                    row.appendChild(outputChar);
                    
                    const attentionCells = document.createElement('div');
                    attentionCells.className = 'attention-cells';
                    
                    // Find max attention weight for this character
                    const maxWeight = Math.max(...weights);
                    
                    // For each input character
                    for (let j = 0; j < example.source.length; j++) {
                        const cell = document.createElement('div');
                        cell.className = 'attention-cell';
                        cell.textContent = example.source[j];
                        
                        // Set background color based on attention weight
                        if (weights[j] === maxWeight) {
                            cell.style.backgroundColor = getHighlightColor();
                            cell.style.color = 'white';
                        } else {
                            cell.style.backgroundColor = getColorFromAttention(weights[j], maxWeight);
                        }
                        
                        attentionCells.appendChild(cell);
                    }
                    
                    row.appendChild(attentionCells);
                    div.appendChild(row);
                });
                
                examplesContainer.appendChild(div);
            });
            
            // Render comparison table
            const comparisonBody = document.getElementById('comparison-body');
            
            examples.forEach(example => {
                const row = document.createElement('tr');
                
                const descCell = document.createElement('td');
                const targetIndex = Math.min(4, example.predicted.length - 1);
                const charToShow = targetIndex >= 0 ? example.predicted[targetIndex] : '?';
                descCell.innerHTML = `character at index ${targetIndex} of ${example.predicted}<br><strong>(${charToShow})</strong>`;
                row.appendChild(descCell);
                
                const visCell = document.createElement('td');
                const attentionDisplay = document.createElement('div');
                attentionDisplay.className = 'attention-display';
                
                if (targetIndex >= 0 && targetIndex < example.attention_weights.length) {
                    const weights = example.attention_weights[targetIndex];
                    
                    // Find max weight for this row
                    const maxWeight = Math.max(...weights);
                    
                    // For each input character
                    for (let j = 0; j < example.source.length; j++) {
                        const cell = document.createElement('div');
                        cell.className = 'attention-cell';
                        cell.textContent = example.source[j];
                        
                        // Set background color based on attention weight
                        if (weights[j] === maxWeight) {
                            cell.style.backgroundColor = getHighlightColor();
                            cell.style.color = 'white';
                        } else {
                            cell.style.backgroundColor = getColorFromAttention(weights[j], maxWeight);
                        }
                        
                        attentionDisplay.appendChild(cell);
                    }
                } else {
                    attentionDisplay.textContent = "Output too short";
                }
                
                visCell.appendChild(attentionDisplay);
                row.appendChild(visCell);
                
                comparisonBody.appendChild(row);
            });
        </script>
    </body>
    </html>
    """
    
    examples_json = []
    for example in examples_results:
        source_text = example['source']
        predicted_text = example['predicted_attention']
        attention_weights = example['attention_weights']
        
        attention_weights_list = [weights.tolist() for weights in attention_weights]
        
        example_data = {
            "source": source_text,
            "predicted": predicted_text,
            "attention_weights": attention_weights_list
        }
        examples_json.append(example_data)
    
    final_html = html_template.replace('EXAMPLES_JSON', json.dumps(examples_json))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_html)
    
    print(f"HTML visualization created at {output_path}")


def load_model_predictions(predictions_dir, model_type):
    """Load model predictions from CSV file.
    
    Args:
        predictions_dir (str): Directory containing prediction files.
        model_type (str): Type of model ("vanilla" or "attention").
        
    Returns:
        DataFrame: Loaded prediction data.
        
    Raises:
        FileNotFoundError: If no prediction files found for the model type.
    """
    files = [f for f in os.listdir(predictions_dir) if f.startswith(f"test_predictions_{model_type}")]
    if not files:
        raise FileNotFoundError(f"No prediction files found for {model_type} model.")
    
    files.sort(key=lambda x: x.split('_')[-1].split('.')[0], reverse=True)
    
    csv_path = os.path.join(predictions_dir, files[0])
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} predictions from {csv_path}")
    return df


def find_error_corrections(vanilla_df, attention_df):
    """Find examples where attention model corrected errors made by vanilla model.
    
    Args:
        vanilla_df (DataFrame): Predictions from vanilla model.
        attention_df (DataFrame): Predictions from attention model.
        
    Returns:
        tuple: (corrections, improvements, examples)
            corrections: DataFrame with complete error corrections
            improvements: DataFrame with significant improvements
            examples: List of example dictionaries for visualization
    """
    merged_df = pd.merge(vanilla_df, attention_df, 
                         on=['source', 'target'], 
                         suffixes=('_vanilla', '_attention'))
    
    corrections = merged_df[(merged_df['word_acc_vanilla'] < 1.0) & (merged_df['word_acc_attention'] == 1.0)]
    
    improvements = merged_df[(merged_df['word_acc_attention'] - merged_df['word_acc_vanilla'] >= 0.5)]
    
    improvements = improvements.sort_values(by='word_acc_vanilla', ascending=True)
    
    print(f"\nFound {len(corrections)} complete error corrections (vanilla wrong, attention correct)")
    print(f"Found {len(improvements)} significant improvements (word_acc diff ≥ 0.5)")
    
    examples_df = pd.concat([corrections.head(5), improvements.head(5)]).drop_duplicates()
    examples_df = examples_df.head(10)
    
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
    """Load attention model and generate attention maps for examples.
    
    Args:
        config: Model configuration.
        examples (list): List of example dictionaries.
        output_dir (str): Output directory for visualizations.
        
    Returns:
        list: Examples with attention weights added.
    """
    attn_dir = os.path.join(output_dir, "attention_visualizations")
    os.makedirs(attn_dir, exist_ok=True)
    
    print("\n" + "="*50)
    print(f"GENERATING ATTENTION VISUALIZATIONS USING {config.rnn_type.upper()} MODEL")
    print("="*50)
    
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
    
    model = Seq2SeqModel(
        source_vocab_size=len(source_tokenizer),
        target_vocab_size=len(target_tokenizer),
        embedding_dim=config.embedding_dim,
        hidden_size=config.hidden_size,
        num_layers=config.encoder_layers,
        dropout=config.dropout,
        rnn_type=config.rnn_type,
        attention=True
    )
    
    print(f"Model created with: {model.encoder.num_layers} encoder layers and {model.decoder.num_layers} decoder layers")
    print(f"Hidden size: {config.hidden_size}, RNN type: {config.rnn_type}")
    
    model_path = os.path.join(output_dir, "models", "best_model_attention.pt")
    if not os.path.exists(model_path):
        print(f"Error: Attention model checkpoint not found at {model_path}")
        return None
    
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"Loaded attention model from {model_path} with weights_only=True")
    except Exception as e1:
        try:
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded attention model from {model_path}")
        except Exception as e2:
            print(f"Error loading model: {e2}")
            return None
    
    device = torch.device(config.device)
    model.eval()
    model.to(device)
    
    for example in examples:
        source_text = example['source']
        
        source_tokens = source_tokenizer.encode(source_text)
        source_tensor = torch.tensor(source_tokens).unsqueeze(0).to(device)
        
        attention_weights = generate_attention_map(model, source_tensor, target_tokenizer, device)
        
        if attention_weights is None:
            print(f"Could not generate attention weights for '{source_text}'")
            example['attention_weights'] = []
            continue
        
        example['attention_weights'] = attention_weights
    
    return examples


def main():
    """Main function to create HTML attention visualizations."""
    parser = argparse.ArgumentParser(description="Create HTML attention visualization")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Directory containing models and predictions")
    parser.add_argument("--data_path", type=str, default="",
                        help="Path to the data directory")
    parser.add_argument("--language", type=str, default="hi",
                        help="Language code for the dataset")
    parser.add_argument("--cpu", action="store_true", 
                        help="Force CPU usage even if GPU is available")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.join(args.output_dir, "attention_visualizations"), exist_ok=True)
    
    class Config:
        def __init__(self):
            self.rnn_type = "rnn"
            self.hidden_size = 256
            self.embedding_dim = 256
            self.encoder_layers = 3
            self.decoder_layers = 3
            self.dropout = 0.354
            self.data_path = args.data_path
            self.language = args.language
            self.device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
            self.sos_token = "<sos>"
            self.eos_token = "<eos>"
            self.output_dir = args.output_dir
            self.batch_size = 128
            self.epochs = 18
            self.learning_rate = 0.00059
            self.attention = True
    
    config = Config()
    print(f"Using device: {config.device}")
    
    try:
        print("\nLoading model predictions...")
        vanilla_df = load_model_predictions(os.path.join(args.output_dir, "predictions_vanilla"), "vanilla")
        attention_df = load_model_predictions(os.path.join(args.output_dir, "predictions_attention"), "attention")
    except Exception as e:
        print(f"Error loading predictions: {e}")
        sys.exit(1)
    
    print("\nFinding examples with significant improvements...")
    corrections, improvements, examples = find_error_corrections(vanilla_df, attention_df)
    
    if not examples:
        print("No examples found with significant improvements.")
        sys.exit(1)
    
    print("\nGenerating attention maps...")
    examples_with_attention = load_model_and_generate_attention_maps(config, examples, args.output_dir)
    
    if not examples_with_attention:
        print("Could not generate attention maps for examples.")
        sys.exit(1)
    
    html_path = os.path.join(args.output_dir, "attention_visualizations", "attention_visualization.html")
    create_html_visualization(examples_with_attention, html_path)
    
    try:
        import webbrowser
        webbrowser.open(f"file://{os.path.abspath(html_path)}")
        print(f"Opened HTML visualization in browser: {html_path}")
    except:
        print(f"HTML visualization saved to: {html_path}")


if __name__ == "__main__":
    main()