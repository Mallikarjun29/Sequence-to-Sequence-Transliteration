# DA6401 Assignment 3: Sequence-to-Sequence Transliteration

This repository contains an implementation of sequence-to-sequence models for transliteration, with both vanilla and attention-based architectures. The project is designed to transliterate text from Latin script to other writing systems, with a primary focus on Indic languages from the Dakshina dataset.

## Project Structure

```
├── comparison.py               # Compare vanilla and attention-based models
├── config.py                   # Configuration settings for models and training
├── html_attention_visualization.py  # Create interactive HTML visualizations of attention
├── README.md                   # This file
├── sweep_experiment.py         # Hyperparameter optimization with W&B sweeps
├── train_test_model.py         # Main script to train and evaluate models
├── data/
│   ├── dataset.py              # Dataset loading and handling
│   └── tokenizer.py            # Character tokenization for input/output
├── model/
│   ├── attention.py            # Attention mechanism implementation
│   ├── decoder.py              # Decoder component with/without attention
│   ├── encoder.py              # Encoder component
│   └── seq2seq.py              # Complete sequence-to-sequence model
├── training/
│   ├── beam_search.py          # Beam search for inference
│   └── trainer.py              # Training loop and evaluation
├── utils/
│   └── metrics.py              # Evaluation metrics (character accuracy, word accuracy, BLEU)
├── outputs/                    # Output directory for models and visualizations
│   ├── models/                 # Saved model weights
│   ├── predictions_vanilla/    # Predictions from vanilla models
│   ├── predictions_attention/  # Predictions from attention models
│   ├── visualizations_vanilla/ # Visualizations for vanilla model results
│   └── visualizations_attention/  # Visualizations for attention model results
└── dakshina_dataset_v1.0/      # Dataset directory (not included in repo)
```

## Usage

### Main Training and Evaluation Script

The main script for training and evaluating models is train_test_model.py. It handles the complete pipeline:

1. Training a seq2seq model (vanilla or with attention)
2. Evaluating on test data
3. Generating visualizations
4. Saving predictions and model weights

```bash
python train_test_model.py --data_path ./dakshina_dataset_v1.0 --language hi --attention
```

#### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_path` | str | "" | Path to the data directory containing dataset |
| `--language` | str | "hi" | Language code for the dataset (e.g., 'hi' for Hindi) |
| `--output_dir` | str | "./outputs" | Directory to save models, predictions, and visualizations |
| `--rnn_type` | str | "lstm" | Type of RNN cell to use (choices: "lstm", "gru", "rnn") |
| `--hidden_size` | int | 256 | Hidden size for RNN layers |
| `--embedding_dim` | int | 128 | Embedding dimension |
| `--num_layers` | int | 2 | Number of RNN layers in encoder/decoder |
| `--dropout` | float | 0.3 | Dropout probability |
| `--attention` | flag | False | Whether to use attention mechanism |
| `--batch_size` | int | 64 | Batch size for training and evaluation |
| `--epochs` | int | 10 | Number of training epochs |
| `--learning_rate` | float | 0.001 | Initial learning rate |
| `--beam_width` | int | 3 | Beam width for beam search during inference |
| `--teacher_forcing_ratio` | float | 1.0 | Teacher forcing ratio during training |
| `--device` | str | "cuda"/cpu" | Device to use for training and testing |
| `--wandb` | flag | False | Whether to log results to Weights & Biases |
| `--wandb_entity` | str | "da24s009..." | W&B entity name |
| `--wandb_project` | str | "da6401..." | W&B project name |

### Comparing Models

To compare vanilla and attention-based models that have already been trained:

```bash
python comparison.py --data_path ./dakshina_dataset_v1.0 --language hi --output_dir ./outputs
```

### Creating Attention Visualizations

To create interactive HTML visualizations of the attention mechanism:

```bash
python html_attention_visualization.py --output_dir ./outputs --data_path ./dakshina_dataset_v1.0 --language hi
```

### Hyperparameter Optimization

To run hyperparameter optimization using Weights & Biases sweeps:

```bash
python sweep_experiment.py --data_path ./dakshina_dataset_v1.0 --language hi --sweep_count 50
```

## Model Architectures

The project implements two main architectures:

1. **Vanilla Seq2Seq**: A basic encoder-decoder architecture with RNNs (LSTM/GRU/RNN)
2. **Attention-based Seq2Seq**: Enhances the vanilla model with an attention mechanism for better performance

Both architectures support:
- Different RNN cell types (LSTM, GRU, vanilla RNN)
- Multiple encoder/decoder layers
- Beam search for improved inference
- Configurable embedding dimensions, hidden sizes, and dropout

## Evaluation Metrics

Models are evaluated using:
- Character accuracy
- Word (exact match) accuracy
- Token accuracy (during training)

## Visualizations

The scripts generate various visualizations:
- Word and character accuracy distributions
- Best/worst predictions
- Character-by-character prediction analysis
- Beam search candidate comparisons
- Attention heatmaps (for attention models)

## Requirements

- Python 3.7+
- PyTorch
- NLTK
- pandas
- matplotlib
- seaborn
- wandb (optional, for experiment tracking)

## Dataset

The implementation is designed for the [Dakshina dataset](https://github.com/google-research-datasets/dakshina), which includes parallel Latin and native script data for several Indic languages.