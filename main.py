import torch
import argparse
import os
from data.dataset import load_data
from data.tokenizer import CharTokenizer
from model.seq2seq import Seq2SeqModel
from training.trainer import Trainer
from training.beam_search import BeamSearch
from utils.metrics import character_accuracy, word_accuracy, bleu_score
from utils.visualization import visualize_attention, visualize_model_outputs, visualize_connectivity
from config import Config
from torch.utils.data import DataLoader
import random
import numpy as np

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

def main(args):
    # Set random seed
    set_seed(42)
    
    # Load config
    config = Config()
    config.language = args.language
    config.embedding_dim = args.embedding_dim
    config.hidden_size = args.hidden_size
    config.encoder_layers = args.encoder_layers
    config.decoder_layers = args.decoder_layers
    config.dropout = args.dropout
    config.rnn_type = args.rnn_type
    config.attention = args.attention
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.teacher_forcing_ratio = args.teacher_forcing_ratio
    config.beam_width = args.beam_width
    config.use_wandb = args.use_wandb
    
    # Load data
    print("Loading data...")
    train_dataset, val_dataset, test_dataset = load_data(
        args.data_path, config.language, 
        sos_token=config.sos_token, eos_token=config.eos_token
    )
    
    # Create model
    print("Creating model...")
    model = Seq2SeqModel(
        source_vocab_size=len(train_dataset.source_tokenizer),
        target_vocab_size=len(train_dataset.target_tokenizer),
        embedding_dim=config.embedding_dim,
        hidden_size=config.hidden_size,
        num_layers=config.encoder_layers,  # Same for encoder and decoder
        dropout=config.dropout,
        rnn_type=config.rnn_type,
        attention=config.attention
    )
    
    # Train model
    if not args.eval_only:
        print("Training model...")
        trainer = Trainer(model, config)
        trainer.fit(train_dataset, val_dataset)
        
        # Save model
        if args.save_path:
            os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            torch.save(model.state_dict(), args.save_path)
            print(f"Model saved to {args.save_path}")
    
    # Load pretrained model if specified
    if args.load_path:
        print(f"Loading model from {args.load_path}")
        model.load_state_dict(torch.load(args.load_path, map_location=config.device))
    
    # Evaluate model
    if args.eval:
        print("Evaluating model...")
        trainer = Trainer(model, config)
        test_loss = trainer.evaluate(test_dataset)
        
        # Test with beam search
        if args.beam_search:
            print("Testing with beam search...")
            beam_searcher = BeamSearch(model, beam_width=config.beam_width)
            
            # Sample some test examples
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=True
            )
            source_batch, target_batch = next(iter(test_dataloader))
            
            # Process a few examples
            num_samples = min(10, len(source_batch))
            for i in range(num_samples):
                source_seq = source_batch[i].tolist()
                target_seq = target_batch[i].tolist()
                
                # Remove padding
                source_seq = [s for s in source_seq if s > 0]
                target_seq = [t for t in target_seq if t > 0]
                
                # Decode sequences
                source_text = train_dataset.source_tokenizer.decode(source_seq)
                target_text = train_dataset.target_tokenizer.decode(target_seq)
                
                # Translate with beam search
                results = beam_searcher.translate(
                    source_text, 
                    train_dataset.source_tokenizer,
                    train_dataset.target_tokenizer,
                    device=config.device
                )
                
                # Print results
                print(f"Source: {source_text}")
                print(f"Target: {target_text}")
                for j, (output, score) in enumerate(results[:3]):  # Show top 3 results
                    print(f"Beam {j+1}: {output} (score: {score:.4f})")
                print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transliteration with Seq2Seq model")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="./data", 
                        help="Path to data directory")
    parser.add_argument("--language", type=str, default="hi",
                        help="Language code (e.g., 'hi' for Hindi)")
    
    # Model arguments
    parser.add_argument("--embedding_dim", type=int, default=256,
                        help="Embedding dimension")
    parser.add_argument("--hidden_size", type=int, default=256,
                        help="Hidden size of RNN")
    parser.add_argument("--encoder_layers", type=int, default=3,
                        help="Number of encoder layers")
    parser.add_argument("--decoder_layers", type=int, default=3,
                        help="Number of decoder layers")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate")
    parser.add_argument("--rnn_type", type=str, default="lstm",
                        choices=["rnn", "lstm", "gru"],
                        help="Type of RNN cell")
    parser.add_argument("--attention", action="store_true",
                        help="Use attention mechanism")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs")
    parser.add_argument("--teacher_forcing_ratio", type=float, default=1.0,
                        help="Teacher forcing ratio")
    parser.add_argument("--beam_width", type=int, default=3,
                        help="Beam width for beam search")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    
    # Evaluation arguments
    parser.add_argument("--eval", action="store_true",
                        help="Evaluate model")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only evaluate model (no training)")
    parser.add_argument("--beam_search", action="store_true",
                        help="Use beam search for evaluation")
    
    # Save/load arguments
    parser.add_argument("--save_path", type=str, default="",
                        help="Path to save model")
    parser.add_argument("--load_path", type=str, default="",
                        help="Path to load model")
    
    args = parser.parse_args()
    
    main(args)
