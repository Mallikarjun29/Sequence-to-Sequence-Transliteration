class Config:
    """Configuration for the transliteration model."""
    
    def __init__(self):
        # Data settings
        self.data_path = "./data"
        self.language = "hi"  # Hindi
        self.sos_token = "\t"
        self.eos_token = "\n"
        
        # Model settings
        self.embedding_dim = 256
        self.hidden_size = 256
        self.encoder_layers = 3
        self.decoder_layers = 3
        self.dropout = 0.2
        self.rnn_type = "lstm"  # Options: "rnn", "gru", "lstm"
        self.attention = False
        
        # Training settings
        self.batch_size = 128
        self.epochs = 30
        self.learning_rate = 0.001
        self.teacher_forcing_ratio = 1.0
        
        # Beam search settings
        self.beam_width = 3
        
        # Device settings
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Logging
        self.use_wandb = False
        self.wandb_project = "transliteration"