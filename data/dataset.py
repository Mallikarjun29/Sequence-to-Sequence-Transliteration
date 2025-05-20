import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from .tokenizer import CharTokenizer

class TransliterationDataset(Dataset):
    """Dataset for transliteration task.
    
    This dataset handles source and target texts for transliteration,
    using CharTokenizer to convert texts to token IDs.
    """
    
    def __init__(self, source_texts, target_texts, source_tokenizer=None, target_tokenizer=None):
        """Initialize the TransliterationDataset.
        
        Args:
            source_texts (list): List of source text strings.
            target_texts (list): List of target text strings.
            source_tokenizer (CharTokenizer, optional): Tokenizer for source texts.
            target_tokenizer (CharTokenizer, optional): Tokenizer for target texts.
        """
        self.source_texts = source_texts
        self.target_texts = target_texts
        
        if source_tokenizer is None:
            self.source_tokenizer = CharTokenizer()
            self.source_tokenizer.fit(source_texts)
        else:
            self.source_tokenizer = source_tokenizer
            
        if target_tokenizer is None:
            self.target_tokenizer = CharTokenizer()
            self.target_tokenizer.fit(target_texts)
        else:
            self.target_tokenizer = target_tokenizer
            
        self.source_ids = [self.source_tokenizer.encode(text) for text in source_texts]
        self.target_ids = [self.target_tokenizer.encode(text) for text in target_texts]
    
    def __len__(self):
        """Return the number of samples in the dataset.
        
        Returns:
            int: Number of samples.
        """
        return len(self.source_texts)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            tuple: (source_tensor, target_tensor) pair.
        """
        return torch.tensor(self.source_ids[idx]), torch.tensor(self.target_ids[idx])
    

def collate_fn(batch):
    """Collate function to handle variable-length sequences.
    
    Args:
        batch (list): List of (source, target) tensor pairs.
        
    Returns:
        tuple: (padded_source_batch, padded_target_batch) with consistent shapes.
    """
    source_batch, target_batch = zip(*batch)
    
    source_batch_padded = pad_sequence(source_batch, batch_first=True, padding_value=0)
    target_batch_padded = pad_sequence(target_batch, batch_first=True, padding_value=0)
    
    return source_batch_padded, target_batch_padded


def load_data(data_path, language, sos_token="\t", eos_token="\n"):
    """Load and preprocess data for the transliteration task.
    
    Args:
        data_path (str): Path to the data directory.
        language (str): Language code (e.g., 'hi' for Hindi).
        sos_token (str, optional): Start of sequence token.
        eos_token (str, optional): End of sequence token.
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset) containing the processed data.
    """
    
    template = os.path.join(data_path, "dakshina_dataset_v1.0/{}/lexicons/{}.translit.sampled.{}.tsv")
    
    train_path = template.format(language, language, "train")
    val_path = template.format(language, language, "dev")
    test_path = template.format(language, language, "test")
    
    train_df = pd.read_csv(train_path, sep="\t", header=None, encoding="utf-8")
    val_df = pd.read_csv(val_path, sep="\t", header=None, encoding="utf-8")
    test_df = pd.read_csv(test_path, sep="\t", header=None, encoding="utf-8")
    
    train_source = [text for text in train_df[1].astype(str)]
    train_target = [text for text in train_df[0].astype(str)]
    
    train_dataset = TransliterationDataset(train_source, train_target)
    
    val_source = [text for text in val_df[1].astype(str)]
    val_target = [text for text in val_df[0].astype(str)]
    val_dataset = TransliterationDataset(val_source, val_target, 
                                        train_dataset.source_tokenizer,
                                        train_dataset.target_tokenizer)
    
    test_source = [text for text in test_df[1].astype(str)]
    test_target = [text for text in test_df[0].astype(str)]
    test_dataset = TransliterationDataset(test_source, test_target,
                                         train_dataset.source_tokenizer,
                                         train_dataset.target_tokenizer)
    
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    data_directory = ''
    language_code = 'hi'

    print(f"Attempting to load data for language: {language_code} from {data_directory}")

    try:
        train_dataset, val_dataset, test_dataset = load_data(data_directory, language_code)
        print("Data loaded successfully!")

        batch_size = 32
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of validation samples: {len(val_dataset)}")
        print(f"Number of test samples: {len(test_dataset)}")

        try:
            sample_source_batch, sample_target_batch = next(iter(train_dataloader))

            print("\nSample batch from training dataloader:")
            print(f"Source batch shape: {sample_source_batch.shape}")
            print(f"Target batch shape: {sample_target_batch.shape}")

        except StopIteration:
            print("Could not get a batch from the training dataloader (dataset might be empty).")
        except AttributeError:
             print("Could not decode sample batch. Ensure CharTokenizer has a 'decode' method and 'pad_token_id'.")
        except Exception as e:
            print(f"An error occurred while processing the sample batch: {e}")

    except FileNotFoundError as e:
        print(f"Error loading data file: {e}")
        print(f"Please ensure the path '{data_directory}' is correct and contains the Dakshina dataset.")
    except ImportError:
         print("ImportError: Could not import CharTokenizer from .tokenizer.")
         print("Please ensure you have a tokenizer.py file with a CharTokenizer class in the appropriate location,")
         print("or run this script from within a package context where .tokenizer is resolvable.")
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")