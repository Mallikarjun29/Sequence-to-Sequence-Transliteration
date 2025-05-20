import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from .tokenizer import CharTokenizer

class TransliterationDataset(Dataset):
    """Dataset for transliteration task."""
    
    def __init__(self, source_texts, target_texts, source_tokenizer=None, target_tokenizer=None):
        self.source_texts = source_texts
        self.target_texts = target_texts
        
        # Create or use tokenizers
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
            
        # Tokenize all texts
        self.source_ids = [self.source_tokenizer.encode(text) for text in source_texts]
        self.target_ids = [self.target_tokenizer.encode(text) for text in target_texts]
    
    def __len__(self):
        return len(self.source_texts)
    
    def __getitem__(self, idx):
        return torch.tensor(self.source_ids[idx]), torch.tensor(self.target_ids[idx])
    

def collate_fn(batch):
    """Collate function to handle variable-length sequences."""
    source_batch, target_batch = zip(*batch)
    
    # Pad sequences
    source_batch_padded = pad_sequence(source_batch, batch_first=True, padding_value=0)
    target_batch_padded = pad_sequence(target_batch, batch_first=True, padding_value=0)
    
    return source_batch_padded, target_batch_padded


def load_data(data_path, language, sos_token="\t", eos_token="\n"):
    """Load and preprocess data for the transliteration task."""
    
    # Path template for data files
    template = os.path.join(data_path, "dakshina_dataset_v1.0/{}/lexicons/{}.translit.sampled.{}.tsv")
    
    # Get paths for train, val, and test files
    train_path = template.format(language, language, "train")
    val_path = template.format(language, language, "dev")
    test_path = template.format(language, language, "test")
    
    # Load dataframes
    train_df = pd.read_csv(train_path, sep="\t", header=None, encoding="utf-8")
    val_df = pd.read_csv(val_path, sep="\t", header=None, encoding="utf-8")
    test_df = pd.read_csv(test_path, sep="\t", header=None, encoding="utf-8")
    
    # Process train data
    train_source = [text for text in train_df[1].astype(str)]
    train_target = [text for text in train_df[0].astype(str)]
    
    # Create dataset
    train_dataset = TransliterationDataset(train_source, train_target)
    
    # Process validation data using the same tokenizers
    val_source = [text for text in val_df[1].astype(str)]
    val_target = [text for text in val_df[0].astype(str)]
    val_dataset = TransliterationDataset(val_source, val_target, 
                                        train_dataset.source_tokenizer,
                                        train_dataset.target_tokenizer)
    
    # Process test data
    test_source = [text for text in test_df[1].astype(str)]
    test_target = [text for text in test_df[0].astype(str)]
    test_dataset = TransliterationDataset(test_source, test_target,
                                         train_dataset.source_tokenizer,
                                         train_dataset.target_tokenizer)
    
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    data_directory = ''  # <--- **MODIFY THIS PATH**
    language_code = 'hi' # <--- **MODIFY THIS LANGUAGE CODE if needed**

    print(f"Attempting to load data for language: {language_code} from {data_directory}")

    try:
        # Load the datasets
        train_dataset, val_dataset, test_dataset = load_data(data_directory, language_code)
        print("Data loaded successfully!")

        # Create DataLoaders
        batch_size = 32
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of validation samples: {len(val_dataset)}")
        print(f"Number of test samples: {len(test_dataset)}")

        # Get a sample batch from the training dataloader
        try:
            sample_source_batch, sample_target_batch = next(iter(train_dataloader))

            print("\nSample batch from training dataloader:")
            print(f"Source batch shape: {sample_source_batch.shape}")
            print(f"Target batch shape: {sample_target_batch.shape}")

            # Optional: Decode and print a sample (requires CharTokenizer's decode method)
            # print("\nFirst sample in the batch (decoded):")
            # # Find the index of the first padding token or end of sequence
            # source_len = (sample_source_batch[0] != train_dataset.source_tokenizer.pad_token_id).sum()
            # target_len = (sample_target_batch[0] != train_dataset.target_tokenizer.pad_token_id).sum()
            # sample_source_decoded = train_dataset.source_tokenizer.decode(sample_source_batch[0][:source_len].tolist())
            # sample_target_decoded = train_dataset.target_tokenizer.decode(sample_target_batch[0][:target_len].tolist())
            #
            # print(f"Source: {sample_source_decoded}")
            # print(f"Target: {sample_target_decoded}")

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