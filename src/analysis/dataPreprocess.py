import pandas as pd
import torch
from torch.utils.data import Dataset
import transformers
from torch.nn.utils.rnn import pad_sequence


class POSTagDataset(Dataset):
    """Dataset for POS tagging.

    Args:
        data (pd.DataFrame): DataFrame containing token and POS tag sequences.
        tokenizer (transformers.AutoTokenizer): Tokenizer for converting tokens to IDs.

    Attributes:
        pos_tags (dict): Mapping from POS tag string to integer ID.
        data (list): List of processed data samples, each as a dict.
    """

    def __init__(self, data: pd.DataFrame, tokenizer: transformers.AutoTokenizer):
        """Initialize the POSTagDataset.

        Builds the POS tag dictionary and processes the input data.

        Args:
            data (pd.DataFrame): DataFrame with 'tokens' and 'pos_tags' columns.
            tokenizer (transformers.AutoTokenizer): Tokenizer for token ID conversion.
        """
        super().__init__()
        self.pos_tags = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2}
        self.data = []

        num = 3
        for ind, tokens in enumerate(data["tokens"]):
            tokens_change = [token.lower() for token in tokens]
            tokens_change.insert(0, "[CLS]")
            tokens_change.append("[SEP]")
            input_ids = tokenizer.convert_tokens_to_ids(tokens_change)
            pos_ids = [1]
            for tk_id, pos in enumerate(data["pos_tags"][ind]):
                if pos not in self.pos_tags:
                    self.pos_tags[pos] = num
                    num += 1
                pos_ids.append(self.pos_tags[pos])
            pos_ids.append(2)
            self.data.append({
                "input_ids": input_ids,
                "pos_ids": pos_ids,
                "tokens": tokens_change
            })

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get a single sample by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Sample containing 'input_ids', 'pos_ids', and 'tokens'.
        """
        return self.data[idx]


def basic_collate_fn(batch):
    """Collate function for batching POSTagDataset samples.

    Pads input and output sequences to the same length for batching.

    Args:
        batch (list): List of samples from POSTagDataset.

    Returns:
        tuple: Dictionary of padded inputs and tensor of padded outputs.
    """
    inputs = [torch.tensor(data["input_ids"]) for data in batch]
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)

    outputs = [torch.tensor(data["pos_ids"]) for data in batch]
    outputs = pad_sequence(outputs, batch_first=True, padding_value=0)

    attention_mask = [torch.ones(size=(len(data["input_ids"]),)) for data in batch]
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    inputs = {"input_ids": inputs, "attention_mask": attention_mask}

    return inputs, outputs

