import pandas as pd
import torch
from torch.utils.data import Dataset
from torch import nn
import transformers
from torch.nn.utils.rnn import pad_sequence
conll2000_dataset = pd.read_csv('conll2000.csv', usecols=('tokens', 'pos_tags'))
conll2000_dataset['tokens'] = conll2000_dataset['tokens'].apply(lambda x: ast.literal_eval(x))
conll2000_dataset['pos_tags'] = conll2000_dataset['pos_tags'].apply(lambda x: ast.literal_eval(x))

class POSTagDataset(Dataset):
    """Dataset for POS tagging"""

    def __init__(self, data: pd.DataFrame, tokenizer: transformers.AutoTokenizer):
        super().__init__()

        # Fill in this dictionary by adding the rest of the POS tags
        self.pos_tags = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2}

        # Append each entry to this list
        self.data = []

        ############################## START OF YOUR CODE ##############################
        num = 3
        for ind, tokens in enumerate(data["tokens"]):
            tokens_change = [token.lower() for token in tokens]
            tokens_change.insert(0,"[CLS]")
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
                "pos_ids":pos_ids,
                "tokens":tokens_change
            })
    
        ############################### END OF YOUR CODE ###############################
    
    def __len__(self):
        ############################## START OF YOUR CODE ##############################
        return len(self.data)
    
        ############################### END OF YOUR CODE ###############################

    def __getitem__(self, idx):
        ############################## START OF YOUR CODE ##############################
        return self.data[idx]
    
        ############################### END OF YOUR CODE ###############################


def basic_collate_fn(batch):
    """Collate function for basic setting."""

    inputs = None
    outputs = None

    ############################## START OF YOUR CODE ##############################
    # Formalize input such that they are all of same length
    inputs = [torch.tensor(data["input_ids"]) for data in batch]
    inputs = pad_sequence(inputs, batch_first=True, padding_value=0)

    # Formalize output such that they are all of same length
    outputs = [torch.tensor(data["pos_ids"]) for data in batch]
    outputs = pad_sequence(outputs, batch_first=True, padding_value=0)

    attention_mask = [torch.ones(size = (len(data["input_ids"]),)) for data in batch]
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    inputs = {"input_ids":inputs, "attention_mask":attention_mask}
    # print(inputs, outputs)
    # print(inputs["input_ids"].shape, inputs["attention_mask"].shape, outputs.shape)

    ############################### END OF YOUR CODE ###############################

    return inputs, outputs

dataset = POSTagDataset(conll2000_dataset, tokenizer)
(train, val, test) = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))

print(dataset[3]) # verify that this has the correct structure