import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import torch
import ast
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, DistilBertModel
import numpy as np
from dataPreprocess import *
from model import *
from utils import *

# Get the model and the processed dataset
conll2000_dataset = pd.read_csv('./data/conll2000.csv', usecols=('tokens', 'pos_tags'))
conll2000_dataset['tokens'] = conll2000_dataset['tokens'].apply(lambda x: ast.literal_eval(x))
conll2000_dataset['pos_tags'] = conll2000_dataset['pos_tags'].apply(lambda x: ast.literal_eval(x))

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = DistilBertModel.from_pretrained(model_name)

# Create DataLoaders
dataset = POSTagDataset(conll2000_dataset, tokenizer)
(train, val, test) = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))

# grab a test minibatch
test_minibatch = [train[0], train[1]]
batch_in, batch_out = basic_collate_fn(test_minibatch)
print(batch_in['input_ids'].size())

train_loader = torch.utils.data.DataLoader(train, batch_size=64, collate_fn=basic_collate_fn, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size=64, collate_fn=basic_collate_fn, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=64, collate_fn=basic_collate_fn, shuffle=False)

batch_in, pos_ids = next(iter(train_loader))
print(batch_in['attention_mask'].size())


# print(dataset[3]) # verify that this has the correct structure
# {'input_ids': [101, 2104, 1996, 4493, 3206, 1010, 25235, 2056, 1010, 2009, 2038, 2525, 5359, 100, 1997, 1996, 100, 2000, 10321, 1012, 102], 'pos_ids': [1, 12, 8, 10, 5, 20, 3, 6, 20, 7, 18, 23, 21, 16, 12, 8, 15, 13, 3, 17, 2], 'tokens': ['[CLS]', 'under', 'the', 'existing', 'contract', ',', 'rockwell', 'said', ',', 'it', 'has', 'already', 'delivered', '793', 'of', 'the', 'shipsets', 'to', 'boeing', '.', '[SEP]']}
# print(dataset[6]) # verify that this has the correct structure
# {'input_ids': [101, 100, 100, 1010, 5354, 2086, 2214, 1010, 2366, 2004, 3639, 3187, 1999, 1996, 11531, 3447, 1012, 102], 'pos_ids': [1, 3, 3, 20, 16, 15, 9, 20, 21, 12, 5, 5, 12, 8, 3, 5, 17, 2], 'tokens': ['[CLS]', 'mr.', 'carlucci', ',', '59', 'years', 'old', ',', 'served', 'as', 'defense', 'secretary', 'in', 'the', 'reagan', 'administration', '.', '[SEP]']}

# Run this lines to reload the model
hidden_dim = 768 # this is fixed for BERT models
num_pos_tags =  len(dataset.pos_tags.keys())
bert_model = DistilBertModel.from_pretrained(model_name)
model = DistilBertForTokenClassification(bert_model, hidden_dim, num_pos_tags)


# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    device = "cuda" # Use NVIDIA GPU (if available)
elif torch.backends.mps.is_available():
    device = "mps" # Use Apple Silicon GPU (if available)
else:
    device = "cpu" # Default to CPU if no GPU is available
print(device)

# Run some test optimization
model.to(device)
optim = get_optimizer(model, lr=5e-5, weight_decay=0)
save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../artifacts'))
best_model, stats = train_model(model, val_loader, val_loader, optim,
                                num_epoch=25, collect_cycle=5, device=device, save_dir=save_dir, save_name="Test_fit_function")

plot_loss(stats,save_dir=save_dir, save_name="Test_fit_function")

# Train the full model
# Run this lines to reload the model
save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../artifacts'))
bert_model = DistilBertModel.from_pretrained(model_name)
model = DistilBertForTokenClassification(bert_model, 768, num_pos_tags)

# Run the full optimization
model.to(device)
optim = get_optimizer(model, lr=5e-5, weight_decay=0)
best_model, stats = train_model(model, train_loader, val_loader, optim,
                                num_epoch=15, collect_cycle=20, device=device, save_dir=save_dir, save_name="Train_full")
plot_loss(stats,save_dir=save_dir, save_name="Train_full_model_loss")