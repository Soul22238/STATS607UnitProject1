
import sys
import os
import json
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

# Load config
with open(os.path.join(os.path.dirname(__file__), '../config.json'), 'r') as f:
    config = json.load(f)


# Get the model and the processed dataset
conll2000_dataset = pd.read_csv(config['paths']['data_path'], usecols=('tokens', 'pos_tags'))
conll2000_dataset['tokens'] = conll2000_dataset['tokens'].apply(lambda x: ast.literal_eval(x))
conll2000_dataset['pos_tags'] = conll2000_dataset['pos_tags'].apply(lambda x: ast.literal_eval(x))

model_name = config['model']['name']
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = DistilBertModel.from_pretrained(model_name)


# Create DataLoaders
dataset = POSTagDataset(conll2000_dataset, tokenizer)
train_size = int(len(dataset) * config['training']['train_split'])
val_size = int(len(dataset) * config['training']['val_split'])
test_size = len(dataset) - train_size - val_size
(train, val, test) = torch.utils.data.random_split(
    dataset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(config['training']['random_seed'])
)

# grab a test minibatch
test_minibatch = [train[0], train[1]]
batch_in, batch_out = basic_collate_fn(test_minibatch)
print(batch_in['input_ids'].size())


batch_size = config['training']['batch_size']
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, collate_fn=basic_collate_fn, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, collate_fn=basic_collate_fn, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, collate_fn=basic_collate_fn, shuffle=False)

batch_in, pos_ids = next(iter(train_loader))
print(batch_in['attention_mask'].size())


# print(dataset[3]) # verify that this has the correct structure
# {'input_ids': [101, 2104, 1996, 4493, 3206, 1010, 25235, 2056, 1010, 2009, 2038, 2525, 5359, 100, 1997, 1996, 100, 2000, 10321, 1012, 102], 'pos_ids': [1, 12, 8, 10, 5, 20, 3, 6, 20, 7, 18, 23, 21, 16, 12, 8, 15, 13, 3, 17, 2], 'tokens': ['[CLS]', 'under', 'the', 'existing', 'contract', ',', 'rockwell', 'said', ',', 'it', 'has', 'already', 'delivered', '793', 'of', 'the', 'shipsets', 'to', 'boeing', '.', '[SEP]']}
# print(dataset[6]) # verify that this has the correct structure
# {'input_ids': [101, 100, 100, 1010, 5354, 2086, 2214, 1010, 2366, 2004, 3639, 3187, 1999, 1996, 11531, 3447, 1012, 102], 'pos_ids': [1, 3, 3, 20, 16, 15, 9, 20, 21, 12, 5, 5, 12, 8, 3, 5, 17, 2], 'tokens': ['[CLS]', 'mr.', 'carlucci', ',', '59', 'years', 'old', ',', 'served', 'as', 'defense', 'secretary', 'in', 'the', 'reagan', 'administration', '.', '[SEP]']}

# Run this lines to reload the model

hidden_dim = config['model']['hidden_dim']
num_pos_tags = len(dataset.pos_tags.keys())
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

optim = get_optimizer(model, lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), config['paths']['artifacts_dir']))
best_model, stats = train_model(
    model, val_loader, val_loader, optim,
    num_epoch=config['training']['num_epochs']['test_fit'],
    collect_cycle=config['training']['collect_cycle']['test_fit'],
    device=device, save_dir=save_dir, save_name="Test_fit_function"
)

plot_loss(stats, save_dir=save_dir, save_name="Test_fit_function")

# Train the full model
# Run this lines to reload the model

save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), config['paths']['artifacts_dir']))
bert_model = DistilBertModel.from_pretrained(model_name)
model = DistilBertForTokenClassification(bert_model, hidden_dim, num_pos_tags)

# Run the full optimization
model.to(device)
optim = get_optimizer(model, lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
best_model, stats = train_model(
    model, train_loader, val_loader, optim,
    num_epoch=config['training']['num_epochs']['full_train'],
    collect_cycle=config['training']['collect_cycle']['full_train'],
    device=device, save_dir=save_dir, save_name="Train_full"
)
plot_loss(stats, save_dir=save_dir, save_name="Train_full_model")