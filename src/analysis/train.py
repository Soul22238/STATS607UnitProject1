import pandas as pd
import torch
import ast
from torch.utils.data import Dataset, DataLoader
from torch import nn
import transformers
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from transformers import DistilBertModel, DistilBertTokenizerFast
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

conll2000_dataset = pd.read_csv('./data/conll2000.csv', usecols=('tokens', 'pos_tags'))
conll2000_dataset['tokens'] = conll2000_dataset['tokens'].apply(lambda x: ast.literal_eval(x))
conll2000_dataset['pos_tags'] = conll2000_dataset['pos_tags'].apply(lambda x: ast.literal_eval(x))


dataset = POSTagDataset(conll2000_dataset, tokenizer)
(train, val, test) = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))

print(dataset[3]) # verify that this has the correct structure


# Run this lines to reload the model
bert_model = DistilBertModel.from_pretrained(model_name)
model = DistilBertForTokenClassification(bert_model, hidden_dim, num_pos_tags)

# Run some test optimization
model.to(device)
optim = get_optimizer(model, lr=5e-5, weight_decay=0)
best_model, stats = train_model(model, val_loader, val_loader, optim,
                                num_epoch=25, collect_cycle=5, device=device)

plot_loss(stats)

# Trian the full model
# Run this lines to reload the model
bert_model = DistilBertModel.from_pretrained(model_name)
model = DistilBertForTokenClassification(bert_model, 768, num_pos_tags)

# Run the full optimization
model.to(device)
optim = get_optimizer(model, lr=5e-5, weight_decay=0)
best_model, stats = train_model(model, train_loader, val_loader, optim,
                                num_epoch=10, collect_cycle=20, device=device)