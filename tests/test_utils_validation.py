import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import pytest
import pandas as pd
import torch
import os
import json
from transformers import AutoTokenizer, DistilBertModel
from src.utils import *
from src.analysis.model import *

def test_data_validation():
    # Test expected columns and no missing critical data
    with open('src/config.json', 'r') as f:
        config = json.load(f)
    data_path = config['paths']['data_path']
    df = pd.read_csv(data_path)
    assert 'tokens' in df.columns
    assert 'pos_tags' in df.columns
    assert df['tokens'].notnull().all()
    assert df['pos_tags'].notnull().all()
    # Check value ranges (tokens should be list, pos_tags should be list)
    for t, p in zip(df['tokens'], df['pos_tags']):
        assert isinstance(eval(t), list)
        assert isinstance(eval(p), list)
        assert len(eval(t)) == len(eval(p))

def test_loss_fn_shape():
    # Test loss function expects correct shape
    loss_fn = get_loss_fn()
    scores = torch.randn(10, 5)  # [batch, num_classes]
    labels = torch.randint(0, 5, (10,))
    loss = loss_fn(scores, labels)
    assert loss.dim() == 0

def test_optimizer_type():
    # Test optimizer returns correct type
    class DummyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 2)
        def forward(self, x):
            return self.fc(x)
    net = DummyNet()
    optim = get_optimizer(net, lr=0.01, weight_decay=0)
    assert hasattr(optim, 'step')

def test_model_output_shape():

    with open('src/config.json', 'r') as f:
        config = json.load(f)
    data_path = config['paths']['data_path']
    df = pd.read_csv(data_path)
    model_name = config['model']['name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = DistilBertModel.from_pretrained(model_name)
    from src.analysis.dataPreprocess import POSTagDataset, basic_collate_fn
    dataset = POSTagDataset(df, tokenizer)
    num_pos_tags = len(dataset.pos_tags.keys())
    batch_size = config['training']['batch_size']

    train_size = int(len(dataset) * config['training']['train_split'])
    val_size = int(len(dataset) * config['training']['val_split'])
    test_size = len(dataset) - train_size - val_size
    train, val, test = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(config['training']['random_seed']))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, collate_fn=basic_collate_fn, shuffle=True)
    batch_in, pos_ids = next(iter(train_loader))
    model = DistilBertForTokenClassification(bert_model, config['model']['hidden_dim'], num_pos_tags)
    output = model(**batch_in)
    # output shape: [batch_size, num_pos_tags, seq_len]
    assert output.shape[0] == batch_size or output.shape[0] == len(train)
    assert output.shape[1] == num_pos_tags
    assert output.shape[2] == batch_in['input_ids'].shape[1]

def test_compute_pos_accuracy_shape():
    # Test compute_pos_accuracy returns float and saves file
    y_true = [['NN', 'VB', 'DT'], ['JJ', 'NN', 'VB']]
    y_pred = [['NN', 'VB', 'DT'], ['JJ', 'NN', 'VB']]
    acc = compute_pos_accuracy(y_true, y_pred, results_dir='results/')
    assert isinstance(acc, float)
    assert os.path.exists('results/pos_accuracy.json')

def test_results_dimensions():
    # Test results.py output dimensions
    with open('results/pos_accuracy.json', 'r') as f:
        res = json.load(f)
    assert 'pos_accuracy' in res
    assert 0.0 <= res['pos_accuracy'] <= 1.0
