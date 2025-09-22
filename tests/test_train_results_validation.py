import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import pytest
import os
import json
import torch
import pandas as pd

from src.analysis.dataPreprocess import *
from src.analysis.model import DistilBertForTokenClassification
from transformers import AutoTokenizer, DistilBertModel


def test_train_data_split():
    # Check train/val/test split sizes add up and are correct type
    with open('src/config.json', 'r') as f:
        config = json.load(f)
    data_path = config['paths']['data_path']
    df = pd.read_csv(data_path)
    model_name = config['model']['name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = POSTagDataset(df, tokenizer)
    train_size = int(len(dataset) * config['training']['train_split'])
    val_size = int(len(dataset) * config['training']['val_split'])
    test_size = len(dataset) - train_size - val_size
    assert train_size + val_size + test_size == len(dataset)
    assert all(isinstance(x, int) for x in [train_size, val_size, test_size])

def test_train_loader_batch_shape():
    # Check train loader batch dimensions
    with open('src/config.json', 'r') as f:
        config = json.load(f)
    data_path = config['paths']['data_path']
    df = pd.read_csv(data_path)
    model_name = config['model']['name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = POSTagDataset(df, tokenizer)
    train_size = int(len(dataset) * config['training']['train_split'])
    val_size = int(len(dataset) * config['training']['val_split'])
    test_size = len(dataset) - train_size - val_size
    train, val, test = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(config['training']['random_seed']))
    batch_size = config['training']['batch_size']
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, collate_fn=basic_collate_fn, shuffle=True)
    batch_in, pos_ids = next(iter(train_loader))
    assert batch_in['input_ids'].shape[0] == batch_size or batch_in['input_ids'].shape[0] == len(train)
    assert batch_in['input_ids'].ndim == 2
    assert pos_ids.ndim == 2 or pos_ids.ndim == 1

def test_results_accuracy_file():
    # Check results/pos_accuracy.json exists and has valid accuracy
    results_path = 'results/pos_accuracy.json'
    assert os.path.exists(results_path)
    with open(results_path, 'r') as f:
        res = json.load(f)
    assert 'pos_accuracy' in res
    assert 0.0 <= res['pos_accuracy'] <= 1.0

def test_results_model_load():
    # Check model loading and prediction shape
    with open('src/config.json', 'r') as f:
        config = json.load(f)
    data_path = config['paths']['data_path']
    df = pd.read_csv(data_path)
    model_name = config['model']['name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = POSTagDataset(df, tokenizer)
    train_size = int(len(dataset) * config['training']['train_split'])
    val_size = int(len(dataset) * config['training']['val_split'])
    test_size = len(dataset) - train_size - val_size
    train, val, test = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(config['training']['random_seed']))
    batch_size = config['training']['batch_size']
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, collate_fn=basic_collate_fn, shuffle=False)
    hidden_dim = config['model']['hidden_dim']
    num_pos_tags = len(dataset.pos_tags.keys())
    bert_model = DistilBertModel.from_pretrained(model_name)
    model = DistilBertForTokenClassification(bert_model, hidden_dim, num_pos_tags)
    model_path = os.path.abspath(os.path.join('artifacts', 'Train_full_model.pt'))
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.to('cpu')
        batch_in, pos_ids = next(iter(test_loader))
        scores = model(**batch_in)
        assert scores.shape[0] == batch_in['input_ids'].shape[0]
        assert scores.shape[1] == num_pos_tags

def test_fine_dataset_1():
    """Test that input_ids, pos_ids, tokens have correct length (+2 for [CLS], [SEP])."""
    with open('src/config.json', 'r') as f:
        config = json.load(f)
    df = pd.read_csv(config['paths']['data_path'])
    model_name = config['model']['name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = POSTagDataset(df, tokenizer)
    # Check first sample
    assert len(dataset[0]['input_ids']) == len(df.iloc[0]['tokens']) + 2
    assert len(dataset[0]['pos_ids']) == len(df.iloc[0]['pos_tags']) + 2
    assert len(dataset[0]['tokens']) == len(df.iloc[0]['tokens']) + 2

def test_fine_dataset_2():
    """Test that input_ids match expected values for a known sentence."""
    with open('src/config.json', 'r') as f:
        config = json.load(f)
    df = pd.read_csv(config['paths']['data_path'])
    model_name = config['model']['name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = POSTagDataset(df, tokenizer)
    # Example input_ids for a known sentence (update as needed)
    input_ids = [101, 1031, 1005, 1054]
    for i in range(len(input_ids)):
        assert dataset[0]['input_ids'][i] == input_ids[i]

def test_fine_collate_1():
    """Test collate_fn output shapes for a minibatch."""
    with open('src/config.json', 'r') as f:
        config = json.load(f)
    df = pd.read_csv(config['paths']['data_path'])
    model_name = config['model']['name']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = POSTagDataset(df, tokenizer)
    test_minibatch = [dataset[i] for i in range(5)]
    batch_in, batch_out = basic_collate_fn(test_minibatch)
    assert batch_in['input_ids'].shape == (5, batch_in['input_ids'].shape[1])
    assert batch_out.shape == (5, batch_out.shape[1])
