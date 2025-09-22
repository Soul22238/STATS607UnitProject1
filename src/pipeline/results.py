import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
from analysis.model import *
from analysis.dataPreprocess import *
import pandas as pd
import ast

# Load config
with open(os.path.join(os.path.dirname(__file__), '../config.json'), 'r') as f:
    config = json.load(f)

# Check if CUDA (GPU) is available
if torch.cuda.is_available():
    device = "cuda" # Use NVIDIA GPU (if available)
elif torch.backends.mps.is_available():
    device = "mps" # Use Apple Silicon GPU (if available)
else:
    device = "cpu" # Default to CPU if no GPU is available
print(device)

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

hidden_dim = config['model']['hidden_dim']
num_pos_tags = len(dataset.pos_tags.keys())
bert_model = DistilBertModel.from_pretrained(model_name)

# Test and make predictions
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), config['paths']['artifacts_dir'], 'Train_full_model.pt'))

best_model = DistilBertForTokenClassification(bert_model, hidden_dim, num_pos_tags)
best_model.load_state_dict(torch.load(model_path))

get_validation_performance(best_model.to(device), get_loss_fn(), test_loader, device)
y_true, y_pred, errors = make_prediction(best_model, train.dataset.pos_tags, test_loader, device)
results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), config['paths']['results_dir']))
results = compute_pos_accuracy(y_true, y_pred, results_dir=results_dir)
print(f"POS Tagging Accuracy: {results}")
