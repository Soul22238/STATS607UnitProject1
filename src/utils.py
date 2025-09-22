import time
import copy
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optimizer
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
import json
from typing import List

def get_loss_fn():
    """Return the loss function for training the model.

    Uses nn.CrossEntropyLoss and ignores the padding index (hardcoded as 0).

    Returns:
        nn.CrossEntropyLoss: Loss function instance.
    """
    PAD_INDEX = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_INDEX)
    return loss_fn


def calculate_loss(scores, labels, loss_fn):
    """Calculate the loss value.

    Args:
        scores (torch.Tensor): Output scores from the model.
        labels (torch.Tensor): True labels.
        loss_fn (nn.Module): Loss function.

    Returns:
        torch.Tensor: Computed loss value.
    """
    return loss_fn(scores, labels)


def get_optimizer(net, lr, weight_decay):
    """Return the Adam optimizer for training the model.

    Args:
        net (nn.Module): Model to optimize.
        lr (float): Initial learning rate.
        weight_decay (float): Weight decay parameter.

    Returns:
        torch.optim.Optimizer: Adam optimizer instance.
    """
    return optimizer.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    # return torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)


def get_hyper_parameters():
    """Return a list of hyperparameters to search.

    Returns:
        tuple: Lists of learning rates, dropout rates, hidden dimensions, and number of layers.
    """
    print("get_hyperparameter function used!")
    lr = [0.1, 0.01, 0.004]
    dropout_rates = [0.05, 0.1]
    hidden_dims = [50, 200]
    num_layers = [1, 3]
    return lr, dropout_rates, hidden_dims, num_layers


def train_model(net, trn_loader, val_loader, optim, num_epoch=50, collect_cycle=30,
        device='cpu', verbose=True, save_dir=None, save_name=None):
    """Train the model and track training statistics.

    Args:
        net (nn.Module): Model to train.
        trn_loader (DataLoader): Dataloader for training data.
        val_loader (DataLoader): Dataloader for validation data.
        optim (torch.optim.Optimizer): Optimizer.
        num_epoch (int): Number of epochs to train.
        collect_cycle (int): Iterations between collecting training statistics.
        device (str): Device to use ('cpu' or 'cuda').
        verbose (bool): Whether to print training details.
        save_dir (str, optional): Directory to save logs and model.
        save_name (str, optional): Prefix for saved files.

    Returns:
        tuple: Best model and training statistics.
    """
    print("train_model used")
    train_loss, train_loss_ind, val_loss, val_loss_ind = [], [], [], []
    num_itr = 0
    best_model, best_accuracy = None, 0
    loss_fn = get_loss_fn()
    logs = []
    if verbose:
        logs.append('------------------------ Start Training ------------------------')
        print('------------------------ Start Training ------------------------')
    t_start = time.time()
    for epoch in range(num_epoch):
        net.train()
        for inputs, pos_ids in trn_loader:
            num_itr += 1
            inputs = {key: value.to(device) if key != 'seq_lens' else value for key, value in inputs.items()}
            pos_ids = pos_ids.to(device)
            net = net.to(device)
            optim.zero_grad()
            scores = net(**inputs)
            loss = calculate_loss(scores=scores, labels=pos_ids, loss_fn=loss_fn)
            loss.backward()
            optim.step()
            if num_itr % collect_cycle == 0:
                train_loss.append(loss.item())
                train_loss_ind.append(num_itr)
        if verbose:
            msg = 'Epoch No. {0}--Iteration No. {1}-- batch loss = {2:.4f}'.format(
                epoch + 1,
                num_itr,
                loss.item()
                )
            logs.append(msg)
            print(msg)
        accuracy, loss = get_validation_performance(net, loss_fn, val_loader, device)
        val_loss.append(loss)
        val_loss_ind.append(num_itr)
        if verbose:
            msg_acc = "Validation accuracy: {:.4f}".format(accuracy)
            msg_loss = "Validation loss: {:.4f}".format(loss)
            logs.append(msg_acc)
            logs.append(msg_loss)
            print(msg_acc)
            print(msg_loss)
        if accuracy > best_accuracy:
            best_model = copy.deepcopy(net)
            best_accuracy = accuracy
    t_end = time.time()
    if verbose:
        msg_time = 'Training lasted {0:.2f} minutes'.format((t_end - t_start)/60)
        logs.append(msg_time)
        logs.append('------------------------ Training Done ------------------------')
        print(msg_time)
        print('------------------------ Training Done ------------------------')
    stats = {'train_loss': train_loss,
             'train_loss_ind': train_loss_ind,
             'val_loss': val_loss,
             'val_loss_ind': val_loss_ind,
             'accuracy': best_accuracy,
    }
    logs.append(f"train_model {stats}")
    print("train_model", stats)
    # Save logs and model if save_dir and save_name are provided
    if save_dir is not None and save_name is not None:
        print("Saving logs and model...")
        os.makedirs(save_dir, exist_ok=True)
        log_path = os.path.join(save_dir, f"{save_name}_log.txt")
        with open(log_path, "w") as f:
            for line in logs:
                f.write(line + "\n")
        model_path = os.path.join(save_dir, f"{save_name}_model.pt")
        print(f"Log will be saved to: {os.path.abspath(log_path)}")
        print(f"Model will be saved to: {os.path.abspath(model_path)}")
        torch.save(best_model.state_dict(), model_path)
    return best_model, stats


def get_validation_performance(net, loss_fn, data_loader, device):
    """Evaluate model performance on validation or test data.

    Args:
        net (nn.Module): Model to evaluate.
        loss_fn (nn.Module): Loss function.
        data_loader (DataLoader): DataLoader for evaluation data.
        device (str): Device to use ('cpu' or 'cuda').

    Returns:
        tuple: Accuracy and average loss on the validation set.
    """
    print("validation function used")
    net.eval()
    y_true = []
    y_pred = []
    total_loss = []
    with torch.no_grad():
        for inputs, pos_ids in data_loader:
            inputs = {key: value.to(device) if key != 'seq_lens' else value for key, value in inputs.items()}
            pos_ids = pos_ids.to(device)
            net = net.to(device)
            scores = net(**inputs)
            loss = calculate_loss(scores=scores, labels=pos_ids, loss_fn=loss_fn)
            pred = torch.argmax(scores, dim=1)
            total_loss.append(loss.item())
            y_true.append(pos_ids)
            y_pred.append(pred)
    correct = 0
    total = 0
    for batch_id, pos_ids in enumerate(y_true):
        mask = pos_ids.bool()
        batch_eval = (y_pred[batch_id][mask] == pos_ids[mask])
        correct += torch.sum(batch_eval)
        valid_lengths = torch.sum(mask, dim=1)
        total += torch.sum(valid_lengths)
    accuracy = (correct / total).item()
    total_loss = sum(total_loss) / len(total_loss)
    print("accuracy is:", accuracy)
    print("total loss is", total_loss)
    return accuracy, total_loss


def make_prediction(net, pos_tags, data_loader, device):
    """Use the model to make predictions on test data.

    Args:
        net (nn.Module): Model to use for prediction.
        pos_tags (dict): Mapping from pos_tag to pos_id.
        data_loader (DataLoader): DataLoader for test data.
        device (str): Device to use ('cpu' or 'cuda').

    Returns:
        tuple: Lists of true tags, predicted tags, and error indices.
    """
    print("make_prediction function used!")
    net.eval()
    y_pred = []
    y_true = []
    errors = []
    ind = 0
    with torch.no_grad():
        for inputs, pos_ids in data_loader:
            inputs = {key: value.to(device) if key != 'seq_lens' else value for key, value in inputs.items()}
            id2pos_tag = {v: k for k, v in pos_tags.items()}
            net = net.to(device)
            scores = net(**inputs)
            pred = torch.argmax(scores, dim=1)
            mask = pos_ids.bool()
            seq_lens = torch.sum(mask, dim=1)
            for index, sentence_pred_ids in enumerate(pred):
                seq_len = seq_lens[index]
                pred_tags = [id2pos_tag[tag_id.item()] for tag_id in sentence_pred_ids[:seq_len]]
                true_tags = [id2pos_tag[tag_id.item()] for tag_id in pos_ids[index][:seq_len]]
                eval = (sentence_pred_ids[:seq_len].cpu() != pos_ids[index][:seq_len].cpu())
                error = [(ind, pos) for pos in range(len(eval)) if eval[pos]]
                y_pred.append(pred_tags)
                y_true.append(true_tags)
                errors.extend(error)
                ind += 1
    return y_true, y_pred, errors


def plot_loss(stats, save_dir=None, save_name=None):
    """Plot training and validation loss curves.

    Args:
        stats (dict): Dictionary containing training and validation loss statistics.
        save_dir (str, optional): Directory to save the plot image.
        save_name (str, optional): Prefix for saved plot image file.
    """
    plt.plot(stats['train_loss_ind'], stats['train_loss'], label='Training loss')
    plt.plot(stats['val_loss_ind'], stats['val_loss'], label='Validation loss')
    plt.legend()
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss')
    if save_dir is not None and save_name is not None:
        os.makedirs(save_dir, exist_ok=True)
        img_path = os.path.join(save_dir, f"{save_name}_loss.png")
        plt.savefig(img_path)
        print(f"Loss plot saved to: {os.path.abspath(img_path)}")
    else:
        plt.show()

def compute_pos_accuracy(y_true, y_pred, results_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results/'))) -> float:
    """Compute POS tagging accuracy excluding special tokens.

    This function compares predicted and true POS tags, ignoring
    the special tokens `[CLS]` and `[SEP]`. The accuracy is defined
    as the number of correctly predicted tags divided by the total
    number of valid tags.

    Args:
        y_true (List[List[str]]): Nested list of true POS tags.
        y_pred (List[List[str]]): Nested list of predicted POS tags.
        results_dir (str): Directory where the accuracy result
            will be saved as JSON. Defaults to "results".

    Returns:
        float: POS tagging accuracy.
    """
    # Get absolute path to results directory
    
    total = 0
    correct = 0

    for true_tags, pred_tags in zip(y_true, y_pred):
        for t, p in zip(true_tags, pred_tags):
            if t in ["[CLS]", "[SEP]"]:
                continue
            total += 1
            if t == p:
                correct += 1

    accuracy = correct / total if total > 0 else 0.0

    # Save to results directory
    Path(results_dir).mkdir(exist_ok=True)
    out_path = Path(results_dir) / "pos_accuracy.json"
    print(f"Saving accuracy result to: {os.path.abspath(out_path)}")
    with open(out_path, "w") as f:
        json.dump({"pos_accuracy": accuracy}, f, indent=4)

    return accuracy
