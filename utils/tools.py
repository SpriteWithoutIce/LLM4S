import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from datetime import datetime
from distutils.util import strtobool
import pandas as pd

plt.switch_backend('agg')


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


def vali(model, test_loader, criterion, args):
    valid_loss = []
    correct_predictions = 0
    total_predictions = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            if args.model == 'gpt2' or args.model == 'llama' or args.model == 'qwen':
                outputs = model(inputs, lstm=args.lstm)
            else:
                outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss.append(loss.item())
            predicted = torch.argmax(outputs, dim=1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    valid_loss = np.average(valid_loss)
    valid_acc = correct_predictions / total_predictions

    return valid_loss, valid_acc


def test(model, test_loader, args):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            if args.model == 'gpt2' or args.model == 'llama' or args.model == 'qwen':
                outputs = model(inputs, lstm=args.lstm)
            else:
                outputs = model(inputs)
            # Accuracy
            predicted = torch.argmax(outputs, dim=1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    test_acc = correct_predictions / total_predictions
    return test_acc


# test
    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            if parameters.model == 'gpt2' or parameters.model == 'llama' or parameters.model == 'qwen':
                outputs = model(inputs, lstm=parameters.lstm)
            else:
                outputs = model(inputs)
            test_loss += criterion(outputs, labels)
            predicted = torch.argmax(outputs, dim=1)

            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    test_loss /= len(test_loader)
    test_acc = correct_predictions / total_predictions
    print(
        f"test_loss: {test_loss:.4f}, test_accuracy: {test_acc:.4f}")
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        # best_model_state = model.state_dict()

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_model_state = model.state_dict()

    res_test = {'test/loss': test_loss, 'test/acc': test_acc,
                'best/test_loss': best_test_loss, 'best/test_acc': best_test_acc}

    print(f"best/test_loss {best_test_loss}, best/test_acc: {best_test_acc}")
    res = {**res_train, **res_test}
    if args.wandb:
        wandb.log(res)
