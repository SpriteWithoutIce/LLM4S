from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, vali, test
from tqdm import tqdm
from models.GPT2 import GPT4TS
from models.Qwen import Qwen4TS
from models.Llama import Llama4TS
from models.lstm import LSTM4TS
from models.gru import GRU4TS
from models.cnn import CNN_TS
import numpy as np
import torch
import torch.nn as nn

import os
import time

import warnings
import numpy as np

import argparse
import random

warnings.filterwarnings('ignore')

fix_seed = 2025
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='GPT4TS')

parser.add_argument('--model_id', type=str, required=True, default='test')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

parser.add_argument('--train_dataset_name', type=str,
                    default='./dataset/action-im/train_dataset_yzt_im.csv')
parser.add_argument('--test_dataset_name', type=str,
                    default='./dataset/action-im/test_dataset_yzt_im.csv')
parser.add_argument('--data', type=str, default='lyh_2')

parser.add_argument('--learning_rate', type=float, default=0.0005)
parser.add_argument('--batch_size', type=int, default=512)

parser.add_argument('--train_epochs', type=int, default=300)
parser.add_argument('--patience', type=int, default=100)

parser.add_argument('--input_dim', type=int, default=1)
parser.add_argument('--seq_len', type=int, default=100)
parser.add_argument('--num_classes', type=int, default=5)
parser.add_argument('--lora', type=bool, default=True)
parser.add_argument('--lstm', type=bool, default=True)
parser.add_argument('--gpt_layers', type=int, default=3)
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--d_model', type=int, default=256)
parser.add_argument('--window', type=int, default=10)
parser.add_argument('--d_LLM', type=int, default=768)
parser.add_argument('--cycle_len', type=int, default=10)
parser.add_argument('--patch', type=bool, default=False)
parser.add_argument('--isconv', type=bool, default=False)

parser.add_argument('--model', type=str, default='Qwen')
parser.add_argument('--stride', type=int, default=8)


args = parser.parse_args()

setting = '{}_gl{}'.format(args.model_id, args.gpt_layers)
path = os.path.join(args.checkpoints, setting)
if not os.path.exists(path):
    os.makedirs(path)

train_loader = data_provider(args, 'train')
test_loader = data_provider(args, 'test')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

time_now = time.time()
train_steps = len(train_loader)

if args.model == 'GPT2':
    model = GPT4TS(args, device).to(device)
elif args.model == 'Qwen':
    model = Qwen4TS(args, device).to(device)
elif args.model == 'Llama':
    model = Llama4TS(args, device).to(device)
elif args.model == 'lstm':
    model = LSTM4TS(args, hidden_size=256, lstm_layers=2).to(device)
elif args.model == 'gru':
    model = GRU4TS(args, gru_layers=2, hidden_size=256).to(device)
elif args.model == 'cnn':
    model = CNN_TS(args).to(device)

criterion = nn.CrossEntropyLoss()
params = model.parameters()
model_optim = torch.optim.Adam(params, lr=args.learning_rate)

early_stopping = EarlyStopping(patience=args.patience, verbose=True)

scheduler = torch.optim.lr_scheduler.StepLR(
    model_optim, step_size=10, gamma=0.5)

for epoch in range(args.train_epochs):
    iter_count = 0
    train_loss = []
    epoch_time = time.time()
    correct_predictions = 0
    total_predictions = 0
    for inputs, labels in train_loader:
        iter_count += 1
        model.train()
        model_optim.zero_grad()

        if args.model == 'GPT2' or args.model == 'llama' or args.model == 'Qwen':
            outputs = model(inputs, lstm=args.lstm)
        else:
            outputs = model(inputs)

        loss = criterion(outputs, labels)
        train_loss.append(loss.item())

        loss.backward()
        model_optim.step()
        # Accuracy
        predicted = torch.argmax(outputs, dim=1)
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

    print("Epoch: {} cost time: {}".format(
        epoch + 1, time.time() - epoch_time))

    train_loss = np.average(train_loss)
    train_acc = correct_predictions / total_predictions
    vali_loss, vali_acc = vali(model, test_loader, criterion, args)
    print("Epoch: {0}, Steps: {1} | Train Loss: {2:.4f} Accuracy: {3:.4f} | Vali Loss: {4:.4f} Accuracy: {5:.4f}".format(
        epoch + 1, train_steps, train_loss, train_acc, vali_loss, vali_acc))
    early_stopping(vali_acc, model, path)
    if early_stopping.early_stop:
        print("Early stopping")
        break

best_model_path = path + '/' + 'checkpoint.pth'
model.load_state_dict(torch.load(best_model_path))
model = model.to(device)
print("------------------------------------")
# test
test_acc = test(model, test_loader, args)
print('test acc:{:.4f}'.format(test_acc))
