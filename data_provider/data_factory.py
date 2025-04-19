from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torch
from sklearn.impute import SimpleImputer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


def data_provider(args, flag):
    # train test
    train_dataset_name = args.train_dataset_name
    test_dataset_name = args.test_dataset_name

    data = pd.read_csv(train_dataset_name)
    data_test = pd.read_csv(test_dataset_name)

    data.head()
    data_1 = data["heartbeat_signals"].str.split(",", expand=True)
    data_test_1 = data_test["heartbeat_signals"].str.split(",", expand=True)
    np.array(data.label)

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    data_2 = np.array(data_1).astype("float32").reshape(-1, 100)
    data_2 = imputer.fit_transform(data_2)
    data_test_2 = np.array(data_test_1).astype("float32").reshape(-1, 100)
    data_test_2 = imputer.fit_transform(data_test_2)

    scaler = StandardScaler()
    data_2 = scaler.fit_transform(data_2.reshape(-1, 100)).reshape(-1, 100, 1)
    data_test_2 = scaler.transform(
        data_test_2.reshape(-1, 100)).reshape(-1, 100, 1)

    torch.set_printoptions(precision=7)

    x_train = torch.tensor(data_2)
    x_test = torch.tensor(data_test_2)
    y_train = torch.tensor(data.label, dtype=int)
    y_test = torch.tensor(data_test.answer, dtype=int)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    dataset = TensorDataset(x_train, y_train)
    dataset_test = TensorDataset(x_test, y_test)
    train_loader = DataLoader(dataset, batch_size=32,
                              shuffle=True, pin_memory=False)
    test_loader = DataLoader(dataset_test, batch_size=32,
                             shuffle=True, pin_memory=False)
    if flag == 'train':
        return train_loader
    elif flag == 'test':
        return test_loader
