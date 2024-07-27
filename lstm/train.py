import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import os

from dataset import create_dataset
from models import LSTM_Base, AirModel

model_types=["simple", "complex"]
in_dir="../smoothed/ws4/"
df = pd.read_csv(os.path.join(in_dir, 'data.csv'))
keys=["MF_Smoothed","OF_Smoothed","MNU_Smoothed","ONU_Smoothed"]

batch_size = 8

n_epochs = 2000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for key in keys:
    print("Processing: "+key)
    key_dir=os.path.join("Checkpoints", key)
    os.makedirs(key_dir, exist_ok=True)
    rmse_limit = 1000
    lookbacks = [2, 4, 6]
    split_ratios = [0.8, 0.85, 0.9]
    model = AirModel().to(device)
    model_type ="complex"
    model = LSTM_Base().to(device)
    model_type="simple"
    for lookback in lookbacks:
        for split_ratio in split_ratios:
            print("SETTING: ")
            print(f"MODEL_TYPE: {model_type} LOOKBACK: {lookback} SPLIT_RATIO: {split_ratio} BATCH_SIZE: {batch_size}")
            timeseries = df[[key]].values.astype('float32')
            train_size = int(len(timeseries) * split_ratio)
            test_size = len(timeseries) - train_size
            train, test = timeseries[:train_size], timeseries[train_size:]
            X_train, y_train = create_dataset(train, lookback=lookback)
            X_test, y_test = create_dataset(test, lookback=lookback)
            optimizer = optim.Adam(model.parameters())
            loss_fn = nn.MSELoss()
            loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size)

            #train
            for epoch in range(n_epochs):
                model.train()
                for X_batch, y_batch in loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    y_pred = model(X_batch)
                    loss = loss_fn(y_pred, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # validation
                if epoch % 100 != 0:
                    continue
                model.eval()
                with torch.no_grad():
                    y_pred = model(X_train.to(device))
                    y_pred = y_pred.cpu()
                    train_rmse = np.sqrt(loss_fn(y_pred, y_train))

                    y_pred = model(X_test.to(device))
                    y_pred = y_pred.cpu()
                    test_rmse = np.sqrt(loss_fn(y_pred, y_test))

                    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

                    if test_rmse < rmse_limit:
                        te_rmse=str(round(test_rmse.item(),2))
                        torch.save(model,  f'{key_dir}/{te_rmse}_{model_type}_{lookback}_{split_ratio}_{batch_size}_{epoch}.pt')
                        rmse_limit=test_rmse
