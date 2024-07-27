import pandas as pd
import os
from dataset import create_dataset
from models import LSTM_Base, AirModel
import torch
import numpy as np
import matplotlib.pyplot as plt

in_dir="../smoothed/ws4/"
df = pd.read_csv(os.path.join(in_dir, 'data.csv'))
keys=["MF_Smoothed","OF_Smoothed","MNU_Smoothed","ONU_Smoothed"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models=["MF_Smoothed/1.55_complex_2_0.9_8_1300.pt",
        "OF_Smoothed/1.21_complex_2_0.9_8_1400.pt",
        "MNU_Smoothed/0.3_complex_2_0.9_8_1100.pt",
        "ONU_Smoothed/0.93_complex_2_0.9_8_1500.pt"]

lookback=2
split_ratio=0.9

df['Year'] = df['Year'].astype(str)
annotations = {'MF_Smoothed': 'Manufacturer-controllable factors',
               'OF_Smoothed': 'Organization- or user-controllable factors',
               'MNU_Smoothed': 'Manufacturer-influenced non-use',
               'ONU_Smoothed': 'Organization- or user-controllable non-use'}
for i, key in enumerate(keys):
    print(i, 'key', models[i])
    timeseries = df[[key]].values.astype('float32')
    train_size = int(len(timeseries) * split_ratio)
    test_size = len(timeseries) - train_size
    train, test = timeseries[:train_size], timeseries[train_size:]
    X_train, y_train = create_dataset(train, lookback=lookback)
    X_test, y_test = create_dataset(test, lookback=lookback)
    model=torch.load(os.path.join("Checkpoints",models[i])).to(device)
    model.eval()

    with torch.no_grad():
        # shift train predictions for plotting
        train_plot = np.ones_like(timeseries) * np.nan
        y_pred_train = model(X_train.to(device))
        y_pred_train = y_pred_train[:, -1, :]
        train_plot[lookback:train_size] = y_pred_train.cpu()
        # shift test predictions for plotting
        test_plot = np.ones_like(timeseries) * np.nan
        y_pred_test = model(X_test.to(device))
        y_pred_test = y_pred_test[:, -1, :]
        test_plot[train_size + lookback:len(timeseries)] = y_pred_test.cpu()

    # plot
    plt.figure(figsize=(12,6))
    plt.plot(timeseries, label='Original Data')
    plt.title(annotations[key])
    plt.plot(df['Year'], train_plot,label='Train Predictions', c='r')
    plt.plot(df['Year'], test_plot, label='Test Predictions', c='g')
    plt.legend()
    plt.xlim([df['Year'].min(), df['Year'].max()])
    tick_spacing = 4
    plt.xticks(df['Year'][::tick_spacing])
    plt.setp(plt.gca().get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    plt.savefig(f'visualize_test/{key}.jpg')
    plt.show()
