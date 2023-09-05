from preprocessing.Preprocessing import ret_df
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from deepdow.benchmarks import OneOverN, Random, Benchmark
from deepdow.callbacks import EarlyStoppingCallback
from deepdow.data import InRAMDataset, RigidDataLoader, prepare_standard_scaler, Scale
from deepdow.data.synthetic import sin_single
from deepdow.experiments import Run
from deepdow.layers import SoftmaxAllocator
from deepdow.losses import MeanReturns, SharpeRatio, MaximumDrawdown
from deepdow.visualize import generate_metrics_table, generate_weights_table, plot_metrics, plot_weight_heatmap

raw_df = ret_df()


#### Portfolio2

idx = pd.IndexSlice
port2_close = raw_df.loc[idx[:], idx[:,'close']]

idx = pd.IndexSlice
port2_rsi = raw_df.loc[idx[:], idx[:,'rsi']]

idx = pd.IndexSlice
port2_high = raw_df.loc[idx[:], idx[:,'high']]

idx = pd.IndexSlice
port2_sma25 = raw_df.loc[idx[:], idx[:,'sma25']]

idx = pd.IndexSlice
port2_sma200 = raw_df.loc[idx[:], idx[:,'sma200']]

idx = pd.IndexSlice
port2_volume = raw_df.loc[idx[:], idx[:,'volume']]



close2 = port2_close[['AAPL', 'ADBE','ABT', 'BMY', 'COF', 'GS', 'SPG', 'AMT', 'PM', 'KHC', 'DUK', 'NEE', 'NFLX', 'T', 'XOM', 'COP', 'EMR','LOW', 'MCD']]

df_returns7 = close2.pct_change()
df_returns7 = df_returns7.dropna()
df_returns7.columns = pd.MultiIndex.from_product([['returns'], df_returns7.columns.get_level_values(1)])

torch.manual_seed(4)
np.random.seed(5)
n_timesteps, n_assets = 1100, 19
lookback, gap, horizon = 40, 2, 20
n_samples = n_timesteps - lookback - horizon - gap + 1
split_ix = int(n_samples * 0.8)
indices_train = list(range(split_ix))
indices_test = list(range(split_ix + lookback + horizon, n_samples))

X_list, y_list = [], []

for i in range(lookback, n_timesteps - horizon - gap + 1):
    X_list.append(df_returns7.iloc[i - lookback:i, :])
    y_list.append(df_returns7.iloc[i + gap:i + gap + horizon, :])

X = np.stack(X_list, axis=0)[:, None, ...]
y = np.stack(y_list, axis=0)[:, None, ...]
means, stds = prepare_standard_scaler(X, indices=indices_train)
dataset = InRAMDataset(X, y, transform=Scale(means, stds))
dataloader_train = RigidDataLoader(dataset, indices=indices_train, batch_size=16)
dataloader_test = RigidDataLoader(dataset, indices=indices_test, batch_size=16)

class GreatNet(torch.nn.Module, Benchmark):
    def __init__(self, n_assets, lookback, p=0.5):
        super().__init__()

        n_features = n_assets * lookback

        self.dropout_layer = torch.nn.Dropout(p=p)
        self.dense_layer = torch.nn.Linear(n_features, n_assets, bias=True)
        self.allocate_layer = SoftmaxAllocator(temperature=None)
        self.temperature = torch.nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Of shape (n_samples, 1, lookback, n_assets).

        Returns
        -------
        weights : torch.Torch
            Tensor of shape (n_samples, n_assets).

        """
        n_samples, _, _, _ = x.shape
        x = x.view(n_samples, -1)  # flatten features
        x = self.dropout_layer(x)
        x = self.dense_layer(x)

        temperatures = torch.ones(n_samples).to(device=x.device, dtype=x.dtype) * self.temperature
        weights = self.allocate_layer(x, temperatures)

        return weights

network = GreatNet(n_assets, lookback)
network = network.train()
loss = MaximumDrawdown() + 2 * MeanReturns() + SharpeRatio()
run = Run(network,
          loss,
          dataloader_train,
          val_dataloaders={'test': dataloader_test},
          optimizer=torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.2),
          callbacks=[EarlyStoppingCallback(metric_name='loss',
                                           dataloader_name='test',
                                           patience=20)])
history = run.launch(100)
weight_table7 = generate_weights_table(network, dataloader_test)



rsi2 = port2_rsi[['AAPL', 'ADBE','ABT', 'BMY', 'COF', 'GS', 'SPG', 'AMT', 'PM', 'KHC', 'DUK', 'NEE', 'NFLX', 'T', 'XOM', 'COP', 'EMR','LOW', 'MCD']]

df_returns8 = rsi2.pct_change()
df_returns8 = df_returns8.dropna()
df_returns8.columns = pd.MultiIndex.from_product([['returns'], df_returns8.columns.get_level_values(1)])

torch.manual_seed(4)
np.random.seed(5)
n_timesteps, n_assets = 1100, 19
lookback, gap, horizon = 40, 2, 20
n_samples = n_timesteps - lookback - horizon - gap + 1
split_ix = int(n_samples * 0.8)
indices_train = list(range(split_ix))
indices_test = list(range(split_ix + lookback + horizon, n_samples))

X_list, y_list = [], []

for i in range(lookback, n_timesteps - horizon - gap + 1):
    X_list.append(df_returns8.iloc[i - lookback:i, :])
    y_list.append(df_returns8.iloc[i + gap:i + gap + horizon, :])

X = np.stack(X_list, axis=0)[:, None, ...]
y = np.stack(y_list, axis=0)[:, None, ...]
means, stds = prepare_standard_scaler(X, indices=indices_train)
dataset = InRAMDataset(X, y, transform=Scale(means, stds))
dataloader_train = RigidDataLoader(dataset, indices=indices_train, batch_size=16)
dataloader_test = RigidDataLoader(dataset, indices=indices_test, batch_size=16)

network = GreatNet(n_assets, lookback)
network = network.train()
loss = MaximumDrawdown() + 2 * MeanReturns() + SharpeRatio()
run = Run(network,
          loss,
          dataloader_train,
          val_dataloaders={'test': dataloader_test},
          optimizer=torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.2),
          callbacks=[EarlyStoppingCallback(metric_name='loss',
                                           dataloader_name='test',
                                           patience=20)])
history = run.launch(100)
weight_table8 = generate_weights_table(network, dataloader_test)


high2 = port2_high[['AAPL', 'ADBE','ABT', 'BMY', 'COF', 'GS', 'SPG', 'AMT', 'PM', 'KHC', 'DUK', 'NEE', 'NFLX', 'T', 'XOM', 'COP', 'EMR','LOW', 'MCD']]

df_returns9 = high2.pct_change()
df_returns9 = df_returns9.dropna()
df_returns9.columns = pd.MultiIndex.from_product([['returns'], df_returns9.columns.get_level_values(1)])

torch.manual_seed(4)
np.random.seed(5)
n_timesteps, n_assets = 1100, 19
lookback, gap, horizon = 40, 2, 20
n_samples = n_timesteps - lookback - horizon - gap + 1
split_ix = int(n_samples * 0.8)
indices_train = list(range(split_ix))
indices_test = list(range(split_ix + lookback + horizon, n_samples))

X_list, y_list = [], []

for i in range(lookback, n_timesteps - horizon - gap + 1):
    X_list.append(df_returns9.iloc[i - lookback:i, :])
    y_list.append(df_returns9.iloc[i + gap:i + gap + horizon, :])

X = np.stack(X_list, axis=0)[:, None, ...]
y = np.stack(y_list, axis=0)[:, None, ...]
means, stds = prepare_standard_scaler(X, indices=indices_train)
dataset = InRAMDataset(X, y, transform=Scale(means, stds))
dataloader_train = RigidDataLoader(dataset, indices=indices_train, batch_size=16)
dataloader_test = RigidDataLoader(dataset, indices=indices_test, batch_size=16)

network = GreatNet(n_assets, lookback)
network = network.train()
loss = MaximumDrawdown() + 2 * MeanReturns() + SharpeRatio()
run = Run(network,
          loss,
          dataloader_train,
          val_dataloaders={'test': dataloader_test},
          optimizer=torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.2),
          callbacks=[EarlyStoppingCallback(metric_name='loss',
                                           dataloader_name='test',
                                           patience=20)])
history = run.launch(100)
weight_table9 = generate_weights_table(network, dataloader_test)


sma25_2 = port2_sma25[['AAPL', 'ADBE','ABT', 'BMY', 'COF', 'GS', 'SPG', 'AMT', 'PM', 'KHC', 'DUK', 'NEE', 'NFLX', 'T', 'XOM', 'COP', 'EMR','LOW', 'MCD']]

df_returns10 = sma25_2.pct_change()
df_returns10 = df_returns10.dropna()
df_returns10.columns = pd.MultiIndex.from_product([['returns'], df_returns10.columns.get_level_values(1)])

torch.manual_seed(4)
np.random.seed(5)
n_timesteps, n_assets = 1100, 19
lookback, gap, horizon = 40, 2, 20
n_samples = n_timesteps - lookback - horizon - gap + 1
split_ix = int(n_samples * 0.8)
indices_train = list(range(split_ix))
indices_test = list(range(split_ix + lookback + horizon, n_samples))

X_list, y_list = [], []

for i in range(lookback, n_timesteps - horizon - gap + 1):
    X_list.append(df_returns10.iloc[i - lookback:i, :])
    y_list.append(df_returns10.iloc[i + gap:i + gap + horizon, :])

X = np.stack(X_list, axis=0)[:, None, ...]
y = np.stack(y_list, axis=0)[:, None, ...]
means, stds = prepare_standard_scaler(X, indices=indices_train)
dataset = InRAMDataset(X, y, transform=Scale(means, stds))
dataloader_train = RigidDataLoader(dataset, indices=indices_train, batch_size=16)
dataloader_test = RigidDataLoader(dataset, indices=indices_test, batch_size=16)

network = GreatNet(n_assets, lookback)
network = network.train()
loss = MaximumDrawdown() + 2 * MeanReturns() + SharpeRatio()
run = Run(network,
          loss,
          dataloader_train,
          val_dataloaders={'test': dataloader_test},
          optimizer=torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.2),
          callbacks=[EarlyStoppingCallback(metric_name='loss',
                                           dataloader_name='test',
                                           patience=20)])
history = run.launch(100)
weight_table10 = generate_weights_table(network, dataloader_test)



sma200_2 = port2_sma200[['AAPL', 'ADBE','ABT', 'BMY', 'COF', 'GS', 'SPG', 'AMT', 'PM', 'KHC', 'DUK', 'NEE', 'NFLX', 'T', 'XOM', 'COP', 'EMR','LOW', 'MCD']]

df_returns11 = sma200_2.pct_change()
df_returns11 = df_returns11.dropna()
df_returns11.columns = pd.MultiIndex.from_product([['returns'], df_returns11.columns.get_level_values(1)])

torch.manual_seed(4)
np.random.seed(5)
n_timesteps, n_assets = 1100, 19
lookback, gap, horizon = 40, 2, 20
n_samples = n_timesteps - lookback - horizon - gap + 1
split_ix = int(n_samples * 0.8)
indices_train = list(range(split_ix))
indices_test = list(range(split_ix + lookback + horizon, n_samples))

X_list, y_list = [], []

for i in range(lookback, n_timesteps - horizon - gap + 1):
    X_list.append(df_returns11.iloc[i - lookback:i, :])
    y_list.append(df_returns11.iloc[i + gap:i + gap + horizon, :])

X = np.stack(X_list, axis=0)[:, None, ...]
y = np.stack(y_list, axis=0)[:, None, ...]
means, stds = prepare_standard_scaler(X, indices=indices_train)
dataset = InRAMDataset(X, y, transform=Scale(means, stds))
dataloader_train = RigidDataLoader(dataset, indices=indices_train, batch_size=16)
dataloader_test = RigidDataLoader(dataset, indices=indices_test, batch_size=16)

network = GreatNet(n_assets, lookback)
network = network.train()
loss = MaximumDrawdown() + 2 * MeanReturns() + SharpeRatio()
run = Run(network,
          loss,
          dataloader_train,
          val_dataloaders={'test': dataloader_test},
          optimizer=torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.2),
          callbacks=[EarlyStoppingCallback(metric_name='loss',
                                           dataloader_name='test',
                                           patience=20)])
history = run.launch(100)
weight_table11 = generate_weights_table(network, dataloader_test)


volume_2 = port2_volume[['AAPL', 'ADBE','ABT', 'BMY', 'COF', 'GS', 'SPG', 'AMT', 'PM', 'KHC', 'DUK', 'NEE', 'NFLX', 'T', 'XOM', 'COP', 'EMR','LOW', 'MCD']]

df_returns12 = volume_2.pct_change()
df_returns12 = df_returns12.dropna()
df_returns12.columns = pd.MultiIndex.from_product([['returns'], df_returns12.columns.get_level_values(1)])

torch.manual_seed(4)
np.random.seed(5)
n_timesteps, n_assets = 1100, 19
lookback, gap, horizon = 40, 2, 20
n_samples = n_timesteps - lookback - horizon - gap + 1
split_ix = int(n_samples * 0.8)
indices_train = list(range(split_ix))
indices_test = list(range(split_ix + lookback + horizon, n_samples))

X_list, y_list = [], []

for i in range(lookback, n_timesteps - horizon - gap + 1):
    X_list.append(df_returns12.iloc[i - lookback:i, :])
    y_list.append(df_returns12.iloc[i + gap:i + gap + horizon, :])

X = np.stack(X_list, axis=0)[:, None, ...]
y = np.stack(y_list, axis=0)[:, None, ...]
means, stds = prepare_standard_scaler(X, indices=indices_train)
dataset = InRAMDataset(X, y, transform=Scale(means, stds))
dataloader_train = RigidDataLoader(dataset, indices=indices_train, batch_size=16)
dataloader_test = RigidDataLoader(dataset, indices=indices_test, batch_size=16)

network = GreatNet(n_assets, lookback)
network = network.train()
loss = MaximumDrawdown() + 2 * MeanReturns() + SharpeRatio()
run = Run(network,
          loss,
          dataloader_train,
          val_dataloaders={'test': dataloader_test},
          optimizer=torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.2),
          callbacks=[EarlyStoppingCallback(metric_name='loss',
                                           dataloader_name='test',
                                           patience=20)])
history = run.launch(100)
weight_table12 = generate_weights_table(network, dataloader_test)


weight_table_final2 = (weight_table7 + weight_table8 + weight_table9 + weight_table10 + weight_table11 + weight_table12)/6
print(weight_table_final2)
