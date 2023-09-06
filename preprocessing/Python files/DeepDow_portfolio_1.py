from preprocessing.Preprocessing import ret_df
import pandas as pd
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

#### PORTFOLIO 1

#close
idx = pd.IndexSlice
port1_close = raw_df.loc[idx[:], idx[:,'close']]

#rsi
idx = pd.IndexSlice
port1_rsi= raw_df.loc[idx[:], idx[:,'rsi']]

#high
idx = pd.IndexSlice
port1_high = raw_df.loc[idx[:], idx[:,'high']]

#sma25
idx = pd.IndexSlice
port1_sma25 = raw_df.loc[idx[:], idx[:,'sma25']]

#sma200
idx = pd.IndexSlice
port1_sma200 = raw_df.loc[idx[:], idx[:,'sma200']]

#volume
idx = pd.IndexSlice
port1_volume = raw_df.loc[idx[:], idx[:,'volume']]

close = port1_close[['V', 'TXN','QCOM', 'PYPL', 'ORCL', 'NVDA', 'MSFT', 'MA', 'INTC', 'IBM', 'CSCO', 'CRM', 'AVGO', 'AMD', 'ADBE', 'ACN', 'AAPL']]


#Computing returns
df_returns1 = close.pct_change()

df_returns1 = df_returns1.dropna()

df_returns1.columns = pd.MultiIndex.from_product([['returns'], df_returns1.columns.get_level_values(1)])


#set seed for reproducibility4
torch.manual_seed(4)
np.random.seed(5)

n_timesteps, n_assets = 1100, 17
lookback, gap, horizon = 40, 2, 20
n_samples = n_timesteps - lookback - horizon - gap + 1

#split
split_ix = int(n_samples * 0.8)
indices_train = list(range(split_ix))
indices_test = list(range(split_ix + lookback + horizon, n_samples))

# Feature matrix X and target y
X_list, y_list = [], []

for i in range(lookback, n_timesteps - horizon - gap + 1):
    X_list.append(df_returns1.iloc[i - lookback:i, :])
    y_list.append(df_returns1.iloc[i + gap:i + gap + horizon, :])

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
                                           patience=10)])
history = run.launch(100)

weight_table = generate_weights_table(network, dataloader_test)


rsi = port1_rsi[['V', 'TXN','QCOM', 'PYPL', 'ORCL', 'NVDA', 'MSFT', 'MA', 'INTC', 'IBM', 'CSCO', 'CRM', 'AVGO', 'AMD', 'ADBE', 'ACN', 'AAPL']]

df_returns2 = rsi.pct_change()

# Remove the first row since it will have NaN values after computing returns
df_returns2 = df_returns2.dropna()

# Optional: If you want to rename the top-level column index from 'close' to 'returns':
df_returns2.columns = pd.MultiIndex.from_product([['returns'], df_returns1.columns.get_level_values(1)])

torch.manual_seed(4)
np.random.seed(5)
n_timesteps, n_assets = 1100, 17
lookback, gap, horizon = 40, 2, 20
n_samples = n_timesteps - lookback - horizon - gap + 1
split_ix = int(n_samples * 0.8)
indices_train = list(range(split_ix))
indices_test = list(range(split_ix + lookback + horizon, n_samples))


X_list, y_list = [], []

for i in range(lookback, n_timesteps - horizon - gap + 1):
    X_list.append(df_returns2.iloc[i - lookback:i, :])
    y_list.append(df_returns2.iloc[i + gap:i + gap + horizon, :])

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
                                           patience=10)])
history = run.launch(100)
weight_table2 = generate_weights_table(network, dataloader_test)

high = port1_high[['V', 'TXN','QCOM', 'PYPL', 'ORCL', 'NVDA', 'MSFT', 'MA', 'INTC', 'IBM', 'CSCO', 'CRM', 'AVGO', 'AMD', 'ADBE', 'ACN', 'AAPL']]

df_returns3 = high.pct_change()
df_returns3 = df_returns3.dropna()
df_returns3.columns = pd.MultiIndex.from_product([['returns'], df_returns3.columns.get_level_values(1)])

torch.manual_seed(4)
np.random.seed(5)
n_timesteps, n_assets = 1100, 17
lookback, gap, horizon = 40, 2, 20
n_samples = n_timesteps - lookback - horizon - gap + 1
split_ix = int(n_samples * 0.8)
indices_train = list(range(split_ix))
indices_test = list(range(split_ix + lookback + horizon, n_samples))
X_list, y_list = [], []

for i in range(lookback, n_timesteps - horizon - gap + 1):
    X_list.append(df_returns3.iloc[i - lookback:i, :])
    y_list.append(df_returns3.iloc[i + gap:i + gap + horizon, :])

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
                                           patience=10)])
history = run.launch(100)
weight_table3 = generate_weights_table(network, dataloader_test)


sma25 = port1_sma25[['V', 'TXN','QCOM', 'PYPL', 'ORCL', 'NVDA', 'MSFT', 'MA', 'INTC', 'IBM', 'CSCO', 'CRM', 'AVGO', 'AMD', 'ADBE', 'ACN', 'AAPL']]

df_returns4 = sma25.pct_change()
df_returns4 = df_returns4.dropna()
df_returns4.columns = pd.MultiIndex.from_product([['returns'], df_returns4.columns.get_level_values(1)])

torch.manual_seed(4)
np.random.seed(5)
n_timesteps, n_assets = 1100, 17
lookback, gap, horizon = 40, 2, 20
n_samples = n_timesteps - lookback - horizon - gap + 1
split_ix = int(n_samples * 0.8)
indices_train = list(range(split_ix))
indices_test = list(range(split_ix + lookback + horizon, n_samples))

X_list, y_list = [], []

for i in range(lookback, n_timesteps - horizon - gap + 1):
    X_list.append(df_returns4.iloc[i - lookback:i, :])
    y_list.append(df_returns4.iloc[i + gap:i + gap + horizon, :])
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
                                           patience=10)])
history = run.launch(100)
weight_table4 = generate_weights_table(network, dataloader_test)


sma200 = port1_sma200[['V', 'TXN','QCOM', 'PYPL', 'ORCL', 'NVDA', 'MSFT', 'MA', 'INTC', 'IBM', 'CSCO', 'CRM', 'AVGO', 'AMD', 'ADBE', 'ACN', 'AAPL']]

df_returns5 = sma200.pct_change()
df_returns5 = df_returns5.dropna()
df_returns5.columns = pd.MultiIndex.from_product([['returns'], df_returns5.columns.get_level_values(1)])

torch.manual_seed(4)
np.random.seed(5)
n_timesteps, n_assets = 1100, 17
lookback, gap, horizon = 40, 2, 20
n_samples = n_timesteps - lookback - horizon - gap + 1
split_ix = int(n_samples * 0.8)
indices_train = list(range(split_ix))
indices_test = list(range(split_ix + lookback + horizon, n_samples))

X_list, y_list = [], []

for i in range(lookback, n_timesteps - horizon - gap + 1):
    X_list.append(df_returns5.iloc[i - lookback:i, :])
    y_list.append(df_returns5.iloc[i + gap:i + gap + horizon, :])

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
                                           patience=10)])
history = run.launch(100)
weight_table5 = generate_weights_table(network, dataloader_test)


volume = port1_volume[['V', 'TXN','QCOM', 'PYPL', 'ORCL', 'NVDA', 'MSFT', 'MA', 'INTC', 'IBM', 'CSCO', 'CRM', 'AVGO', 'AMD', 'ADBE', 'ACN', 'AAPL']]

df_returns6 = volume.pct_change()
df_returns6 = df_returns6.dropna()
df_returns6.columns = pd.MultiIndex.from_product([['returns'], df_returns6.columns.get_level_values(1)])

torch.manual_seed(4)
np.random.seed(5)
n_timesteps, n_assets = 1100, 17
lookback, gap, horizon = 40, 2, 20
n_samples = n_timesteps - lookback - horizon - gap + 1
split_ix = int(n_samples * 0.8)
indices_train = list(range(split_ix))
indices_test = list(range(split_ix + lookback + horizon, n_samples))

X_list, y_list = [], []

for i in range(lookback, n_timesteps - horizon - gap + 1):
    X_list.append(df_returns6.iloc[i - lookback:i, :])
    y_list.append(df_returns6.iloc[i + gap:i + gap + horizon, :])

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
weight_table6 = generate_weights_table(network, dataloader_test)

######## FINAL WEIGHTS PORTFOLIO 1 #########

weight_table_final = (weight_table + weight_table2 + weight_table3 + weight_table4 + weight_table5 + weight_table6)/6
pd.set_option('display.max_columns', None)
weights_DeepDow1 = (weight_table_final.iloc[0]*100)
df_weights = pd.DataFrame(weights_DeepDow1)

# Save to CSV
df_weights.to_csv("DeepDow_weights1.csv", index=False)
