# imports

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
from deepdow.utils import raw_to_Xy


#Preprocessing pipeline that lead to our data

import os
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn import set_config; set_config(display='diagram')

key = os.environ.get('API_KEY')

# Step 1: Define a function to read CSV files and convert them to dataframes
# Get the current working directory (where your script is located)
current_directory = os.getcwd()

# Specify the path to the "raw_data" folder
raw_data_folder = os.path.join(current_directory, "raw_data")

# List all CSV files in the "raw_data" folder
csv_files = [os.path.join(raw_data_folder, file) for file in os.listdir(raw_data_folder) if file.endswith(".csv")]

# passing csv files in to a dataframe
dataframes = [pd.read_csv(file) for file in csv_files]



# Step 2: Define a function to read CSV files and convert them to dataframes
def read_csv_to_dataframe(file_path):
    df = pd.read_csv(file_path)
    return df

# Step 4: Read CSV files, preprocess data, and stack them into a 3D tensor
data = []
stock_names =[]

# print(type(csv_files))

for file in csv_files:
    # Read the CSV file into a DataFrame
    data_df = pd.read_csv(file)
    stock_name = file.split('/')[-1].split('.')[0]
    stock_names.append(stock_name)
    data_df["stock"] = stock_name
    data.append(data_df)

    impute_columns = ["sma25", "sma100", "sma200", "rsi", "macd", "signal", "histogram"]

imputer = Pipeline(
    [
        ('imputer', ColumnTransformer(
            transformers=[
                ('impute', KNNImputer(n_neighbors=10), impute_columns),  # Apply imputation to specific columns
            ],
            remainder='passthrough'  # Keep the remaining columns
        ))
    ]
)


def df_and_column_transform(arr):
    df = pd.DataFrame(arr, columns=["sma25", "sma100", "sma200", "rsi", "macd", "signal", "histogram",\
    "Unnamed: 0", "open", "high","low", "close", "volume", "vwap", "timestamp", "transactions", "otc", "stock"])

    first_col = df.pop("sma25")
    df.insert(16, "sma25", first_col)

    sec_col = df.pop("sma100")
    df.insert(16, "sma100", sec_col)

    third_col = df.pop("sma200")
    df.insert(16, "sma200", third_col)

    fourth_col = df.pop("rsi")
    df.insert(16, "rsi", fourth_col)

    fifth_col = df.pop("macd")
    df.insert(16, "macd", fifth_col)

    sixth_col = df.pop("signal")
    df.insert(16, "signal", sixth_col)

    seventh_col = df.pop("histogram")
    df.insert(16, "histogram", seventh_col)

    return df

transform_pipe = make_pipeline(
    FunctionTransformer(df_and_column_transform)
)

preprocessor = preprocessor = Pipeline(
    [
        ("transformation", transform_pipe),
    ],
)
preprocessor


def timestamp_transform(df):
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
    first_col = df.pop("date")
    df.insert(0, 'date', first_col)
    # df.set_index(keys='date', inplace=True)

    return df

time_pipe = make_pipeline(
    FunctionTransformer(timestamp_transform)
)

preprocessor = Pipeline(
    [
        ("transformation", transform_pipe),
        ("timestamp_convertor", time_pipe),
    ],
)
preprocessor

drop_columns = ["Unnamed: 0", 'timestamp', "transactions", "otc"]

def drop(df):
    unwanted_columns = drop_columns
    df = df.drop(columns=unwanted_columns)

    return df

drop_pipe = make_pipeline(
    FunctionTransformer(drop)
)

preprocessor = Pipeline(
    [
        ("transformation", transform_pipe),
        ("timestamp_convertor", time_pipe),
        ("unwanted_columns", drop_pipe),
    ],
)
sec_pipe = Pipeline(
    [
        ("imputer", imputer),
        ("preprocessor", preprocessor),
    ]
)
def final_transformation(arr):
    cols = ["open", "high", "low", "close", "volume", "vwap", "sma25", "sma100", "sma200", "rsi", "macd",\
        "signal", "histogram", "date", "stock"]
    df = pd.DataFrame(arr, columns=cols)

    first_col = df.pop("date")
    df.insert(0, "date", first_col)

    df.set_index(keys='date', inplace=True)

    return df

final_transformation_pipe = make_pipeline(
    FunctionTransformer(final_transformation)
)

final_processing = Pipeline(
    [
        ("final_transformation", final_transformation_pipe)
    ]
)
final_pipe = Pipeline(
    [
        ("imputer", imputer),
        ("preprocessor", preprocessor),
        ("transformation", final_processing)
    ]
)
dataframes = []

for df in data:
    preprocessed_df = final_pipe.fit_transform(df)

    dataframes.append(preprocessed_df)
    filtered_dataframes = []

for df in dataframes:
    if df.shape == (1257, 14):
        filtered_dataframes.append(df)
        unequal_shape = []

for df in dataframes:
    if df.shape != (1257, 14):
        unequal_shape.append(df)
        # Check
num_dataframes = len(dataframes)


unequal = len(unequal_shape)


f = len(filtered_dataframes)


def transform_dataframe(df):
    return df.pivot(columns='stock').swaplevel(0, 1, axis=1).sort_index(axis=1)

# Apply the transformation to each data frame
transformed_dataframes = [transform_dataframe(df) for df in filtered_dataframes]

# Merge all transformed data frames into one
merged_df = pd.concat(transformed_dataframes, axis=1)

def ret_df():
    return merged_df
