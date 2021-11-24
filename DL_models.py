# tensorlow
from tensorflow.keras.layers import (
    Input,
    Dense,
    SimpleRNN,
    GRU,
    LSTM,
    Flatten,
    Dropout,
    Conv1D,
    BatchNormalization,
    ReLU,
    Add,
    ZeroPadding1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import *

from typing import Dict, List, Tuple


def RNNmodel(
    n_hidden_neurons=64,
    cell_type="LSTM",
    act_fun="tanh",
    dropout_rate=0.2,
    T=168,
    D=2,
    return_sequences=False,
):
    i = Input(shape=(T, D), name="Input layer")
    print(i)
    if cell_type == "SimpleRNN":
        x = SimpleRNN(n_hidden_neurons, activation=act_fun, name="SimpleRNN")(i)
    elif cell_type == "LSTM":
        x = LSTM(
            n_hidden_neurons,
            activation=act_fun,
            return_sequences=return_sequences,
            recurrent_activation="sigmoid",
            name="LSTM",
        )(i)
        if return_sequences:
            x = LSTM(
                n_hidden_neurons,
                activation=act_fun,
                return_sequences=False,
                recurrent_activation="sigmoid",
            )(x)
    elif cell_type == "GRU":
        x = GRU(
            n_hidden_neurons,
            activation=act_fun,
            return_sequences=return_sequences,
            name="GRU",
        )(i)
        if return_sequences:
            x = GRU(n_hidden_neurons, activation=act_fun, return_sequences=False)(x)
    else:
        raise Exception(
            "Error: Cell type not recognized! Choose between 'SimpleRNN','LSTM', or 'GRU'."
        )
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation="relu", name="Hidden1")(x)
    x = Dense(24, activation="linear", name="Output")(x)
    model = Model(i, x)
    return model


def OneDCNN(
    number_blocks=5,
    number_filters=32,
    dropout_rate=0.2,
    filter_size_block=2,
    filter_size_branch=1,
    T=168,
    D=2,
):
    dilation_rates = (2 ** exp for exp in range(0, number_blocks))
    i = Input(shape=(T, D), name="Input layer")
    x01 = ZeroPadding1D(padding=(filter_size_block - 1, 0))(i)
    concatenated = x01
    for d in dilation_rates:
        x1 = Conv1D(filters=number_filters, kernel_size=(filter_size_branch))(
            concatenated
        )
        x2 = Conv1D(
            number_filters,
            kernel_size=(filter_size_block),
            padding="causal",
            dilation_rate=d,
        )(concatenated)
        x2 = BatchNormalization()(x2)
        x2 = ReLU()(x2)
        x2 = Dropout(rate=dropout_rate)(x2)
        concatenated = Add()([x2, x1])
    x = Flatten()(concatenated)
    x = Dense(units=24, activation="linear", name="Output")(x)
    model = Model(i, x)
    return model
