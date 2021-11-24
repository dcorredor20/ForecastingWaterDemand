"""Functions for the steps involved in the data pre-processing."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Dict, List, Tuple


def create_sequences(
    series: np.array, lookback: int, horizon: int = 24, expand_dims: bool = True
) -> Tuple[np.array, np.array]:
    """
    Return X and Y sequences.

    Args:
        series: np.array with the data.
        lookback: number of timesteps.
        horizon: number of timesteps to predict.
        expand_dims: bool if reshape by default True
    Returns:
        X, y.

    """
    X = []
    Y = []
    for t in range(len(series) - lookback - horizon):
        x = series[t : t + lookback]
        X.append(x)
        y = series[t + lookback : t + lookback + horizon]
        Y.append(y)

    X = np.array(X)
    Y = np.array(Y)

    if expand_dims:
        X = np.expand_dims(X, axis=2)

    return X, Y


def build_dataset(
    water_demand: np.array, days: np.array, lookback: int, horizon: int = 24
) -> Tuple[np.array, np.array]:
    """
    Return X and Y sequences.

    Args:
        water_demand: np.array with the water demand data.
        days: np.array with the days data.
        lookback: number of timesteps.
        horizon: number of timesteps to predict.
        expand_dims: bool if reshape by default True
    Returns:
        X, y.

    """
    X1, y = create_sequences(water_demand, lookback, horizon, expand_dims=True)
    X2, _ = create_sequences(days, lookback, horizon, expand_dims=False)
    X = np.concatenate([X1, X2], axis=-1)
    return X, y


def scaler(X, Y, sc, variables=2, train=True):
    """
    Return X, Y sclaed and the parameters of the scaler sc.

    Args:
        X: np.array with sequences.
        Y: np.array with target sequences.
        variables: number of variables by default 2.
        train: bool that decides whether is training data.
    Returns:
        X, Y, sc.
    """
    X_shape = X.shape
    Y_shape = Y.shape
    X = X.reshape((-1, variables))
    Y = Y.reshape((-1, 1))
    X1, X2 = split_X(X)
    if train:
        X1 = sc.fit_transform(X1)
    else:
        X1 = sc.transform(X1)

    Y = sc.transform(Y).reshape(Y_shape)
    X = np.concatenate((X1, X2), axis=-1).reshape(X_shape)
    return X, Y, sc


def split_X(X: np.array) -> Tuple[np.array, np.array]:
    """
    Return X1, and X2.

    Args:
        X: np.array with sequences.
    Returns:
        X1, X2
    """
    X1 = np.array([i[0] for i in X])
    X1 = np.expand_dims(X1, axis=1)
    X2 = np.array([i[1] for i in X])
    X2 = np.expand_dims(X2, axis=1)
    return X1, X2


def data_preprocessing(X, Y, Xtest, Ytest, parameters: Dict, variables=2):
    """
    Return Xtra, Ytra, Xval, Yval, Xtest, Ytest, and sc.

    Args:
        X: np.array with sequences.
        Y: np.array with target sequences.
        Xtest: np.array with sequences.
        Ytest: np.array with target sequences
        parameters: Dict with defined parameters.
        variables: number of variables by default 2.
    Returns:
        Xtra, Ytra, Xval, Yval, Xtest, Ytest, sc
    """
    # Create train and validation sets
    Xtra, Xval, Ytra, Yval = train_test_split(
        X,
        Y,
        test_size=parameters["test_size"],
        shuffle=True,
        random_state=parameters["seed"],
    )

    assert parameters["type_scaler"] in [
        "MinMax",
        "Standard",
    ], "Scaler must be MinMax or Standard"
    # Scalers
    if parameters["type_scaler"] == "MinMax":
        sc = MinMaxScaler([0, 1])
    if parameters["type_scaler"] == "Standard":
        sc = StandardScaler()
    # For train
    Xtra, Ytra, sc = scaler(Xtra, Ytra, sc, variables=variables, train=True)

    # For validation
    Xval, Yval, _ = scaler(Xval, Yval, sc, variables=variables, train=False)

    # For test
    Xtest, Ytest, _ = scaler(Xtest, Ytest, sc, variables=variables, train=False)
    return Xtra, Ytra, Xval, Yval, Xtest, Ytest, sc