import sys

sys.path.append("/home/wenhan/Dev/news")

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Dict, Callable, Optional, Literal, Tuple, Union
import cvxopt
from sklearn.metrics.pairwise import rbf_kernel
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from src.pipeline.features import (
    ALL_INSTRUMENT_FEATURES,
    ALL_ISSUER_FEATURES,
    ALL_INDUSTRY_FEATURES,
    ALL_MACRO_FEATURES,
)


# Source for implementation see:
# https://www.sciencedirect.com/science/article/abs/pii/S0377221714005463?fr=RR-2&ref=pdf_download&rr=8e406ab2fb29ce5b

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

train_features = pd.read_excel(
    os.path.join(BASE_PATH, "data/recovery_rate/train_features2.xlsx"), dtype=float
)
train_labels = pd.read_excel(
    os.path.join(BASE_PATH, "data/recovery_rate/train_labels2.xlsx")
)
test_features = pd.read_excel(
    os.path.join(BASE_PATH, "data/recovery_rate/test_features2.xlsx"), dtype=float
)
test_labels = pd.read_excel(
    os.path.join(BASE_PATH, "data/recovery_rate/test_labels2.xlsx")
)

INSTRUMENT_FEATURES = [
    "seniorioty_adj_Junior Unsecured or Junior Subordinated Unsecured",
    "seniorioty_adj_Secured",
    "seniorioty_adj_Senior Secured",
    "seniorioty_adj_Senior Subordinated Unsecured",
    "seniorioty_adj_Senior Unsecured",
    "seniorioty_adj_Subordinated Unsecured",
    "seniorioty_adj_Unsecured",
]  # missing tading volume and previous rating history

ISSUER_FEATURES = [
    "Industry_sector_Utilities",
    "Industry_group_Utilities",
]

INDUSTRY_FEATURES = []  # missing industry index performance and sale

MACRO_FEATURES = [
    "1-year_GDP_growth",
    "employment_rate",
    "interest_rate",
]  # missing number of defaulted bonds in the respective year and high-yield index

y = train_labels["rr1_30"].values

def preprocess_X(X: np.ndarray):
    # remove all constant features
    X_std = X.std(axis=0)
    idx = np.where(X_std !=0)[0]
    X = X[:, idx]
    X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)
    return X_normalized 

X0 = train_features[
    INSTRUMENT_FEATURES + ISSUER_FEATURES + INDUSTRY_FEATURES + MACRO_FEATURES
].values
X0_normalized = preprocess_X(X0)

X1 = train_features[ISSUER_FEATURES + INDUSTRY_FEATURES + MACRO_FEATURES].values
senority_classes = train_features[INSTRUMENT_FEATURES].values
X1_normalized = preprocess_X(X1)

X2 = train_features[
    ALL_INSTRUMENT_FEATURES
    + ALL_ISSUER_FEATURES
    + ALL_INDUSTRY_FEATURES
    + ALL_MACRO_FEATURES
].values
X2_normalized = preprocess_X(X2)

INSTRUMENT_FEATURES_EX_SENORITY = list(
    set(ALL_INSTRUMENT_FEATURES).difference(set(INSTRUMENT_FEATURES))
)
X3 = train_features[
    INSTRUMENT_FEATURES_EX_SENORITY
    + ALL_ISSUER_FEATURES
    + ALL_INDUSTRY_FEATURES
    + ALL_MACRO_FEATURES
].values
X3_normalized = preprocess_X(X3)

N_FOLDS = 5
KF = KFold(n_splits=N_FOLDS)


def rmse(y_pred, y_true):
    error = np.sqrt(np.mean((y_pred - y_true) ** 2))
    return error


def LS_SVR(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float,
) -> Dict[Literal["a", "b"], np.ndarray]:
    N = X_train.shape[0]
    e = np.ones((N, 1))
    r = y_train.reshape(N, 1)
    K = rbf_kernel(X_train, X_train)
    K_bar = K + 1 / C * np.eye(N)

    A = np.concatenate(
        (
            np.concatenate((np.array([[0]]), e.T), axis=1),
            np.concatenate((e, K_bar), axis=1),
        ),
        axis=0,
    )
    b = np.concatenate((np.array([[0]]), r), axis=0)

    solution = np.linalg.solve(A, b)

    intercept = solution[0]
    alphas = solution[1:]
    return {"b": intercept, "a": alphas}


def LS_SVR_predict(
    X_new: np.ndarray, X_train: np.ndarray, solution: Dict[str, np.ndarray]
) -> np.ndarray:
    alphas = solution["a"]
    intercept = solution["b"]
    K = rbf_kernel(X_new, X_train)
    y_pred = K.dot(alphas) + intercept
    return y_pred.flatten()


def LS_SVR_diff_intercept(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float,
    senority_classes: np.ndarray,
) -> Dict[Literal["a", "b"], np.ndarray]:
    N = X_train.shape[0]
    K = rbf_kernel(X_train, X_train)
    group_counts = senority_classes.sum(axis=0).astype(int)
    W = np.zeros((N, N))
    prev_k = 0
    for k in group_counts:
        W[prev_k : prev_k + k, prev_k : prev_k + k] = np.ones((k, k))
        prev_k = prev_k + k
    P = cvxopt.matrix(K + W + 1 / C * np.eye(N))
    q = cvxopt.matrix(-y_train.reshape(N, 1))
    solution = cvxopt.solvers.qp(P=P, q=q)
    alphas = np.array(solution["x"])
    group_intercepts = alphas.T.dot(senority_classes)
    return {"b": group_intercepts, "a": alphas}


def LS_SVR_diff_intercept_predict(
    X_new: np.ndarray,
    senority_classes: np.ndarray,
    X_train: np.ndarray,
    solution: Dict[str, np.ndarray],
) -> np.ndarray:
    alphas = solution["a"]
    group_intercepts = solution["b"]
    b_k = senority_classes.dot(group_intercepts.T)
    K = rbf_kernel(X_new, X_train)
    y_pred = K.dot(alphas) + b_k
    return y_pred.flatten()


def SP_LS_SVR(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float,
    senority_classes: np.ndarray,
) -> Dict[Literal["a", "b", "beta"], np.ndarray]:

    N = X_train.shape[0]
    z = senority_classes
    Z = z.dot(z.T)
    K = rbf_kernel(X_train, X_train)
    V = np.ones((N, N))
    P = cvxopt.matrix(K + Z + V + 1 / C * np.eye(N))
    q = cvxopt.matrix(-y_train.reshape(N, 1))
    solution = cvxopt.solvers.qp(P=P, q=q)
    alphas = np.array(solution["x"])
    b = alphas.sum()
    beta = alphas.T.dot(z)
    return {"b": b, "a": alphas, "beta": beta}


def SP_LS_SVR_predict(
    X_new: np.ndarray,
    senority_classes: np.ndarray,
    X_train: np.ndarray,
    solution: Dict[str, np.ndarray],
) -> np.ndarray:
    alphas = solution["a"]
    b = solution["b"]
    beta = solution["beta"]
    K = rbf_kernel(X_new, X_train)
    y_pred = K.dot(alphas) + senority_classes.dot(beta.T) + b
    return y_pred.flatten()


search_grid = np.arange(1.5, 2.5, 0.01)
errors = np.zeros((search_grid.shape[0], 5))
for i, C in tqdm(enumerate(search_grid), total=search_grid.shape[0]):
    fold_errors = []
    for j, (train_index, test_index) in enumerate(KF.split(X2)):
        X_train, X_val = X2[train_index], X2[test_index]
        y_train, y_val = y[train_index], y[test_index]

        solution = LS_SVR(X_train=X_train, y_train=y_train, C=C)
        error = rmse(
            LS_SVR_predict(X_new=X_val, X_train=X_train, solution=solution), y_val
        )
        errors[i, j] = error

errors_mean = errors.mean(axis=1)
errors_std = errors.std(axis=1)
best_idx = np.argmin(errors_mean)
best_C = search_grid[best_idx]
best_error = errors_mean[best_idx]
f, ax = plt.subplots(1, 1)
ax.plot(search_grid, errors_mean, label=f"Best C = {best_C.round(3)}")
# ax.fill_between(search_grid, errors_mean - errors_std, errors_mean + errors_std, color='blue', alpha=0.2, label='Std Dev Band')
ax.scatter(
    best_C,
    best_error,
    marker="*",
    color="red",
    label=f"best val loss = {best_error.round(3)}",
)
plt.legend()
plt.savefig(os.path.join(BASE_PATH, "data/images/LS_SVR_all.png"))
plt.show()
solution = LS_SVR(X_train=X2, y_train=y, C=1.87)
print(
    rmse(
        LS_SVR_predict(
            X_new=test_features[
                ALL_INSTRUMENT_FEATURES
                + ALL_ISSUER_FEATURES
                + ALL_INDUSTRY_FEATURES
                + ALL_MACRO_FEATURES
            ].values,
            X_train=X2,
            solution=solution,
        ),
        test_labels["rr1_30"].values,
    )
)


search_grid = np.arange(0.0001, 0.05, 0.001)
errors = np.zeros((search_grid.shape[0], 5))
for i, C in tqdm(enumerate(search_grid), total=search_grid.shape[0]):
    fold_errors = []
    for j, (train_index, test_index) in enumerate(KF.split(X1_normalized)):
        X_train, X_val = X1_normalized[train_index], X1_normalized[test_index]
        senority_classes_train, senority_classes_val = senority_classes[train_index], senority_classes[test_index]
        y_train, y_val = y[train_index], y[test_index]

        solution = LS_SVR_diff_intercept(X_train=X_train, y_train=y_train, C=C, senority_classes=senority_classes_train)
        error = rmse(
            LS_SVR_diff_intercept_predict(X_new=X_val, senority_classes=senority_classes_val, X_train=X_train, solution=solution), y_val
        )
        errors[i, j] = error

errors_mean = errors.mean(axis=1)
errors_std = errors.std(axis=1)
best_idx = np.argmin(errors_mean)
best_C = search_grid[best_idx]
best_error = errors_mean[best_idx]
f, ax = plt.subplots(1, 1)
ax.plot(search_grid, errors_mean, label=f"Best C = {best_C.round(3)}")
# ax.fill_between(search_grid, errors_mean - errors_std, errors_mean + errors_std, color='blue', alpha=0.2, label='Std Dev Band')
ax.scatter(
    best_C,
    best_error,
    marker="*",
    color="red",
    label=f"best val loss = {best_error.round(3)}",
)
plt.legend()
# plt.savefig(os.path.join(BASE_PATH, "data/images/LS_SVR_diff_intercepts_all.png"))
plt.show()

solution = LS_SVR_diff_intercept(
    X_train=X3, y_train=y, C=0.01, senority_classes=senority_classes
)
print(
    rmse(
        LS_SVR_diff_intercept_predict(
            X_new=test_features[INSTRUMENT_FEATURES_EX_SENORITY + ALL_ISSUER_FEATURES + ALL_INDUSTRY_FEATURES + ALL_MACRO_FEATURES].values,
            senority_classes=test_features[INSTRUMENT_FEATURES].values,
            X_train=X3,
            solution=solution,
        ),
        test_labels["rr1_30"].values,
    )
)

search_grid = np.arange(2.0, 3.0, 0.05)
errors = np.zeros((search_grid.shape[0], 5))
for i, C in tqdm(enumerate(search_grid), total=search_grid.shape[0]):
    fold_errors = []
    for j, (train_index, test_index) in enumerate(KF.split(X3)):
        X_train, X_val = X3[train_index], X3[test_index]
        senority_classes_train, senority_classes_val = senority_classes[train_index], senority_classes[test_index]
        y_train, y_val = y[train_index], y[test_index]

        solution = SP_LS_SVR(X_train=X_train, y_train=y_train, C=C, senority_classes=senority_classes_train)
        error = rmse(
            SP_LS_SVR_predict(X_new=X_val, senority_classes=senority_classes_val, X_train=X_train, solution=solution), y_val
        )
        errors[i, j] = error

errors_mean = errors.mean(axis=1)
errors_std = errors.std(axis=1)
best_idx = np.argmin(errors_mean)
best_C = search_grid[best_idx]
best_error = errors_mean[best_idx]
f, ax = plt.subplots(1, 1)
ax.plot(search_grid, errors_mean, label=f"Best C = {best_C.round(3)}")
# ax.fill_between(search_grid, errors_mean - errors_std, errors_mean + errors_std, color='blue', alpha=0.2, label='Std Dev Band')

ax.scatter(
    best_C,
    best_error,
    marker="*",
    color="red",
    label=f"best val loss = {best_error.round(3)}",
)
plt.legend()
plt.savefig(os.path.join(BASE_PATH, "data/images/SP_LS_SVR_all.png"))
plt.show()
solution = SP_LS_SVR(
    X_train=X3, y_train=y, C=2.45, senority_classes=senority_classes
)
print(
    rmse(
        SP_LS_SVR_predict(
            X_new=test_features[INSTRUMENT_FEATURES_EX_SENORITY + ALL_ISSUER_FEATURES + ALL_INDUSTRY_FEATURES + ALL_MACRO_FEATURES].values,
            senority_classes=test_features[INSTRUMENT_FEATURES].values,
            X_train=X3,
            solution=solution,
        ),
        test_labels["rr1_30"].values,
    )
)
