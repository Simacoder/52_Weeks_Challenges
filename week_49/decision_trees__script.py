__generated_with = "0.14.16"

# %%
import marimo as mo
import time
import timeit
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ShuffleSplit, RepeatedKFold, cross_validate
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)
from skopt import BayesSearchCV
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import networkx as nx

# =============================================================================
# LOAD LOCAL DATASET (CSV)
# =============================================================================

california_housing_df = pd.read_csv("california_housing_alternative.csv")

target = "MedHouseVal"

chosen = pd.DataFrame(california_housing_df.iloc[-1]).T.reset_index(drop=True)
california_housing_df = california_housing_df[:-1]

print(california_housing_df.columns.tolist())


# %%
california_housing_df.iloc[:5]

# %%
chosen

california_housing_df.rename(columns={
    'median_income': 'MedInc',
    'housing_median_age': 'HouseAge',
    'total_rooms': 'AveRooms',         # You may want to divide by households later
    'total_bedrooms': 'AveBedrms',     # Same here
    'population': 'Population',
    'households': 'AveOccup',          # avg occupancy = population / households
    'latitude': 'Latitude',
    'longitude': 'Longitude',
    'MedHouseVal': 'MedHouseVal'
}, inplace=True)


# %%
X = california_housing_df[
    [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]
]
y = california_housing_df[target]

# %%
y.describe()

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# %%
dt_shallow = DecisionTreeRegressor(max_depth=3, random_state=42)
dt_deep = DecisionTreeRegressor(max_depth=None, random_state=42)
dt_prune = DecisionTreeRegressor(max_depth=None, ccp_alpha=0.005, random_state=42)

# %%
def fitpred_single(dt, timing_count=1):
    fit_time = timeit.timeit(lambda: dt.fit(X_train, y_train), number=timing_count)
    print(f"Time to fit:\t\t{fit_time/timing_count}")

    pred_time = timeit.timeit(lambda: dt.predict(X_test), number=timing_count)
    print(f"Time to predict:\t{pred_time/timing_count}")
    dt_predict = dt.predict(X_test)

    print()
    print(f"MAE:\t{mean_absolute_error(y_test, dt_predict)}")
    print(f"MAPE:\t{mean_absolute_percentage_error(y_test, dt_predict)}")
    print(f"MSE:\t{mean_squared_error(y_test, dt_predict)}")
    print(f"RMSE:\t{root_mean_squared_error(y_test, dt_predict)}")
    print(f"R2:\t\t{r2_score(y_test, dt_predict)}")
    print()

    print(f"Chosen prediction:\t{float(dt.predict(chosen.drop(columns=target))[0])}")
    print(f"Chosen actual:\t\t{chosen[target].iloc[0]}")

    plt.figure(figsize=(12, 4))
    plot_tree(
        dt,
        feature_names=X.columns,
        label="all",
        filled=True,
        impurity=True,
        proportion=True,
        rounded=True,
    )
    plt.show()

# %%
fitpred_single(dt_shallow)
fitpred_single(dt_deep)
fitpred_single(dt_prune)

# %%
def fitpred_loop(dt, loops=1000):
    _start = time.time()

    maes, mapes, mses, rmses, r2s = [], [], [], [], []

    for _ in mo.status.progress_bar(range(loops)):
        X_train_loop, X_test_loop, y_train_loop, y_test_loop = train_test_split(X, y)
        dt.fit(X_train_loop, y_train_loop)
        _pred = dt.predict(X_test_loop)

        maes.append(mean_absolute_error(y_test_loop, _pred))
        mapes.append(mean_absolute_percentage_error(y_test_loop, _pred))
        mses.append(mean_squared_error(y_test_loop, _pred))
        rmses.append(root_mean_squared_error(y_test_loop, _pred))
        r2s.append(r2_score(y_test_loop, _pred))

    return maes, mapes, mses, rmses, r2s, time.time() - _start

# %%
def print_info(dt_suffix):
    dt = globals()[f"dt_{dt_suffix}"]
    print(f"Depth:\t\t{dt.get_depth()}")
    print(f"Leaves:\t\t{dt.get_n_leaves()}")
    print(f"Nodes:\t\t{dt.tree_.node_count}")
    print()

    for metric in ["maes", "mapes", "mses", "rmses", "r2s"]:
        values = globals()[f"{metric}_{dt_suffix}"]
        print(f"{metric.upper()} mean:\t{np.mean(values):.3f}, std: {np.std(values):.3f}")

    print(f"\nTime:\t\t{globals()[f'time_{dt_suffix}']:.1f}s")

# %%
maes_shallow, mapes_shallow, mses_shallow, rmses_shallow, r2s_shallow, time_shallow = fitpred_loop(dt_shallow)
print_info("shallow")

# %%
maes_deep, mapes_deep, mses_deep, rmses_deep, r2s_deep, time_deep = fitpred_loop(dt_deep)
print_info("deep")

# %%
maes_prune, mapes_prune, mses_prune, rmses_prune, r2s_prune, time_prune = fitpred_loop(dt_prune)
print_info("prune")

# %%
search_space = {
    "max_depth": (1, 100),
    "min_samples_split": (2, 100),
    "min_samples_leaf": (1, 100),
    "max_features": (0.1, 1.0, "uniform"),
    "criterion": ["squared_error", "friedman_mse", "absolute_error"],
    "ccp_alpha": (0.0, 1.0, "uniform"),
}

opt = BayesSearchCV(
    DecisionTreeRegressor(random_state=42),
    search_spaces=search_space,
    n_iter=250,
    cv=5,
    n_jobs=-1,
    scoring="neg_mean_absolute_error",
    random_state=42,
)

opt.fit(X_train, y_train)

print("Best Parameters:", opt.best_params_)
print("Best MAE:", -opt.best_score_)

# %%
dt_bscv = opt.best_estimator_
dt_bscv.fit(X_train, y_train)

# %%
dt_bscv.predict(chosen.drop(columns=target))

# %%
maes_bscv, mapes_bscv, mses_bscv, rmses_bscv, r2s_bscv, time_bscv = fitpred_loop(dt_bscv)
print_info("bscv")

# %%
scoring = [
    "neg_mean_absolute_error",
    "neg_mean_absolute_percentage_error",
    "neg_mean_squared_error",
    "r2",
]

_cv = ShuffleSplit(n_splits=1000, test_size=0.2, random_state=42)
sscv_results = cross_validate(dt_bscv, X, y, cv=_cv, scoring=scoring)

_cv = RepeatedKFold(n_splits=5, n_repeats=200, random_state=42)
rkf_results = cross_validate(dt_bscv, X, y, cv=_cv, scoring=scoring)
__generated_with = "0.14.16"

# %%
import marimo as mo
import time
import timeit
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ShuffleSplit, RepeatedKFold, cross_validate
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)
from skopt import BayesSearchCV
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import networkx as nx

# =============================================================================
# LOAD LOCAL DATASET (CSV)
# =============================================================================

california_housing_df = pd.read_csv("california_housing_alternative.csv")

target = "MedHouseVal"

chosen = pd.DataFrame(california_housing_df.iloc[-1]).T.reset_index(drop=True)
california_housing_df = california_housing_df[:-1]

# %%
california_housing_df.iloc[:5]

# %%
chosen

# %%
X = california_housing_df[
    [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]
]
y = california_housing_df[target]

# %%
y.describe()

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# %%
dt_shallow = DecisionTreeRegressor(max_depth=3, random_state=42)
dt_deep = DecisionTreeRegressor(max_depth=None, random_state=42)
dt_prune = DecisionTreeRegressor(max_depth=None, ccp_alpha=0.005, random_state=42)

# %%
def fitpred_single(dt, timing_count=1):
    fit_time = timeit.timeit(lambda: dt.fit(X_train, y_train), number=timing_count)
    print(f"Time to fit:\t\t{fit_time/timing_count}")

    pred_time = timeit.timeit(lambda: dt.predict(X_test), number=timing_count)
    print(f"Time to predict:\t{pred_time/timing_count}")
    dt_predict = dt.predict(X_test)

    print()
    print(f"MAE:\t{mean_absolute_error(y_test, dt_predict)}")
    print(f"MAPE:\t{mean_absolute_percentage_error(y_test, dt_predict)}")
    print(f"MSE:\t{mean_squared_error(y_test, dt_predict)}")
    print(f"RMSE:\t{root_mean_squared_error(y_test, dt_predict)}")
    print(f"R2:\t\t{r2_score(y_test, dt_predict)}")
    print()

    print(f"Chosen prediction:\t{float(dt.predict(chosen.drop(columns=target))[0])}")
    print(f"Chosen actual:\t\t{chosen[target].iloc[0]}")

    plt.figure(figsize=(12, 4))
    plot_tree(
        dt,
        feature_names=X.columns,
        label="all",
        filled=True,
        impurity=True,
        proportion=True,
        rounded=True,
    )
    plt.show()

# %%
fitpred_single(dt_shallow)
fitpred_single(dt_deep)
fitpred_single(dt_prune)

# %%
def fitpred_loop(dt, loops=1000):
    _start = time.time()

    maes, mapes, mses, rmses, r2s = [], [], [], [], []

    for _ in mo.status.progress_bar(range(loops)):
        X_train_loop, X_test_loop, y_train_loop, y_test_loop = train_test_split(X, y)
        dt.fit(X_train_loop, y_train_loop)
        _pred = dt.predict(X_test_loop)

        maes.append(mean_absolute_error(y_test_loop, _pred))
        mapes.append(mean_absolute_percentage_error(y_test_loop, _pred))
        mses.append(mean_squared_error(y_test_loop, _pred))
        rmses.append(root_mean_squared_error(y_test_loop, _pred))
        r2s.append(r2_score(y_test_loop, _pred))

    return maes, mapes, mses, rmses, r2s, time.time() - _start

# %%
def print_info(dt_suffix):
    dt = globals()[f"dt_{dt_suffix}"]
    print(f"Depth:\t\t{dt.get_depth()}")
    print(f"Leaves:\t\t{dt.get_n_leaves()}")
    print(f"Nodes:\t\t{dt.tree_.node_count}")
    print()

    for metric in ["maes", "mapes", "mses", "rmses", "r2s"]:
        values = globals()[f"{metric}_{dt_suffix}"]
        print(f"{metric.upper()} mean:\t{np.mean(values):.3f}, std: {np.std(values):.3f}")

    print(f"\nTime:\t\t{globals()[f'time_{dt_suffix}']:.1f}s")

# %%
maes_shallow, mapes_shallow, mses_shallow, rmses_shallow, r2s_shallow, time_shallow = fitpred_loop(dt_shallow)
print_info("shallow")

# %%
maes_deep, mapes_deep, mses_deep, rmses_deep, r2s_deep, time_deep = fitpred_loop(dt_deep)
print_info("deep")

# %%
maes_prune, mapes_prune, mses_prune, rmses_prune, r2s_prune, time_prune = fitpred_loop(dt_prune)
print_info("prune")

# %%
search_space = {
    "max_depth": (1, 100),
    "min_samples_split": (2, 100),
    "min_samples_leaf": (1, 100),
    "max_features": (0.1, 1.0, "uniform"),
    "criterion": ["squared_error", "friedman_mse", "absolute_error"],
    "ccp_alpha": (0.0, 1.0, "uniform"),
}

opt = BayesSearchCV(
    DecisionTreeRegressor(random_state=42),
    search_spaces=search_space,
    n_iter=250,
    cv=5,
    n_jobs=-1,
    scoring="neg_mean_absolute_error",
    random_state=42,
)

opt.fit(X_train, y_train)

print("Best Parameters:", opt.best_params_)
print("Best MAE:", -opt.best_score_)

# %%
dt_bscv = opt.best_estimator_
dt_bscv.fit(X_train, y_train)

# %%
dt_bscv.predict(chosen.drop(columns=target))

# %%
maes_bscv, mapes_bscv, mses_bscv, rmses_bscv, r2s_bscv, time_bscv = fitpred_loop(dt_bscv)
print_info("bscv")

# %%
scoring = [
    "neg_mean_absolute_error",
    "neg_mean_absolute_percentage_error",
    "neg_mean_squared_error",
    "r2",
]

_cv = ShuffleSplit(n_splits=1000, test_size=0.2, random_state=42)
sscv_results = cross_validate(dt_bscv, X, y, cv=_cv, scoring=scoring)

_cv = RepeatedKFold(n_splits=5, n_repeats=200, random_state=42)
rkf_results = cross_validate(dt_bscv, X, y, cv=_cv, scoring=scoring)
