from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
import lightgbm as lgb
import matplotlib as plt
import pandas as pd
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    veri = pd.DataFrame({"Real": y_true, "Predicted": y_pred}).sample(n=100, random_state=42)
    percentage_error = np.abs(veri["Real"] - veri["Predicted"]) / veri["Real"] * 100
    mean_percentage_error = round(percentage_error.mean(), 4)
    metrics = {"MSE": round(mse, 4), "RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4), "Mean % Error": mean_percentage_error}
    return veri, metrics

def preprocess_data(X_train, X_test, scaling_method=None):
    if scaling_method == 'standardize':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test

def find_optimal_parameters(X, y, model_type):
    if model_type == "XGBoost":
        param_grid = {'learning_rate': [0.01, 0.1, 0.3], 'max_depth': [3, 5, 7], 'n_estimators': [50, 100, 200]}
        model = XGBRegressor()
    elif model_type == "LightGBM":
        param_grid = {'learning_rate': [0.01, 0.1, 0.3], 'max_depth': [3, 5, 7], 'n_estimators': [50, 100, 200]}
        model = lgb.LGBMRegressor(verbosity=-1)
    elif model_type == "DecisionTree":
        param_grid = {'max_depth': [5, 10, 15], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
        model = DecisionTreeRegressor()
    elif model_type == "RandomForest":
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, 15], 'min_samples_split': [2, 5, 10]}
        model = RandomForestRegressor()
    elif model_type == "LinearRegression":
        param_grid = {'fit_intercept': [True, False]}
        model = SklearnLinearRegression()
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X, y)
    return grid_search.best_params_

def XGBoost(df, l_rate, m_depth, n_est, böl, scaling_method=None):
    y = df["price"]
    X = df.drop("price", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=böl, random_state=1)
    X_train, X_test = preprocess_data(X_train, X_test, scaling_method)
    model = XGBRegressor(learning_rate=l_rate, max_depth=m_depth, n_estimators=n_est).fit(X_train, y_train)
    preds = model.predict(X_test)
    return calculate_metrics(y_test, preds)

def LightGBM(df, l_rate, m_depth, n_est, böl, scaling_method=None):
    y = df["price"]
    X = df.drop("price", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=böl, random_state=1)
    X_train, X_test = preprocess_data(X_train, X_test, scaling_method)
    model = lgb.LGBMRegressor(learning_rate=l_rate, max_depth=m_depth, n_estimators=n_est, verbosity=-1).fit(X_train, y_train)
    preds = model.predict(X_test)
    return calculate_metrics(y_test, preds)

def DecisionTree(df, max_depth, min_samples_split, min_samples_leaf, böl, scaling_method=None):
    y = df["price"]
    X = df.drop("price", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=böl, random_state=1)
    X_train, X_test = preprocess_data(X_train, X_test, scaling_method)
    model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf).fit(X_train, y_train)
    preds = model.predict(X_test)
    return calculate_metrics(y_test, preds)

def RandomForest(df, n_estimators, max_depth, min_samples_split, böl, scaling_method=None):
    y = df["price"]
    X = df.drop("price", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=böl, random_state=1)
    X_train, X_test = preprocess_data(X_train, X_test, scaling_method)
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split).fit(X_train, y_train)
    preds = model.predict(X_test)
    return calculate_metrics(y_test, preds)

def LinearRegressionModel(df, fit_intercept, normalize, böl, scaling_method=None):
    y = df["price"]
    X = df.drop("price", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=böl, random_state=1)
    X_train, X_test = preprocess_data(X_train, X_test, scaling_method)
    model = SklearnLinearRegression(fit_intercept=fit_intercept).fit(X_train, y_train)
    preds = model.predict(X_test)
    return calculate_metrics(y_test, preds)
