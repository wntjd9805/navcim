import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import random
import argparse
import joblib
import re
from sklearn.preprocessing import PolynomialFeatures
import pickle
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import os

parser = argparse.ArgumentParser(description='PyTorch CIFAR-X Example')
parser.add_argument('--metric',required=True, help='latency|energy')
parser.add_argument('--model',required=True, help='ResNet50,,,')
parser.add_argument('--train',required=True, help='mlp|poly')
parser.add_argument('--ADC', type=int, required=True, help='ADC')
parser.add_argument('--Cellbit', type=int, required=True, help='Cellbit')

args = parser.parse_args()

def extract_and_convert_to_float(s):
    match = re.search(r'\(?\d+\.\d*\)?', s)
    if match:
        return float(s.split("(")[1][:-1])
    else:
        return None

model_dir = f'./predict_model/{args.model}'
os.makedirs(model_dir, exist_ok=True)

data = pd.read_csv(f"./meta_learner_dataset/booksim+neurosim_scalefactor_{args.model}_{args.ADC}_{args.Cellbit}_with_power.csv")
data = data.sample(frac=1).reset_index(drop=True)

# str -> float으로 변환
data["booksim_energy"] = data["booksim_latency"] * data['booksim_power'] * 1000

data['pred_latency']=data['pred_latency'].apply(extract_and_convert_to_float)
data['pred_power']=data['pred_power'].apply(extract_and_convert_to_float)

data['pred_energy2'] = data['pred_power'] * data['booksim_latency'] * 1000

if args.metric == "latency":
    data['realadd_latency'] = data['booksim_latency'] + data['neurosim_latency']
    X = data[['pred_latency','neurosim_latency']].values
    y = data[['realadd_latency']].values
elif args.metric == "energy":
    data['pred_energy2'] = data['pred_power'] * data['booksim_latency'] * 1000
    data['realadd_energy'] = data['booksim_energy'] +  data['neurosim_energy']
    X = data[['pred_energy2','neurosim_energy']].values
    y = data[['realadd_energy']].values

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

if args.train == "mlp":
    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)
    scaler_x_params = {
        'mean': scaler_x.mean_,
        'scale': scaler_x.scale_
    }

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)
    scaler_y_params = {
        'mean': scaler_y.mean_,
        'scale': scaler_y.scale_
    }

    mlp = MLPRegressor(hidden_layer_sizes=(64,32), activation='relu', alpha=0.0001, solver='adam', learning_rate='constant', learning_rate_init=0.001, max_iter=1000, batch_size=256, shuffle=True, random_state=777, early_stopping=True, validation_fraction=0.3)

    # Fit the model on the training data
    mlp.fit(X_train_scaled, y_train_scaled)

    # Predict on the testing data
    y_pred = mlp.predict(X_test_scaled)

    r2 = r2_score(y_test_scaled, y_pred)
    print(f"R-squared for Output : {r2:.4f}")
    # Calculate the mean squared error of the model
    mse = mean_squared_error(y_test_scaled, y_pred)

    print("mse: ", mse)
    print("pred max: ", max(y_pred))
    print("real max: ", max(y_test_scaled))
    plt.plot([min(y_test_scaled), max(y_test_scaled)], [min(y_test_scaled), max(y_test_scaled)], color='red', linestyle='--', linewidth=2, alpha=0.6,zorder=0)  # Adding x=y line
    plt.scatter(y_test_scaled, y_pred, color='blue', alpha=0.6, s=0.9)
    # plt.axhline(y=, color='red', linestyle='--', linewidth=2)  # Adding a horizontal line at y=0 for reference
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted  Values')
    plt.title(f'Predicted vs Actual({args.metric})')
    plt.savefig(f'predicted_vs_actual_scalerfactor_{args.model}_{args.metric}_{args.ADC}_{args.Cellbit}.png')

    with open(f'./predict_model/{args.model}/mlp_{args.model}_{args.metric}_{args.ADC}_{args.Cellbit}_scale_final.pkl', 'wb') as file:
        pickle.dump(mlp, file)
    joblib.dump(scaler_x_params, f'./predict_model/{args.model}/mlp_scaler_x_{args.model}_{args.metric}_{args.ADC}_{args.Cellbit}final.pkl')
    joblib.dump(scaler_y_params, f'./predict_model/{args.model}/mlp_scaler_y_{args.model}_{args.metric}_{args.ADC}_{args.Cellbit}final.pkl')
else:
    linear_model = LinearRegression()
    polynomial_features = PolynomialFeatures(degree=2)

    X_train_poly = polynomial_features.fit_transform(X_train)
    with open(f'./predict_model/{args.model}/polynomial_features_{args.model}_{args.metric}_{args.ADC}_{args.Cellbit}final.pkl', 'wb') as file:
        joblib.dump(polynomial_features, file)

    X_test_poly = polynomial_features.transform(X_test)
    linear_model.fit(X_train_poly, y_train)
    poly_predictions = linear_model.predict(X_test_poly)
    poly_mae = mean_absolute_error(y_test, poly_predictions)
    poly_mse = mean_squared_error(y_test, poly_predictions)

    print('MAE: ', poly_mae, 'MSE: ', poly_mse)
    # 예측 결과 시각화
    plt.scatter(poly_predictions, y_test, color='black', label='Actual')
    plt.savefig(f'predicted_vs_actual_scalerfactor_{args.model}_{args.metric}_{args.ADC}_{args.Cellbit}.png')
    plt.show()

    #save model
    model_filename = f'./predict_model/{args.model}/polynomial_regression_{args.model}_{args.metric}_{args.ADC}_{args.Cellbit}final.pkl'

    # Open the file in write-binary mode and save the model
    with open(model_filename, 'wb') as file:
        joblib.dump(linear_model, file)

    loaded_model_filename = f'./predict_model/{args.model}/polynomial_regression_{args.model}_{args.metric}_{args.ADC}_{args.Cellbit}final.pkl'

    # Open the file in read-binary mode and load the model
    with open(loaded_model_filename, 'rb') as file:
        loaded_linear_model = joblib.load(file)

    test_pred = loaded_linear_model.predict(X_test_poly)
    poly_mae_ = mean_absolute_error(y_test, test_pred)
    poly_mse_ = mean_squared_error(y_test, test_pred)

    print('MAE: ', poly_mae_, 'MSE: ', poly_mse_)