# Crypto Price Prediction with Hybrid Model
# This script demonstrates how to use both the original XGBoost model and the new hybrid model (LSTM + XGBoost) for crypto price prediction.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from model import CryptoPricePredictor
from hybrid_model import HybridStockPredictor
import time
import os

# 1. Load and Preprocess Data
def load_data(data_path='data/book_updates.csv'):
    """Load and preprocess the orderbook data"""
    # Read the data
    df = pd.read_csv(data_path)
    
    # Convert timestamp to datetime
    df['COLLECTION_TIME'] = pd.to_datetime(df['COLLECTION_TIME'])
    
    # Calculate basic features
    df['mid_price'] = (df['BID_PRICE_1'] + df['ASK_PRICE_1']) / 2
    df['spread'] = df['ASK_PRICE_1'] - df['BID_PRICE_1']
    df['total_bid_size'] = df[['BID_SIZE_1', 'BID_SIZE_2', 'BID_SIZE_3', 'BID_SIZE_4', 'BID_SIZE_5']].sum(axis=1)
    df['total_ask_size'] = df[['ASK_SIZE_1', 'ASK_SIZE_2', 'ASK_SIZE_3', 'ASK_SIZE_4', 'ASK_SIZE_5']].sum(axis=1)
    df['order_book_imbalance'] = df['total_bid_size'] / (df['total_bid_size'] + df['total_ask_size'])
    
    return df

# Load the data
df = load_data()
print(f"Loaded {len(df)} rows of data")

# 2. Train and Evaluate Both Models
def train_and_evaluate_models(df, train_size=0.8):
    """Train and evaluate both XGBoost and hybrid models"""
    # Split data into train and test sets
    train_size = int(len(df) * train_size)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    
    # Initialize models
    xgb_model = CryptoPricePredictor(model_path='models/xgboost_model.json')
    hybrid_model = HybridStockPredictor(xgb_model_path='models/xgboost_model.json')
    
    # Train XGBoost model
    print("Training XGBoost model...")
    xgb_start_time = time.time()
    xgb_model.update_model(train_data)
    xgb_train_time = time.time() - xgb_start_time
    
    # Train hybrid model
    print("\nTraining hybrid model...")
    hybrid_start_time = time.time()
    
    # Train LSTM feature extractor
    print("Training LSTM feature extractor...")
    hybrid_model.train_lstm(train_data, epochs=50, batch_size=32)
    
    # Train XGBoost part
    print("Training XGBoost part...")
    hybrid_model.train_xgboost(train_data)
    
    hybrid_train_time = time.time() - hybrid_start_time
    
    # Make predictions
    print("\nMaking predictions...")
    xgb_pred = xgb_model.predict(test_data)
    hybrid_pred = hybrid_model.predict(test_data)
    
    # Calculate metrics
    actual = test_data['mid_price'].values
    
    xgb_mse = mean_squared_error(actual, xgb_pred)
    xgb_r2 = r2_score(actual, xgb_pred)
    
    hybrid_mse = mean_squared_error(actual, hybrid_pred)
    hybrid_r2 = r2_score(actual, hybrid_pred)
    
    return {
        'xgb': {
            'predictions': xgb_pred,
            'mse': xgb_mse,
            'r2': xgb_r2,
            'train_time': xgb_train_time
        },
        'hybrid': {
            'predictions': hybrid_pred,
            'mse': hybrid_mse,
            'r2': hybrid_r2,
            'train_time': hybrid_train_time
        }
    }

# Train and evaluate models
results = train_and_evaluate_models(df)

# Print results
print("\nResults:")
print("XGBoost Model:")
print(f"MSE: {results['xgb']['mse']:.6f}")
print(f"R2 Score: {results['xgb']['r2']:.6f}")
print(f"Training Time: {results['xgb']['train_time']:.2f} seconds")

print("\nHybrid Model:")
print(f"MSE: {results['hybrid']['mse']:.6f}")
print(f"R2 Score: {results['hybrid']['r2']:.6f}")
print(f"Training Time: {results['hybrid']['train_time']:.2f} seconds")

# 3. Visualize Results
def plot_predictions(actual, xgb_pred, hybrid_pred):
    """Plot actual vs predicted prices for both models"""
    plt.figure(figsize=(15, 7))
    
    # Plot actual prices
    plt.plot(actual, label='Actual', color='black', alpha=0.7)
    
    # Plot XGBoost predictions
    plt.plot(xgb_pred, label='XGBoost', color='blue', alpha=0.5)
    
    # Plot hybrid model predictions
    plt.plot(hybrid_pred, label='Hybrid (LSTM+XGBoost)', color='red', alpha=0.5)
    
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# Get test data
test_data = df.iloc[int(len(df) * 0.8):]
actual = test_data['mid_price'].values

# Plot predictions
plot_predictions(actual, results['xgb']['predictions'], results['hybrid']['predictions'])

# 4. Feature Importance Analysis
def plot_feature_importance(model, title):
    """Plot feature importance for a model"""
    importance = model.get_feature_importance()
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importance)), importance.values())
    plt.xticks(range(len(importance)), importance.keys(), rotation=45, ha='right')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Plot feature importance for both models
xgb_model = CryptoPricePredictor(model_path='models/xgboost_model.json')
hybrid_model = HybridStockPredictor(xgb_model_path='models/xgboost_model.json')

plot_feature_importance(xgb_model, 'XGBoost Feature Importance')
plot_feature_importance(hybrid_model, 'Hybrid Model Feature Importance')

# 5. Model Comparison
def compare_models(results):
    """Compare the performance of both models"""
    metrics = ['mse', 'r2', 'train_time']
    models = ['XGBoost', 'Hybrid']
    
    comparison = pd.DataFrame({
        'Metric': metrics * 2,
        'Model': ['XGBoost'] * 3 + ['Hybrid'] * 3,
        'Value': [
            results['xgb']['mse'],
            results['xgb']['r2'],
            results['xgb']['train_time'],
            results['hybrid']['mse'],
            results['hybrid']['r2'],
            results['hybrid']['train_time']
        ]
    })
    
    # Pivot the dataframe for better visualization
    comparison_pivot = comparison.pivot(index='Metric', columns='Model', values='Value')
    
    # Calculate improvement
    comparison_pivot['Improvement'] = ((comparison_pivot['XGBoost'] - comparison_pivot['Hybrid']) / comparison_pivot['XGBoost'] * 100).round(2)
    
    return comparison_pivot

# Compare models
comparison = compare_models(results)
print("\nModel Comparison:")
print(comparison) 