#!/usr/bin/env python
# coding: utf-8

# # Crypto Price Prediction with Hybrid Model
# This notebook demonstrates how to use both the original XGBoost model and the new hybrid model (LSTM + XGBoost) for crypto price prediction.

# In[55]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from model import CryptoPricePredictor
from hybrid_model import HybridStockPredictor
import time
import os


# ## 1. Load and Preprocess Data

# In[56]:


def load_data(data_path='data/book_updates.csv'):  # Changed from '../data/book_updates.csv' to 'data/book_updates.csv'
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


# ## 2. Train and Evaluate Both Models

# In[59]:


def train_and_evaluate_models(df, train_size=0.8):
    """
    Train and evaluate both XGBoost and hybrid models.
    
    Args:
        df (pd.DataFrame): Input DataFrame with orderbook data
        train_size (float): Proportion of data to use for training
        
    Returns:
        dict: Dictionary containing predictions and metrics for both models
    """
    # Split data into training and testing sets
    train_idx = int(len(df) * train_size)
    train_data = df.iloc[:train_idx]
    test_data = df.iloc[train_idx:]
    
    # Initialize models
    xgb_model = CryptoPricePredictor(model_path='best_xgb_model.json')
    hybrid_model = HybridStockPredictor(xgb_model_path='best_xgb_model.json')
    
    # Process training data and update models
    print("Processing training data...")
    xgb_model.process_data(train_data)
    hybrid_model.process_data(train_data)
    
    # Train models
    print("Training XGBoost model...")
    xgb_model.update_model()
    
    print("Training hybrid model...")
    try:
        hybrid_model.update_model()
    except Exception as e:
        print(f"Error training hybrid model: {str(e)}")
        print("Will continue with XGBoost predictions only")
    
    # Save the trained XGBoost model
    xgb_model.model.save_model('best_xgb_model.json')
    
    # Make predictions
    print("Making predictions...")
    xgb_pred = xgb_model.predict(test_data)
    
    try:
        hybrid_pred = hybrid_model.predict(test_data)
    except Exception as e:
        print(f"Error making hybrid predictions: {str(e)}")
        print("Using XGBoost predictions as hybrid predictions")
        hybrid_pred = xgb_pred
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np
    
    # Get actual mid prices for the test period
    actual_prices = test_data['mid_price'].values
    
    # Calculate metrics for XGBoost model
    xgb_mse = mean_squared_error(actual_prices, xgb_pred)
    xgb_r2 = r2_score(actual_prices, xgb_pred)
    
    # Calculate metrics for hybrid model
    hybrid_mse = mean_squared_error(actual_prices, hybrid_pred)
    hybrid_r2 = r2_score(actual_prices, hybrid_pred)
    
    # Print results
    print("\nModel Performance:")
    print("XGBoost Model:")
    print(f"MSE: {xgb_mse:.4f}")
    print(f"R2 Score: {xgb_r2:.4f}")
    print("\nHybrid Model:")
    print(f"MSE: {hybrid_mse:.4f}")
    print(f"R2 Score: {hybrid_r2:.4f}")
    
    # Return results
    return {
        'xgb_predictions': xgb_pred,
        'hybrid_predictions': hybrid_pred,
        'actual_prices': actual_prices,
        'metrics': {
            'xgb': {'mse': xgb_mse, 'r2': xgb_r2},
            'hybrid': {'mse': hybrid_mse, 'r2': hybrid_r2}
        }
    }


# In[60]:


# Train and evaluate models
results = train_and_evaluate_models(df)

# Print results
print("\nResults:")
print("XGBoost Model:")
print(f"MSE: {results['metrics']['xgb']['mse']:.6f}")
print(f"R2 Score: {results['metrics']['xgb']['r2']:.6f}")

print("\nHybrid Model:")
print(f"MSE: {results['metrics']['hybrid']['mse']:.6f}")
print(f"R2 Score: {results['metrics']['hybrid']['r2']:.6f}")


# In[51]:


# Get test data
test_data = df.iloc[int(len(df) * 0.8):]
actual = test_data['mid_price'].values

# Plot predictions
plot_predictions(actual, results['xgb_predictions'], results['hybrid_predictions'])


# ## 3. Visualize Results

# In[29]:


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


# ## 4. Feature Importance Analysis

# In[ ]:


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


# ## 5. Model Comparison

# In[ ]:


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

