import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# Import our models
from model import CryptoPricePredictor as XGBoostPredictor
from hybrid_model import HybridStockPredictor as HybridPredictor

def load_and_preprocess_data(data_path):
    """Load and preprocess the orderbook data"""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Preprocess timestamps
    df['COLLECTION_TIME'] = pd.to_datetime(df['COLLECTION_TIME'])
    df.set_index('COLLECTION_TIME', inplace=True)
    
    # Calculate basic features
    df['mid_price'] = (df['BID_PRICE_1'] + df['ASK_PRICE_1']) / 2
    df['spread'] = df['ASK_PRICE_1'] - df['BID_PRICE_1']
    
    bid_size_cols = [f'BID_SIZE_{i}' for i in range(1, 12)]
    ask_size_cols = [f'ASK_SIZE_{i}' for i in range(1, 12)]
    
    df['total_bid_size'] = df[bid_size_cols].sum(axis=1)
    df['total_ask_size'] = df[ask_size_cols].sum(axis=1)
    df['ob_imbalance'] = (df['total_bid_size'] - df['total_ask_size']) / (df['total_bid_size'] + df['total_ask_size'] + 1e-10)
    
    print(f"Loaded {len(df)} data points")
    return df

def train_and_evaluate(df, train_size=0.8, lstm_epochs=50, lstm_batch_size=32):
    """Train and evaluate the hybrid model"""
    # Create output directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Split data into train and test sets
    train_size = int(len(df) * train_size)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    print(f"Training set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Initialize models
    xgb_model_path = 'models/xgboost_model.json'
    lstm_model_path = 'models/lstm_model.pt'
    
    # Initialize hybrid model
    hybrid_model = HybridPredictor(xgb_model_path)
    
    # Train LSTM feature extractor
    print("Training LSTM feature extractor...")
    start_time = time.time()
    hybrid_model.train_lstm(train_df, epochs=lstm_epochs, batch_size=lstm_batch_size)
    lstm_training_time = time.time() - start_time
    print(f"LSTM training completed in {lstm_training_time:.2f} seconds")
    
    # Train XGBoost model
    print("Training XGBoost model...")
    start_time = time.time()
    hybrid_model.train_xgboost(train_df)
    xgb_training_time = time.time() - start_time
    print(f"XGBoost training completed in {xgb_training_time:.2f} seconds")
    
    # Save models
    print("Saving models...")
    hybrid_model.save_lstm_model(lstm_model_path)
    
    # Make predictions
    print("Making predictions...")
    predictions = hybrid_model.predict(test_df)
    
    # Calculate metrics
    mse = mean_squared_error(test_df['close'], predictions)
    r2 = r2_score(test_df['close'], predictions)
    
    print(f"Test MSE: {mse:.6f}")
    print(f"Test R2: {r2:.6f}")
    
    # Plot predictions
    plt.figure(figsize=(12, 6))
    plt.plot(test_df.index, test_df['close'], label='Actual')
    plt.plot(test_df.index, predictions, label='Predicted')
    plt.legend()
    plt.title('Hybrid Model Predictions vs Actual Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.savefig('models/prediction_plot.png')
    plt.close()
    
    # Plot feature importance
    importance = hybrid_model.get_feature_importance()
    plt.figure(figsize=(12, 6))
    plt.bar(importance['feature'][:15], importance['importance'][:15])
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 15 Feature Importance')
    plt.tight_layout()
    plt.savefig('models/feature_importance.png')
    plt.close()
    
    # Save metrics to file
    with open('models/metrics.txt', 'w') as f:
        f.write(f"Test MSE: {mse:.6f}\n")
        f.write(f"Test R2: {r2:.6f}\n")
        f.write(f"LSTM Training Time: {lstm_training_time:.2f} seconds\n")
        f.write(f"XGBoost Training Time: {xgb_training_time:.2f} seconds\n")
    
    return hybrid_model

if __name__ == "__main__":
    # Load and preprocess data
    data_path = 'data/book_updates.csv'
    df = load_and_preprocess_data(data_path)
    
    # Train and evaluate model
    model = train_and_evaluate(df)
    
    print("Training completed successfully!")
    print("Models and evaluation results saved to the 'models' directory.") 