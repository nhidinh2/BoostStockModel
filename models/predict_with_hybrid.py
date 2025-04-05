import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from hybrid_model import HybridStockPredictor

def load_latest_data(data_path, num_samples=100):
    """Load the most recent data samples"""
    df = pd.read_csv(data_path)
    df['COLLECTION_TIME'] = pd.to_datetime(df['COLLECTION_TIME'])
    return df.tail(num_samples)

def make_prediction(model, data):
    """Make a prediction using the hybrid model"""
    # Convert data to the format expected by the model
    data_list = data.values.tolist()
    
    # Make prediction
    start_time = time.time()
    prediction = model.predict(data_list)
    prediction_time = time.time() - start_time
    
    return prediction, prediction_time

def plot_prediction(data, prediction, title="Price Prediction"):
    """Plot the prediction result"""
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['close'], label='Actual')
    plt.axhline(y=prediction, color='r', linestyle='--', label='Predicted')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig('models/latest_prediction.png')
    plt.close()

def main():
    # Initialize the hybrid model
    print("Initializing hybrid model...")
    model = HybridStockPredictor(
        xgb_model_path='models/xgboost_model.json',
        lstm_model_path='models/lstm_model.pt'
    )
    
    # Load latest data
    print("Loading latest data...")
    data_path = 'data/book_updates.csv'
    latest_data = load_latest_data(data_path)
    
    # Make prediction
    print("Making prediction...")
    prediction, prediction_time = make_prediction(model, latest_data)
    
    # Print results
    print(f"\nPrediction Results:")
    print(f"Predicted Price: {prediction:.2f}")
    print(f"Prediction Time: {prediction_time:.4f} seconds")
    
    # Plot results
    print("Plotting results...")
    plot_prediction(latest_data, prediction)
    
    # Get feature importance
    importance = model.get_feature_importance()
    print("\nTop 5 Most Important Features:")
    print(importance.head())
    
    print("\nPrediction completed successfully!")
    print("Results saved to 'models/latest_prediction.png'")

if __name__ == "__main__":
    main() 