"""
Fixed implementation of the hybrid model for crypto price prediction.
This file contains the corrected version of the HybridStockPredictor class.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

class HybridStockPredictor:
    def __init__(self, xgb_model_path: str = None, lstm_model_path: str = None):
        """Initialize the hybrid predictor with XGBoost and LSTM components"""
        # Initialize XGBoost model with default parameters
        self.xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=6,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1
        )
        self.xgb_model_path = xgb_model_path
        
        # Initialize LSTM feature extractor
        self.lstm_model = self._build_lstm_model()
        self.lstm_model_path = lstm_model_path
        
        # Load existing models if paths are provided
        if xgb_model_path is not None:
            self.load_xgb_model(xgb_model_path)
        if lstm_model_path is not None:
            self.load_lstm_model(lstm_model_path)
        
        # Feature engineering parameters
        self.window_length = 20
        self.rolling_window = 20
        self.resample_interval = '500L'
        self.lag_periods = 3
        self.ema_spans = [3, 5, 8]
        self.forecast_horizon = 1
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'avg_total_size', 'spread', 'ob_imbalance',
            'rolling_mean', 'rolling_std', 'rolling_min', 'rolling_max',
            'rolling_spread_mean', 'rolling_spread_std',
            'rolling_ob_imbalance_mean', 'rolling_ob_imbalance_std',
            'lag_close_1', 'lag_close_2', 'lag_close_3',
            'ema_close_3', 'ema_close_5', 'ema_close_8',
            'rsi', 'macd', 'macd_signal', 'macd_hist'
        ]
        
        # The total candlestick_df is the cumulative data of orderbook
        self.total_candlestick_df = None
        self.temp = None
    
    def _build_lstm_model(self):
        """Build the LSTM model architecture"""
        model = Sequential([
            LSTM(64, input_shape=(self.window_length, len(self.feature_columns)), return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def load_xgb_model(self, model_path):
        """Load a saved XGBoost model from the specified path"""
        try:
            self.xgb_model.load_model(model_path)
            print(f"Successfully loaded XGBoost model from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load XGBoost model from {model_path}: {e}")
            print("Initializing a new XGBoost model instead.")
    
    def load_lstm_model(self, model_path):
        """Load a saved LSTM model from the specified path"""
        try:
            self.lstm_model = load_model(model_path)
            print(f"Successfully loaded LSTM model from {model_path}")
        except Exception as e:
            print(f"Warning: Could not load LSTM model from {model_path}: {e}")
            print("Initializing a new LSTM model instead.")
    
    def process_data(self, X) -> pd.DataFrame:
        """Process raw orderbook data into features"""
        # Convert list to DataFrame
        columns = ['COLLECTION_TIME']
        for i in range(1, 12):
            columns.extend([f'BID_PRICE_{i}', f'BID_SIZE_{i}', f'ASK_PRICE_{i}', f'ASK_SIZE_{i}'])
        
        # Initialize DataFrame and calculate basic features
        df = pd.DataFrame(X, columns=columns)
        df['COLLECTION_TIME'] = pd.to_datetime(df['COLLECTION_TIME'])
        df.set_index('COLLECTION_TIME', inplace=True)
        df['mid_price'] = (df['BID_PRICE_1'] + df['ASK_PRICE_1']) / 2
        df['spread'] = df['ASK_PRICE_1'] - df['BID_PRICE_1']
        
        bid_size_cols = [f'BID_SIZE_{i}' for i in range(1, 12)]
        ask_size_cols = [f'ASK_SIZE_{i}' for i in range(1, 12)]
        df['total_bid_size'] = df[bid_size_cols].sum(axis=1)
        df['total_ask_size'] = df[ask_size_cols].sum(axis=1)
        df['ob_imbalance'] = (df['total_bid_size'] - df['total_ask_size']) / (df['total_bid_size'] + df['total_ask_size'] + 1e-10)
        
        # Resample and calculate candlestick data
        candlestick_df = pd.DataFrame()
        candlestick_df['open'] = df['mid_price'].resample(self.resample_interval).first()
        candlestick_df['high'] = df['mid_price'].resample(self.resample_interval).max()
        candlestick_df['low'] = df['mid_price'].resample(self.resample_interval).min()
        candlestick_df['close'] = df['mid_price'].resample(self.resample_interval).last()
        candlestick_df['avg_total_size'] = (df['total_bid_size'] + df['total_ask_size']).resample(self.resample_interval).mean()
        candlestick_df['spread'] = df['spread'].resample(self.resample_interval).mean()
        candlestick_df['ob_imbalance'] = df['ob_imbalance'].resample(self.resample_interval).mean()
        candlestick_df.fillna(method='ffill', inplace=True)
        
        # Update total candlestick DataFrame
        candlestick_df.reset_index(inplace=True)
        if self.total_candlestick_df is not None:
            if candlestick_df['COLLECTION_TIME'].iloc[0] == self.total_candlestick_df['COLLECTION_TIME'].iloc[-1]:
                candlestick_df = candlestick_df.iloc[1:]
            self.total_candlestick_df = pd.concat([self.total_candlestick_df, candlestick_df])
        else:
            self.total_candlestick_df = candlestick_df
        
        # Calculate technical indicators
        self._calculate_technical_indicators()
        
        # Reset index and prepare for prediction
        self.total_candlestick_df.reset_index(drop=True, inplace=True)
        self.total_candlestick_df['future_close'] = self.total_candlestick_df['close'].shift(-self.forecast_horizon)
        
        return self.total_candlestick_df[self.feature_columns].iloc[[-1]]
    
    def _calculate_technical_indicators(self):
        """Calculate technical indicators for the data"""
        # Rolling statistics
        self.total_candlestick_df['rolling_mean'] = self.total_candlestick_df['close'].rolling(window=self.rolling_window).mean()
        self.total_candlestick_df['rolling_std'] = self.total_candlestick_df['close'].rolling(window=self.rolling_window).std()
        self.total_candlestick_df['rolling_min'] = self.total_candlestick_df['close'].rolling(window=self.rolling_window).min()
        self.total_candlestick_df['rolling_max'] = self.total_candlestick_df['close'].rolling(window=self.rolling_window).max()
        
        # Rolling statistics for spread and order book imbalance
        self.total_candlestick_df['rolling_spread_mean'] = self.total_candlestick_df['spread'].rolling(window=self.rolling_window).mean()
        self.total_candlestick_df['rolling_spread_std'] = self.total_candlestick_df['spread'].rolling(window=self.rolling_window).std()
        self.total_candlestick_df['rolling_ob_imbalance_mean'] = self.total_candlestick_df['ob_imbalance'].rolling(window=self.rolling_window).mean()
        self.total_candlestick_df['rolling_ob_imbalance_std'] = self.total_candlestick_df['ob_imbalance'].rolling(window=self.rolling_window).std()
        
        # Lag features
        for lag in range(1, self.lag_periods + 1):
            self.total_candlestick_df[f'lag_close_{lag}'] = self.total_candlestick_df['close'].shift(lag)
        
        # Exponential Moving Averages
        for span in self.ema_spans:
            self.total_candlestick_df[f'ema_close_{span}'] = self.total_candlestick_df['close'].ewm(span=span, adjust=False).mean()
        
        # RSI
        delta = self.total_candlestick_df['close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=self.window_length).mean()
        avg_loss = loss.rolling(window=self.window_length).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        self.total_candlestick_df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_5 = self.total_candlestick_df['close'].ewm(span=5, adjust=False).mean()
        ema_10 = self.total_candlestick_df['close'].ewm(span=10, adjust=False).mean()
        self.total_candlestick_df['macd'] = ema_5 - ema_10
        self.total_candlestick_df['macd_signal'] = self.total_candlestick_df['macd'].ewm(span=9, adjust=False).mean()
        self.total_candlestick_df['macd_hist'] = self.total_candlestick_df['macd'] - self.total_candlestick_df['macd_signal']
    
    def update_model(self, train_data=None, **kwargs):
        """Update both XGBoost and LSTM models with new data"""
        if train_data is None:
            return
        
        # Convert the DataFrame to the format expected by process_data
        orderbook_data = []
        for _, row in train_data.iterrows():
            orderbook_row = [row.name]  # COLLECTION_TIME
            for i in range(1, 12):
                orderbook_row.extend([
                    row.get(f'BID_PRICE_{i}', 0),
                    row.get(f'BID_SIZE_{i}', 0),
                    row.get(f'ASK_PRICE_{i}', 0),
                    row.get(f'ASK_SIZE_{i}', 0)
                ])
            orderbook_data.append(orderbook_row)
        
        # Process the data
        df = self.process_data(orderbook_data)
        df.dropna(inplace=True)
        
        # Prepare features and target
        X = df[self.feature_columns]
        y = df['close'].shift(-1).fillna(method='ffill')
        
        # Remove rows with NaN values
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        # Update XGBoost model
        self._update_xgb_model(X, y)
        
        # Update LSTM model
        self._update_lstm_model(X, y)
    
    def _update_xgb_model(self, X, y):
        """Update the XGBoost model with new data"""
        param_grid = {
            'n_estimators': [500, 1000],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.8],
            'gamma': [0.1],
            'reg_lambda': [1, 1.5, 2],
        }
        
        random_search = RandomizedSearchCV(
            estimator=self.xgb_model,
            param_distributions=param_grid,
            n_iter=10,
            scoring='neg_mean_squared_error',
            cv=5,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
        
        random_search.fit(X, y)
        self.xgb_model = random_search.best_estimator_
        
        if self.xgb_model_path is not None:
            self.xgb_model.save_model(self.xgb_model_path)
    
    def _update_lstm_model(self, X, y):
        """Update the LSTM model with new data"""
        # Prepare sequences for LSTM
        X_seq = []
        y_seq = []
        for i in range(len(X) - self.window_length):
            X_seq.append(X.iloc[i:(i + self.window_length)].values)
            y_seq.append(y.iloc[i + self.window_length])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # Train LSTM model
        self.lstm_model.fit(
            X_seq, y_seq,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )
        
        if self.lstm_model_path is not None:
            self.lstm_model.save(self.lstm_model_path)
    
    def predict(self, X):
        """Make predictions using both XGBoost and LSTM models"""
        # Convert the DataFrame to the format expected by process_data
        orderbook_data = []
        for _, row in X.iterrows():
            orderbook_row = [row.name]  # COLLECTION_TIME
            for i in range(1, 12):
                orderbook_row.extend([
                    row.get(f'BID_PRICE_{i}', 0),
                    row.get(f'BID_SIZE_{i}', 0),
                    row.get(f'ASK_PRICE_{i}', 0),
                    row.get(f'ASK_SIZE_{i}', 0)
                ])
            orderbook_data.append(orderbook_row)
        
        # Process the data
        df = self.process_data(orderbook_data)
        
        # Make predictions with XGBoost
        xgb_pred = self.xgb_model.predict(df[self.feature_columns])
        
        # Prepare sequences for LSTM
        X_seq = []
        for i in range(len(df) - self.window_length + 1):
            X_seq.append(df[self.feature_columns].iloc[i:(i + self.window_length)].values)
        X_seq = np.array(X_seq)
        
        # Make predictions with LSTM
        lstm_pred = self.lstm_model.predict(X_seq)
        
        # Combine predictions (simple average)
        combined_pred = (xgb_pred + lstm_pred.flatten()) / 2
        return combined_pred 