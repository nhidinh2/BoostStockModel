import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import logging
import os
from typing import List, Optional, Union
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CryptoPricePredictor:
    def __init__(self, model_path: str, max_history_rows: int = 100000):
        """
        Initialize the predictor with data path and setup logging.
        
        Args:
            model_path: Path to the XGBoost model file
            max_history_rows: Maximum number of rows to keep in total_candlestick_df (for memory management)
        """
        self.model = xgb.XGBRegressor()
        self.model_path = model_path
        self.max_history_rows = max_history_rows
        
        # The total candlestick_df is the cumulative data of orderbook
        self.total_candlestick_df: Optional[pd.DataFrame] = None
        
        # Feature engineering parameters
        self.window_length = 20
        self.rolling_window = 20
        self.resample_interval = '500ms' 
        self.lag_periods = 3 
        self.ema_spans = [3, 5, 8]
        self.forecast_horizon = 2
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'avg_total_size', 'spread', 'ob_imbalance',
            'rolling_mean', 'rolling_std', 'rolling_min', 'rolling_max',
            'rolling_spread_mean', 'rolling_spread_std',
            'rolling_ob_imbalance_mean', 'rolling_ob_imbalance_std',
            # Lagged variables
            'lag_close_1', 'lag_close_2', 'lag_close_3',
            # Exponential Moving Averages
            'ema_close_3', 'ema_close_5', 'ema_close_8',
            # Technical Indicators
            'rsi', 'macd', 'macd_signal', 'macd_hist'
        ]
        
        # Load existing model with error handling
        try:
            if os.path.exists(model_path):
                self.model.load_model(model_path)
                logger.info(f"Successfully loaded model from {model_path}")
            else:
                logger.warning(f"Model file not found at {model_path}. Using default XGBRegressor.")
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}. Using default XGBRegressor.")
    
    # X is list of orderbook in the format of ['COLLECTION_TIME', 'BID_PRICE_1', 'BID_SIZE_1', 'BID_PRICE_2', 'BID_SIZE_2', ..., 'BID_PRICE_11', 'BID_SIZE_11', 'ASK_PRICE_1',
    #    'ASK_SIZE_1', 'ASK_PRICE_2', 'ASK_SIZE_2', ..., 'ASK_PRICE_11', 'ASK_SIZE_11']
    def process_data(self, X: List[List[Union[str, float]]]) -> pd.DataFrame:
        """
        Process orderbook data and extract features.
        
        Args:
            X: List of orderbook snapshots, each containing COLLECTION_TIME and 44 orderbook fields
            
        Returns:
            DataFrame containing the latest feature row
        """
        # Input validation
        if not X or len(X) == 0:
            raise ValueError("Input X cannot be empty")
        
        expected_columns = 1 + 11 * 4  # COLLECTION_TIME + 11 levels * (BID_PRICE + BID_SIZE + ASK_PRICE + ASK_SIZE)
        
        # Validate input format
        for i, row in enumerate(X):
            if not isinstance(row, (list, tuple)):
                raise ValueError(f"Row {i} must be a list or tuple, got {type(row)}")
            if len(row) != expected_columns:
                raise ValueError(f"Row {i} has {len(row)} columns, expected {expected_columns}")
        
        try:
            # Convert list to DataFrame
            columns = ['COLLECTION_TIME']
            for i in range(1, 12):
                columns.extend([f'BID_PRICE_{i}', f'BID_SIZE_{i}', f'ASK_PRICE_{i}', f'ASK_SIZE_{i}'])
            
            # Init the df and calculate the mid_price, spread, total_bid_size, total_ask_size, ob_imbalance
            df = pd.DataFrame(X, columns=columns)
            df['COLLECTION_TIME'] = pd.to_datetime(df['COLLECTION_TIME'])
            df.set_index('COLLECTION_TIME', inplace=True)
            
            # Validate price and size data
            if df['BID_PRICE_1'].isna().any() or df['ASK_PRICE_1'].isna().any():
                logger.warning("Missing price data detected, attempting to fill")
            
            df['mid_price'] = (df['BID_PRICE_1'] + df['ASK_PRICE_1']) / 2
            df['spread'] = df['ASK_PRICE_1'] - df['BID_PRICE_1']
            
            bid_size_cols = [f'BID_SIZE_{i}' for i in range(1, 12)]
            ask_size_cols = [f'ASK_SIZE_{i}' for i in range(1, 12)] 
            df['total_bid_size'] = df[bid_size_cols].sum(axis=1)
            df['total_ask_size'] = df[ask_size_cols].sum(axis=1)
            df['ob_imbalance'] = (df['total_bid_size'] - df['total_ask_size']) / (df['total_bid_size'] + df['total_ask_size'] + 1e-10)
            
            # Resample the data and calculate the candlestick df
            candlestick_df = pd.DataFrame()
            candlestick_df['open'] = df['mid_price'].resample(self.resample_interval).first()
            candlestick_df['high'] = df['mid_price'].resample(self.resample_interval).max()
            candlestick_df['low'] = df['mid_price'].resample(self.resample_interval).min()
            candlestick_df['close'] = df['mid_price'].resample(self.resample_interval).last()
            candlestick_df['avg_total_size'] = (df['total_bid_size'] + df['total_ask_size']).resample(self.resample_interval).mean()
            candlestick_df['spread'] = df['spread'].resample(self.resample_interval).mean()
            candlestick_df['ob_imbalance'] = df['ob_imbalance'].resample(self.resample_interval).mean()
            # Use non-deprecated fillna method
            candlestick_df = candlestick_df.ffill()
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            raise
        
        # Concatenate the new candlestick_df with the total_candlestick_df
        candlestick_df.reset_index(inplace=True)
        if self.total_candlestick_df is not None:
            # Concatenate the historical data with the new candlestick data
            if len(candlestick_df) > 0 and len(self.total_candlestick_df) > 0:
                if candlestick_df['COLLECTION_TIME'].iloc[0] == self.total_candlestick_df['COLLECTION_TIME'].iloc[-1]:
                    candlestick_df = candlestick_df.iloc[1:]
            if len(candlestick_df) > 0:
                self.total_candlestick_df = pd.concat([self.total_candlestick_df, candlestick_df], ignore_index=True)
        else:
            # If no historical data exists, use the new candlestick data directly
            self.total_candlestick_df = candlestick_df
        
        # Memory management: limit the size of total_candlestick_df
        if self.total_candlestick_df is not None and len(self.total_candlestick_df) > self.max_history_rows:
            rows_to_keep = self.max_history_rows
            self.total_candlestick_df = self.total_candlestick_df.iloc[-rows_to_keep:].reset_index(drop=True)
            logger.info(f"Trimmed total_candlestick_df to {rows_to_keep} rows for memory management")
        
        # Calculate the rolling mean, std, min, max
        self.total_candlestick_df['rolling_mean'] = self.total_candlestick_df['close'].rolling(window=self.rolling_window).mean()
        self.total_candlestick_df['rolling_std'] = self.total_candlestick_df['close'].rolling(window=self.rolling_window).std()
        self.total_candlestick_df['rolling_min'] = self.total_candlestick_df['close'].rolling(window=self.rolling_window).min()
        self.total_candlestick_df['rolling_max'] = self.total_candlestick_df['close'].rolling(window=self.rolling_window).max()
        
        # Rolling statistics for 'spread'
        self.total_candlestick_df['rolling_spread_mean'] = self.total_candlestick_df['spread'].rolling(window=self.rolling_window).mean()
        self.total_candlestick_df['rolling_spread_std'] = self.total_candlestick_df['spread'].rolling(window=self.rolling_window).std()

        # Rolling statistics for 'ob_imbalance'
        self.total_candlestick_df['rolling_ob_imbalance_mean'] = self.total_candlestick_df['ob_imbalance'].rolling(window=self.rolling_window).mean()
        self.total_candlestick_df['rolling_ob_imbalance_std'] = self.total_candlestick_df['ob_imbalance'].rolling(window=self.rolling_window).std()
        
        # Calculate lag features
        for lag in range(1, self.lag_periods + 1):
            self.total_candlestick_df[f'lag_close_{lag}'] = self.total_candlestick_df['close'].shift(lag)
            
        ema_spans = [3, 5, 8]  # EMA periods
        for span in ema_spans:
            self.total_candlestick_df[f'ema_close_{span}'] = self.total_candlestick_df['close'].ewm(span=span, adjust=False).mean()
        
        # Calculate Relative Strength Index (RSI)
        window_length = 20
        delta = self.total_candlestick_df['close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        avg_gain = gain.rolling(window=window_length).mean()
        avg_loss = loss.rolling(window=window_length).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        self.total_candlestick_df['rsi'] = 100 - (100 / (1 + rs))

        # Calculate Moving Average Convergence Divergence (MACD)
        ema_5 = self.total_candlestick_df['close'].ewm(span=5, adjust=False).mean()
        ema_10 = self.total_candlestick_df['close'].ewm(span=10, adjust=False).mean()
        self.total_candlestick_df['macd'] = ema_5 - ema_10
        self.total_candlestick_df['macd_signal'] = self.total_candlestick_df['macd'].ewm(span=9, adjust=False).mean()
        self.total_candlestick_df['macd_hist'] = self.total_candlestick_df['macd'] - self.total_candlestick_df['macd_signal']

        # Reset index and fill the future close for the past data
        self.total_candlestick_df.reset_index(drop=True, inplace=True)
        self.total_candlestick_df['future_close'] = self.total_candlestick_df['close'].shift(-self.forecast_horizon)

        return self.total_candlestick_df[self.feature_columns].iloc[[-1]]
    
    def update_model(self, min_samples: int = 100) -> None:
        """
        Update the model with the latest data using randomized search.
        
        Args:
            min_samples: Minimum number of samples required to update the model
        """
        # If there is no data, no need to update the model
        if self.total_candlestick_df is None:
            logger.warning("No data available for model update")
            return
        
        df = self.total_candlestick_df.copy()
        df.dropna(inplace=True)
        
        # Check if we have enough data
        if len(df) < min_samples:
            logger.warning(f"Insufficient data for model update: {len(df)} samples (minimum: {min_samples})")
            return
        
        # Check if future_close column exists and has valid data
        if 'future_close' not in df.columns or df['future_close'].isna().all():
            logger.warning("No valid future_close data available for model update")
            return
        
        try:
            # Since self.total_candlestick_df is actively updated with new data, we can use the latest data to update the model
            param_grid = {
                'n_estimators': [500, 1000],
                'max_depth': [9, 15, 20],
                'learning_rate': [0.01, 0.02, 0.04],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.8],
                'gamma': [0.1],
                'reg_lambda': [1, 1.5, 2],
            }
            
            X = df[self.feature_columns]
            y = df['future_close']
            
            # Validate that we have all required feature columns
            missing_cols = set(self.feature_columns) - set(X.columns)
            if missing_cols:
                logger.error(f"Missing feature columns: {missing_cols}")
                return
            
            xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

            logger.info(f"Starting model update with {len(X)} samples")
            random_search = RandomizedSearchCV(
                estimator=xgb_model,
                param_distributions=param_grid,
                n_iter=50,  # Number of parameter settings sampled
                scoring='neg_mean_squared_error',
                cv=5,
                verbose=0,
                random_state=42,
                n_jobs=-1
            )
            
            random_search.fit(X, y)
            best_model = random_search.best_estimator_
            best_model.save_model(self.model_path)
            self.model = best_model
            logger.info(f"Model updated successfully and saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error updating model: {e}")
            raise
    
    def predict(self, X: List[List[Union[str, float]]]) -> np.ndarray:
        """
        Make predictions on new orderbook data.
        
        Args:
            X: List of orderbook snapshots
            
        Returns:
            Array of predicted prices
        """
        try:
            df = self.process_data(X)
            # df already contains only feature columns from process_data return
            y = self.model.predict(df)
            return y
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    
# Sample Usage
if __name__=='__main__':
    model = CryptoPricePredictor('models/xgboost_model.json')
    input_1 = [
        ['2024-11-11 19:18:40.084535040', 68.51, 100, 68.53, 1600, 68.5, 100, 68.54, 1900, 68.49, 100, 68.57, 300, 68.48, 1600, 68.63, 300, 68.47, 1600, 68.64, 1900, 68.46, 1600, 68.65, 100, 68.45, 1600, 68.7, 1600, 68.44, 1600, 68.75, 1600, 68.38, 1900, 68.78, 1900, 68.25, 1600, 68.8, 100, 68.24, 2000, 69.05, 1900],
        ['2024-11-11 19:18:41.084535040', 68.52, 100, 68.53, 1600, 68.5, 100, 68.54, 1900, 68.49, 100, 68.57, 300, 68.48, 1600, 68.63, 300, 68.47, 1600, 68.64, 1900, 68.46, 1600, 68.65, 100, 68.45, 1600, 68.7, 1600, 68.44, 1600, 68.75, 1600, 68.38, 1900, 68.78, 1900, 68.25, 1600, 68.8, 100, 68.24, 2000, 69.05, 1900],
    ]
    price_1 = model.predict(input_1)
    
    input_2 = [
        ['2024-11-11 19:18:42.084535040', 68.51, 100, 68.53, 1600, 68.5, 100, 68.54, 1900, 68.49, 100, 68.57, 300, 68.48, 1600, 68.63, 300, 68.47, 1600, 68.64, 1900, 68.46, 1600, 68.65, 100, 68.45, 1600, 68.7, 1600, 68.44, 1600, 68.75, 1600, 68.38, 1900, 68.78, 1900, 68.25, 1600, 68.8, 100, 68.24, 2000, 69.05, 1900],
    ]
    price_2 = model.predict(input_2)
    
    # Update the model with new data
    model.update_model()
    