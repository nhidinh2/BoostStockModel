import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

class CryptoPricePredictor:
    def __init__(self, model_path: str):
        """Initialize the predictor with data path and setup logging"""
        self.model = xgb.XGBRegressor()
        self.model_path = model_path
        
        # The total candlestick_df is the cumulative data of orderbook
        self.total_candlestick_df = None
        self.temp = None
        
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
            # Lagged variables
            'lag_close_1', 'lag_close_2', 'lag_close_3',
            # Exponential Moving Averages
            'ema_close_3', 'ema_close_5', 'ema_close_8',
            # Technical Indicators
            'rsi', 'macd', 'macd_signal', 'macd_hist'
        ]
        
        # Load existing model
        self.model.load_model(model_path)
    
    # X is list of orderbook in the format of ['COLLECTION_TIME', 'BID_PRICE_1', 'BID_SIZE_1', 'BID_PRICE_2', 'BID_SIZE_2', ..., 'BID_PRICE_11', 'BID_SIZE_11', 'ASK_PRICE_1',
    #    'ASK_SIZE_1', 'ASK_PRICE_2', 'ASK_SIZE_2', ..., 'ASK_PRICE_11', 'ASK_SIZE_11']
    def process_data(self, X) -> pd.DataFrame:
        # Convert list to DataFrame
        columns = ['COLLECTION_TIME']
        for i in range(1, 12):
            columns.extend([f'BID_PRICE_{i}', f'BID_SIZE_{i}', f'ASK_PRICE_{i}', f'ASK_SIZE_{i}'])
        
        # Init the df and caculate the mid_price, spread, total_bid_size, total_ask_size, ob_imbalance
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
        
        # Resample the data and calculate the candlestick df
        candlestick_df = pd.DataFrame()
        candlestick_df['open'] = df['mid_price'].resample(self.resample_interval).first()
        candlestick_df['high'] = df['mid_price'].resample(self.resample_interval).max()
        candlestick_df['low'] = df['mid_price'].resample(self.resample_interval).min()
        candlestick_df['close'] = df['mid_price'].resample(self.resample_interval).last()
        candlestick_df['avg_total_size'] = (df['total_bid_size'] + df['total_ask_size']).resample(self.resample_interval).mean()
        candlestick_df['spread'] = df['spread'].resample(self.resample_interval).mean()
        candlestick_df['ob_imbalance'] = df['ob_imbalance'].resample(self.resample_interval).mean()
        candlestick_df.fillna(method='ffill', inplace=True)
        
        # Concate the new candlestick_df with the total_candlestick_df
        candlestick_df.reset_index(inplace=True)
        if self.total_candlestick_df is not None:
            # Concatenate the historical data with the new candlestick data
            if candlestick_df['COLLECTION_TIME'].iloc[0] == self.total_candlestick_df['COLLECTION_TIME'].iloc[-1]:
                candlestick_df = candlestick_df.iloc[1:]
            self.total_candlestick_df = pd.concat([self.total_candlestick_df, candlestick_df])
        else:
            # If no historical data exists, use the new candlestick data directly
            self.total_candlestick_df = candlestick_df
        
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
    
    def update_model(self):
        # If there is no data, no need to update the model
        if self.total_candlestick_df is None:
            return
        
        # Since self.total_candlestick_df is actively updated with new data, we can use the latest data to update the model
        param_grid = param_grid = {
            'n_estimators': [500, 1000],
            'max_depth': [9, 15, 20],
            'learning_rate': [0.01, 0.02, 0.04],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.8],
            'gamma': [0.1],
            'reg_lambda': [1, 1.5, 2],
        }
        
        df = self.total_candlestick_df.copy()
        df.dropna(inplace=True)
        X, y = df[self.feature_columns], df['future_close']
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

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
    
    def predict(self, X):
        df = self.process_data(X)
        y = self.model.predict(df[self.feature_columns])
        return y
    
    
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
    