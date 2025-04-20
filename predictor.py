import logging
import yfinance as yf
from datetime import datetime, timedelta
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
import warnings
import threading
import time
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ShortTermPredictor:
    def __init__(self, symbol, minutes):
        self.symbol = symbol
        self.minutes = minutes
        # Modified pipeline with GradientBoostingClassifier
        self.pipeline = Pipeline([
            ('smote', SMOTE(random_state=42, k_neighbors=5)),
            ('scaler', RobustScaler()),
            ('classifier', GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.06,
                max_depth=6,
                subsample=0.8,
                random_state=42,
                min_samples_split=50,
                min_samples_leaf=20,
                max_features=0.7,
                validation_fraction=0.1,
                n_iter_no_change=10,
                tol=1e-4
            ))
        ])
        self.feature_columns = None
        self.important_features = None
        self.HIGH_CONFIDENCE_THRESHOLD = 0.70
        self.MEDIUM_CONFIDENCE_THRESHOLD = 0.65
        self.LOW_CONFIDENCE_THRESHOLD = 0.55
        # New attributes to store average price changes
        self.avg_change_up = None
        self.avg_change_down = None

        # Fallback values for when prediction fails
        self.fallback_model = None
        self.fallback_features = None

        # Set minimum data requirements
        self.min_data_points = 30  # Absolute minimum needed data points
        self.ideal_data_points = 100  # Ideal number of data points

        # Retry parameters
        self.max_retries = 3
        self.retry_delay = 5  # seconds

    def safe_download(self, symbol, start, end, interval):
        """Safely download data with retries and fallbacks"""
        retries = 0
        while retries < self.max_retries:
            try:
                df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)

                # Check if we have enough data
                if len(df) >= self.min_data_points:
                    return df
                else:
                    logging.warning(
                        f"Downloaded only {len(df)} rows for {symbol}, which is below minimum requirement of {self.min_data_points}")

                    # Try extending the time period
                    new_start = start - timedelta(days=max(3, (end - start).days))
                    logging.info(f"Extending time period from {start} to {new_start} to get more data")
                    start = new_start
                    retries += 1
                    time.sleep(self.retry_delay)
            except Exception as e:
                logging.error(f"Error downloading data for {symbol}: {str(e)}")
                retries += 1
                time.sleep(self.retry_delay)

        # If we still couldn't get enough data, raise exception
        raise ValueError(f"Failed to download sufficient data for {symbol} after {self.max_retries} attempts")

    def prepare_features(self, df):
        try:
            # Handle empty dataframe
            if df.empty or len(df) < self.min_data_points:
                raise ValueError(f"Insufficient data points: {len(df)}")

            # Create a DataFrame with single-level columns
            data = pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if (col, self.symbol) in df.columns:
                        data[col] = df[(col, self.symbol)]
                    else:
                        # Handle missing columns
                        logging.warning(f"Missing column: {col} for {self.symbol}, using fallback")
                        # Use fallback values like Close for missing columns
                        if col == 'Volume':
                            data[col] = 0
                        elif (col in ['High', 'Low', 'Open']) and ('Close', self.symbol) in df.columns:
                            data[col] = df[('Close', self.symbol)]
                        else:
                            raise ValueError(f"Critical data column missing: {col}")
            else:
                # Check for required columns
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                missing_columns = [col for col in required_columns if col not in df.columns]

                if missing_columns:
                    # Try to fill in missing columns with reasonable values
                    for col in missing_columns:
                        if col == 'Volume':
                            df[col] = 0
                        elif col in ['High', 'Low', 'Open'] and 'Close' in df.columns:
                            df[col] = df['Close']
                        else:
                            raise ValueError(f"Critical data column missing: {col}")

                data = df.copy()

            # Handle NaN values in data
            for col in data.columns:
                if data[col].isna().any():
                    logging.warning(f"NaN values found in {col}, using forward fill")
                    data[col] = data[col].fillna(method='ffill')
                    # If still have NaNs at beginning, backward fill
                    data[col] = data[col].fillna(method='bfill')
                    # If somehow still have NaNs, use median
                    data[col] = data[col].fillna(data[col].median() if len(data[col].dropna()) > 0 else 0)

            # Basic features that we always need
            data['returns'] = data['Close'].pct_change()
            data['vol'] = data['returns'].rolling(window=min(5, len(data) // 2)).std()
            data['momentum'] = data['Close'] - data['Close'].shift(min(5, len(data) // 4))

            # Simple moving averages with adaptive window sizes
            window_size_small = min(5, max(2, len(data) // 10))
            window_size_med = min(10, max(3, len(data) // 8))
            window_size_large = min(20, max(5, len(data) // 5))

            data['SMA5'] = data['Close'].rolling(window=window_size_small, min_periods=1).mean()
            data['SMA10'] = data['Close'].rolling(window=window_size_med, min_periods=1).mean()
            data['SMA20'] = data['Close'].rolling(window=window_size_large, min_periods=1).mean()

            # Exponential moving averages
            span_small = min(12, max(3, len(data) // 6))
            span_large = min(26, max(6, len(data) // 4))

            data['EMA12'] = data['Close'].ewm(span=span_small, adjust=False, min_periods=1).mean()
            data['EMA26'] = data['Close'].ewm(span=span_large, adjust=False, min_periods=1).mean()

            # Crossovers
            data['MA_crossover'] = data['SMA5'] - data['SMA10']
            data['EMA_crossover'] = data['EMA12'] - data['EMA26']

            # Price ranges
            data['high_low_range'] = data['High'] - data['Low']
            data['close_open_range'] = data['Close'] - data['Open']

            # Average True Range with adaptive window
            atr_window = min(14, max(3, len(data) // 6))
            data['ATR'] = data['high_low_range'].rolling(window=atr_window, min_periods=1).mean()

            # Volume indicators - handling zero volume
            data['Volume'] = data['Volume'].replace(0, 1)  # Replace zeros with ones to avoid division by zero
            volume_ma5 = data['Volume'].rolling(window=window_size_small, min_periods=1).mean()
            volume_ma10 = data['Volume'].rolling(window=window_size_med, min_periods=1).mean()

            # Add a small value to avoid division by zero
            epsilon = 1.0e-10
            data['volume_ratio'] = data['Volume'].div(volume_ma5 + epsilon).clip(upper=10)
            data['volume_ratio_10'] = data['Volume'].div(volume_ma10 + epsilon).clip(upper=10)

            # RSI calculation with safeguards and adaptive windows
            for window in [7, 14, 21]:
                window = min(window, max(3, len(data) // 4))  # Adaptive window size
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
                rs = gain / (loss + epsilon)  # Add epsilon to avoid division by zero
                data[f'RSI_{window}'] = 100 - (100 / (1 + rs))

            # Bollinger bands with adaptive window
            bb_window = min(20, max(5, len(data) // 5))
            data['bollinger_mid'] = data['Close'].rolling(window=bb_window, min_periods=1).mean()
            data['bollinger_std'] = data['Close'].rolling(window=bb_window, min_periods=1).std()
            data['bollinger_upper'] = data['bollinger_mid'] + (data['bollinger_std'] * 2)
            data['bollinger_lower'] = data['bollinger_mid'] - (data['bollinger_std'] * 2)
            data['bollinger_width'] = data['bollinger_upper'] - data['bollinger_lower']

            # MACD
            data['MACD'] = data['EMA12'] - data['EMA26']
            data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
            data['MACD_hist'] = data['MACD'] - data['MACD_signal']

            # Stochastic oscillator with adaptive window
            stoch_window = min(14, max(3, len(data) // 6))
            data['%K'] = ((data['Close'] - data['Low'].rolling(window=stoch_window, min_periods=1).min()) /
                          (data['High'].rolling(window=stoch_window, min_periods=1).max() -
                           data['Low'].rolling(window=stoch_window, min_periods=1).min() + epsilon)) * 100
            data['%D'] = data['%K'].rolling(window=min(3, max(2, len(data) // 20)), min_periods=1).mean()

            # Rate of Change with adaptive period
            roc_period = min(12, max(3, len(data) // 7))
            data['ROC'] = data['Close'].pct_change(periods=roc_period) * 100

            # On Balance Volume
            data['OBV'] = (data['Volume'] * (~data['Close'].diff().le(0) * 2 - 1)).cumsum()

            # Binary features
            data['open_equals_low'] = (data['Open'] == data['Low']).astype(int)
            data['open_equals_high'] = (data['Open'] == data['High']).astype(int)

            # Volume Price Trend
            data['VPT'] = (data['Volume'] * data['returns']).cumsum()

            # Chaikin Money Flow with adaptive window
            cmf_window = min(20, max(5, len(data) // 5))
            money_flow_volume = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (
                    data['High'] - data['Low'] + epsilon) * data['Volume']
            data['CMF'] = money_flow_volume.rolling(window=cmf_window, min_periods=1).sum() / (
                        data['Volume'].rolling(window=cmf_window, min_periods=1).sum() + epsilon)

            # Combined indicators
            data['RSI_vol_ratio'] = data['RSI_14'] * data['volume_ratio']
            data['momentum_volume'] = data['momentum'] * data['volume_ratio']
            data['bollinger_momentum'] = data['bollinger_width'] * data['momentum']

            # Target variable - Price going up in the next interval
            data['target'] = (data['Close'].shift(-self.minutes) > data['Close']).astype(int)

            # Lag features - only include if we have enough data
            if len(data) > 10:
                for lag in range(1, min(4, len(data) // 10)):
                    data[f'returns_lag{lag}'] = data['returns'].shift(lag)
                    data[f'vol_lag{lag}'] = data['vol'].shift(lag)
                    data[f'RSI_14_lag{lag}'] = data['RSI_14'].shift(lag)
                    data[f'MACD_lag{lag}'] = data['MACD'].shift(lag)
                    data[f'volume_ratio_lag{lag}'] = data['volume_ratio'].shift(lag)

            # Drop rows with NaN values
            data = data.dropna()

            if len(data) < 20:
                logging.warning(f"After dropping NaN values, only {len(data)} rows remain. This may be insufficient.")
                if len(data) < 10:
                    raise ValueError(f"Insufficient data points after preprocessing: {len(data)}")

            # Define feature columns based on what was actually calculated
            self.feature_columns = [
                'returns', 'vol', 'momentum', 'MA_crossover', 'EMA_crossover',
                'high_low_range', 'close_open_range', 'ATR', 'volume_ratio', 'volume_ratio_10',
                'RSI_7', 'RSI_14', 'RSI_21', 'bollinger_width', 'MACD',
                'MACD_signal', 'MACD_hist', '%K', '%D', 'ROC',
                'OBV', 'open_equals_low', 'open_equals_high', 'VPT', 'CMF',
                'RSI_vol_ratio', 'momentum_volume', 'bollinger_momentum'
            ]

            # Add lag features if they exist
            for lag in range(1, 4):
                if f'returns_lag{lag}' in data.columns:
                    self.feature_columns.extend([
                        f'returns_lag{lag}', f'vol_lag{lag}', f'RSI_14_lag{lag}',
                        f'MACD_lag{lag}', f'volume_ratio_lag{lag}'
                    ])

            # Only use features that actually exist in the data
            self.feature_columns = [col for col in self.feature_columns if col in data.columns]

            features = data[self.feature_columns]
            target = data['target']

            # Log successful preparation
            logging.info(f"Successfully prepared features. Shape: {features.shape}")

            return features, target

        except Exception as e:
            logging.error(f"Error in feature preparation: {str(e)}")
            logging.error(traceback.format_exc())
            raise

    def select_features(self, X, y, max_features=None):
        try:
            # Limit the number of features to select based on data size
            if max_features is None:
                max_features = min(30, max(5, len(X.columns) // 2))

            # Use a simpler model for feature selection if we have limited data
            if len(X) < 50:
                n_estimators = min(50, max(10, len(X) // 2))
                max_depth = min(3, max(1, len(X) // 20))
            else:
                n_estimators = 100
                max_depth = 4

            feature_selector = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=0.05,
                max_depth=max_depth,
                subsample=0.8,
                random_state=42
            )

            # Handle case where we have few samples
            if len(X) < 20 or len(np.unique(y)) < 2:
                logging.warning("Not enough diverse samples for feature selection, using all features")
                return X.columns.tolist()

            feature_selector.fit(X, y)
            importances = feature_selector.feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)

            # Select top features, but no more than our calculated maximum
            top_features_count = min(max_features, len(X.columns))
            top_features = feature_importance['Feature'][:top_features_count].tolist()

            logging.info(f"Selected {len(top_features)} features based on importance")
            logging.info(f"Top 10 features: {top_features[:min(10, len(top_features))]}")

            return top_features
        except Exception as e:
            logging.error(f"Error in feature selection: {str(e)}")
            logging.error(traceback.format_exc())
            # Return all features if selection fails
            return X.columns.tolist()

    def train_model(self, hyperparameter_optimization=False):
        try:
            end = datetime.now()
            num_intervals_train = 1000

            # Calculate days needed based on interval size, with safeguards
            minutes_per_day = 1440  # 1440 minutes in a day
            days = (num_intervals_train * self.minutes) / minutes_per_day

            # Adjust days based on interval size
            if self.minutes <= 5:
                days = max(7, days)  # Keep at least 7 days for 5-minute intervals
            elif self.minutes <= 60:
                days = max(14, days)  # Keep at least 14 days for hourly data
            else:
                days = max(30, days)  # Keep at least 30 days for larger intervals

            # For crypto, sometimes we need more data
            if 'BTC' in self.symbol or 'ETH' in self.symbol:
                days *= 1.5  # Increase data for major cryptos

            # For newer/smaller cryptos, we may need to go back further
            if any(token in self.symbol for token in ['SHIB', 'DOGE', 'AVAX', 'SOL']):
                days *= 2  # Double the days for newer tokens

            start = end - timedelta(days=days)

            # Try to download data with increased safety measures
            try:
                df = self.safe_download(self.symbol, start=start, end=end, interval=f'{self.minutes}m')
                logging.info(f"Downloaded {len(df)} rows of data for {self.symbol}")
            except ValueError as e:
                logging.error(f"Failed to download data: {str(e)}")

                # Try with a different interval as fallback
                fallback_minutes = max(30, self.minutes * 2)  # Use a larger interval
                fallback_start = end - timedelta(days=days * 2)  # Go back further in time
                logging.info(f"Trying fallback: {fallback_minutes}m interval from {fallback_start}")

                df = self.safe_download(self.symbol, start=fallback_start, end=end, interval=f'{fallback_minutes}m')
                logging.info(f"Downloaded {len(df)} rows of fallback data for {self.symbol}")

                # Adjust minutes parameter to match what we got
                self.minutes = fallback_minutes

            # Prepare features with retry mechanism
            try:
                X, y = self.prepare_features(df)
            except Exception as e:
                logging.error(f"Error preparing features: {str(e)}")

                # Try with simplified feature preparation
                logging.info("Trying with simplified feature preparation")
                try:
                    # Create a simplified DataFrame
                    data = pd.DataFrame()
                    if isinstance(df.columns, pd.MultiIndex):
                        data['Close'] = df[('Close', self.symbol)]
                        data['Volume'] = df[('Volume', self.symbol)] if ('Volume', self.symbol) in df.columns else 0
                    else:
                        data['Close'] = df['Close']
                        data['Volume'] = df['Volume'] if 'Volume' in df.columns else 0

                    # Create minimal features
                    data['returns'] = data['Close'].pct_change()
                    data['vol'] = data['returns'].rolling(window=min(5, len(data) // 2), min_periods=1).std()
                    data['SMA5'] = data['Close'].rolling(window=min(5, len(data) // 4), min_periods=1).mean()
                    data['SMA10'] = data['Close'].rolling(window=min(10, len(data) // 3), min_periods=1).mean()
                    data['target'] = (data['Close'].shift(-self.minutes) > data['Close']).astype(int)

                    # Remove NaN values
                    data = data.dropna()

                    X = data[['returns', 'vol', 'SMA5', 'SMA10']]
                    y = data['target']
                    logging.info(f"Created simplified features with shape: {X.shape}")
                except Exception as inner_e:
                    logging.error(f"Even simplified feature creation failed: {str(inner_e)}")
                    raise ValueError("Unable to create features from available data")

            # Check if we have enough training data
            if len(X) < 20:
                raise ValueError(f"Insufficient training data: {len(X)} rows")

            # Feature selection with safeguards
            try:
                self.important_features = self.select_features(X, y)
                X_selected = X[self.important_features]
            except Exception as e:
                logging.error(f"Feature selection failed: {str(e)}")
                logging.info("Using all features as fallback")
                self.important_features = X.columns.tolist()
                X_selected = X

            # Check class balance and adjust parameters if needed
            class_counts = y.value_counts()
            logging.info(f"Class distribution: {class_counts}")

            # If we have extreme imbalance or few samples of one class, adjust SMOTE
            imbalance_ratio = class_counts.max() / class_counts.min() if class_counts.min() > 0 else float('inf')
            min_class_samples = class_counts.min()

            if imbalance_ratio > 3 or min_class_samples < 10:
                logging.warning(
                    f"Class imbalance detected: ratio={imbalance_ratio:.2f}, min samples={min_class_samples}")

                if min_class_samples < 5:
                    # Too few samples for SMOTE, use weights instead
                    logging.info("Too few samples for SMOTE, using class_weight instead")
                    self.pipeline.steps[0] = ('passthrough', None)
                    self.pipeline.steps[2] = ('classifier', GradientBoostingClassifier(
                        n_estimators=100,
                        learning_rate=0.06,
                        max_depth=4,  # Reduce complexity
                        subsample=0.8,
                        random_state=42,
                        min_samples_split=20,
                        min_samples_leaf=5,
                        max_features=0.5
                    ))
                else:
                    # Adjust SMOTE parameters
                    k_neighbors = min(min_class_samples - 1, 3)
                    sampling_strategy = min(0.8, (min_class_samples / class_counts.max()) * 2)
                    logging.info(
                        f"Adjusting SMOTE: k_neighbors={k_neighbors}, sampling_strategy={sampling_strategy:.2f}")
                    self.pipeline.steps[0] = ('smote', SMOTE(
                        random_state=42,
                        k_neighbors=k_neighbors,
                        sampling_strategy=sampling_strategy
                    ))

            # Initialize cross-validation with fewer splits if we have limited data
            n_splits = min(4, max(2, len(X) // 50))
            tscv = TimeSeriesSplit(n_splits=n_splits)

            # Store data before potential modification as fallback
            self.fallback_features = X_selected.iloc[-1:].copy()

            # Train the model with simpler settings if we have limited data
            if len(X) < 50:
                logging.info("Limited data detected, simplifying the model")
                self.pipeline.steps[2] = ('classifier', GradientBoostingClassifier(
                    n_estimators=50,
                    learning_rate=0.1,
                    max_depth=3,
                    subsample=0.9,
                    random_state=42,
                    min_samples_split=5,
                    min_samples_leaf=2
                ))

            # Train with safeguards
            try:
                for train_index, test_index in tscv.split(X_selected):
                    X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                    # Only fit if we have enough data
                    if len(X_train) >= 20 and len(np.unique(y_train)) > 1:
                        self.pipeline.fit(X_train, y_train)
                        y_pred = self.pipeline.predict(X_test)
                        logging.info("\nModel Performance:")
                        logging.info(classification_report(y_test, y_pred))
                    else:
                        logging.warning("Skipping fold due to insufficient or imbalanced data")

                # Final fit on all data
                if len(X_selected) >= 20 and len(np.unique(y)) > 1:
                    self.pipeline.fit(X_selected, y)
                    logging.info("Final model fit completed")
                else:
                    # Fallback to a very simple model
                    logging.warning("Insufficient data for full model, creating fallback classifier")
                    self.pipeline = Pipeline([
                        ('scaler', RobustScaler()),
                        ('classifier', GradientBoostingClassifier(
                            n_estimators=20,
                            max_depth=2,
                            learning_rate=0.1
                        ))
                    ])
                    self.pipeline.fit(X_selected, y)

                # Store the model for potential reuse
                self.fallback_model = self.pipeline

                # Calculate average percentage price changes for each class with robust error handling
                try:
                    future_close = df['Close'].shift(-1).loc[X.index] if isinstance(df.columns, pd.Series) else df.loc[
                        X.index, 'Close'].shift(-1)
                    current_close = df['Close'].loc[X.index] if isinstance(df.columns, pd.Series) else df.loc[
                        X.index, 'Close']

                    price_change = (future_close - current_close) / current_close
                    up_changes = price_change[y == 1]
                    down_changes = price_change[y == 0]

                    # Calculate means with safeguards
                    self.avg_change_up = float(
                        up_changes.mean()) if not up_changes.empty else 0.003  # Default to 0.3% if no data
                    self.avg_change_down = float(
                        down_changes.mean()) if not down_changes.empty else -0.003  # Default to -0.3% if no data

                    # Handle potential NaN values
                    if pd.isna(self.avg_change_up) or abs(self.avg_change_up) > 0.1:
                        self.avg_change_up = 0.003  # Default to 0.3% if NaN or extreme value
                    if pd.isna(self.avg_change_down) or abs(self.avg_change_down) > 0.1:
                        self.avg_change_down = -0.003  # Default to -0.3% if NaN or extreme value

                    logging.info(
                        f"Average % change (up): {self.avg_change_up:.4f}, (down/no change): {self.avg_change_down:.4f}")
                except Exception as e:
                    logging.error(f"Error calculating price changes: {str(e)}")
                    self.avg_change_up = 0.003  # Default to 0.3%
                    self.avg_change_down = -0.003  # Default to -0.3%

                return self.pipeline

            except Exception as e:
                logging.error(f"Error during model training: {str(e)}")
                logging.error(traceback.format_exc())

                # Create a fallback model
                logging.info("Creating minimal fallback model")
                self.pipeline = Pipeline([
                    ('scaler', RobustScaler()),
                    ('classifier', GradientBoostingClassifier(
                        n_estimators=10,
                        max_depth=1
                    ))
                ])

                # Try to fit with just a few basic features
                basic_features = ['returns', 'vol'] if 'returns' in X.columns and 'vol' in X.columns else X.columns[:2]
                self.important_features = basic_features
                self.pipeline.fit(X[basic_features], y)

                # Set default price change expectations
                self.avg_change_up = 0.003  # Default to 0.3%
                self.avg_change_down = -0.003  # Default to -0.3%

                return self.pipeline

        except Exception as e:
            logging.error(f"Critical error in model training: {str(e)}")
            logging.error(traceback.format_exc())
            raise

    def predict_next_movement(self):
        try:
            end = datetime.now()
            start_time = time.time()  # Track execution time

            # Calculate how far back we need to go to get sufficient data
            num_intervals_pred = max(100, 5 * max(21, self.minutes))  # Enough to compute features
            start = end - timedelta(minutes=self.minutes * num_intervals_pred)

            # Attempt to get recent data with retries
            try:
                recent_data = self.safe_download(self.symbol, start=start, end=end, interval=f'{self.minutes}m')
                logging.info(f"Downloaded {len(recent_data)} rows of recent data for prediction")

                if recent_data.empty or len(recent_data) < self.min_data_points:
                    raise ValueError(f"Insufficient data points: {len(recent_data) if not recent_data.empty else 0}")

            except Exception as e:
                logging.error(f"Failed to download recent data: {str(e)}")

                # Try with a larger interval as fallback
                fallback_minutes = min(60, self.minutes * 2)  # Use a larger interval, but max 1 hour
                fallback_start = end - timedelta(days=5)  # Go back further in time

                logging.info(f"Trying fallback download: {fallback_minutes}m interval from {fallback_start}")
                recent_data = self.safe_download(self.symbol, start=fallback_start, end=end,
                                                 interval=f'{fallback_minutes}m')

                if recent_data.empty or len(recent_data) < self.min_data_points:
                    raise ValueError("Even fallback data retrieval failed to get enough data points")

            # Prepare features with error handling
            try:
                X, _ = self.prepare_features(recent_data)
                X_full = X.copy()  # Keep a copy for market indicators

                if self.important_features:
                    # Filter to only use the important features that we trained on
                    available_features = [f for f in self.important_features if f in X.columns]

                    if len(available_features) < len(self.important_features) * 0.7:
                        logging.warning(
                            f"Missing {len(self.important_features) - len(available_features)} important features")

                    if not available_features:
                        raise ValueError("No important features available in the prediction data")

                    X_pred = X[available_features]
                else:
                    # If no important features specified, use all
                    X_pred = X

                if len(X_pred) == 0:
                    raise ValueError("No valid rows in feature data")

            except Exception as e:
                logging.error(f"Error preparing prediction features: {str(e)}")

                if self.fallback_features is not None and self.fallback_model is not None:
                    logging.info("Using fallback features and model")
                    X_pred = self.fallback_features
                    X_full = X_pred.copy()
                    self.pipeline = self.fallback_model
                else:
                    # Create extremely simple features as last resort
                    logging.info("Creating minimal features for prediction")
                    data = pd.DataFrame()

                    if isinstance(recent_data.columns, pd.MultiIndex):
                        data['Close'] = recent_data[('Close', self.symbol)]
                        data['Volume'] = recent_data[('Volume', self.symbol)] if ('Volume',
                                                                                  self.symbol) in recent_data.columns else 1
                    else:
                        data['Close'] = recent_data['Close']
                        data['Volume'] = recent_data['Volume'] if 'Volume' in recent_data.columns else 1

                    # Create minimal features
                    data['returns'] = data['Close'].pct_change().fillna(0)
                    data['vol'] = data['returns'].rolling(window=min(5, len(data) // 2), min_periods=1).std()

                    # Filter out NaN values
                    data = data.fillna(method='ffill').fillna(0)

                    X_pred = data[['returns', 'vol']].iloc[-1:]
                    X_full = X_pred.copy()

            # Keep as DataFrame to preserve feature names
            latest_features = X_pred.iloc[-1:].copy()  # Get the most recent data point

            # Predict with error handling
            try:
                probabilities = self.pipeline.predict_proba(latest_features)[0]
                prediction = 1 if probabilities[1] >= 0.5 else 0
            except Exception as e:
                logging.error(f"Prediction failed: {str(e)}")
                # Default to a balanced prediction
                probabilities = np.array([0.5, 0.5])
                prediction = 1  # Slightly optimistic default

            # Get current price with safeguards
            try:
                if isinstance(recent_data.columns, pd.MultiIndex):
                    current_price = float(recent_data[('Close', self.symbol)].iloc[-1])
                else:
                    current_price = float(recent_data['Close'].iloc[-1])
            except (IndexError, KeyError) as e:
                logging.error(f"Error getting current price: {str(e)}")
                # Use last known price or a default
                current_price = 100.0  # This is just a placeholder

            # Calculate market indicators if available
            try:
                if 'ATR' in X_full.columns and len(X_full) > 20:
                    market_volatility = X_full['ATR'].iloc[-5:].mean() / X_full['ATR'].iloc[-20:].mean()
                else:
                    # Calculate simple volatility measure
                    if isinstance(recent_data.columns, pd.MultiIndex):
                        closes = recent_data[('Close', self.symbol)]
                    else:
                        closes = recent_data['Close']

                    recent_std = closes.iloc[-5:].std() / closes.iloc[-5:].mean()
                    older_std = closes.iloc[-20:].std() / closes.iloc[-20:].mean()
                    market_volatility = recent_std / older_std if older_std > 0 else 1.0

                if 'returns' in X_full.columns and len(X_full) > 5:
                    recent_trend = X_full['returns'].iloc[-5:].mean()
                else:
                    if isinstance(recent_data.columns, pd.MultiIndex):
                        closes = recent_data[('Close', self.symbol)]
                    else:
                        closes = recent_data['Close']

                    if len(closes) > 5:
                        recent_trend = (closes.iloc[-1] / closes.iloc[-5] - 1) / 5
                    else:
                        recent_trend = 0

                # Adjust confidence based on market conditions
                confidence_adjustment = 0

                if np.isnan(market_volatility):
                    market_volatility = 1.0

                if market_volatility > 1.5:
                    confidence_adjustment -= min(0.15, (market_volatility - 1.5) * 0.1)
                    logging.info(f"High market volatility detected ({market_volatility:.2f}), reducing confidence")

                if np.isnan(recent_trend):
                    recent_trend = 0

                if (prediction == 1 and recent_trend < -0.01) or (prediction == 0 and recent_trend > 0.01):
                    confidence_adjustment -= 0.05
                    logging.info(f"Prediction conflicts with recent trend ({recent_trend:.4f}), reducing confidence")

                # Adjust probabilities based on our confidence adjustment
                adjusted_probabilities = probabilities.copy()
                if confidence_adjustment != 0:
                    if prediction == 1:
                        adjusted_probabilities[1] = max(0.5, min(0.95, probabilities[1] + confidence_adjustment))
                        adjusted_probabilities[0] = 1 - adjusted_probabilities[1]
                    else:
                        adjusted_probabilities[0] = max(0.5, min(0.95, probabilities[0] + confidence_adjustment))
                        adjusted_probabilities[1] = 1 - adjusted_probabilities[0]
            except Exception as e:
                logging.error(f"Error calculating market indicators: {str(e)}")
                market_volatility = 1.0
                recent_trend = 0
                adjusted_probabilities = probabilities.copy()

            # Determine confidence level
            max_prob = max(adjusted_probabilities)
            if max_prob >= self.HIGH_CONFIDENCE_THRESHOLD:
                confidence_level = "High"
            elif max_prob >= self.MEDIUM_CONFIDENCE_THRESHOLD:
                confidence_level = "Medium"
            elif max_prob >= self.LOW_CONFIDENCE_THRESHOLD:
                confidence_level = "Low"
            else:
                confidence_level = "Very Low"

            # If the probabilities are close, reduce confidence
            if abs(adjusted_probabilities[0] - adjusted_probabilities[1]) < 0.1:
                confidence_level = "Very Low"
                logging.info("Prediction uncertainty is high, confidence set to Very Low")

            # Calculate estimated future price with proper error handling
            try:
                # Make sure we have valid values for price change estimates
                if self.avg_change_up is None or pd.isna(self.avg_change_up):
                    self.avg_change_up = 0.003  # Default to 0.3%
                if self.avg_change_down is None or pd.isna(self.avg_change_down):
                    self.avg_change_down = -0.003  # Default to -0.3%

                # Ensure they're in a reasonable range
                self.avg_change_up = min(0.1, max(0.0001, self.avg_change_up))
                self.avg_change_down = max(-0.1, min(-0.0001, self.avg_change_down))

                prob_up = adjusted_probabilities[1]
                prob_down = adjusted_probabilities[0]
                expected_change = prob_up * self.avg_change_up + prob_down * self.avg_change_down
                estimated_future_price = float(current_price * (1 + expected_change))
            except Exception as e:
                logging.error(f"Error calculating future price: {str(e)}")
                # Just estimate a small change in the predicted direction
                if prediction == 1:
                    estimated_future_price = current_price * 1.005  # 0.5% increase
                else:
                    estimated_future_price = current_price * 0.995  # 0.5% decrease

            # Log execution time
            execution_time = time.time() - start_time
            logging.info(f"Prediction completed in {execution_time:.2f} seconds")

            # Log prediction details
            logging.info(f"Prediction: {'Up' if prediction == 1 else 'Down'}, "
                         f"Probabilities: {probabilities}, Adjusted: {adjusted_probabilities}, "
                         f"Confidence: {confidence_level}, "
                         f"Current Price: {current_price:.6f}, "
                         f"Estimated Future Price: {estimated_future_price:.6f}")

            return prediction, adjusted_probabilities, current_price, datetime.now(), confidence_level, estimated_future_price

        except Exception as e:
            logging.error(f"Critical error in prediction: {str(e)}")
            logging.error(traceback.format_exc())

            # Return fail-safe values
            return 1, np.array([0.4, 0.6]), 0.0, datetime.now(), "Error", 0.0
        
# def get_current_price(symbol, retries=3, delay=60):
#     ticker = yf.Ticker(symbol)
#     for attempt in range(retries):
#         try:
#             current_data = ticker.history(period='1d', interval='1m')
#             if current_data.empty:
#                 raise ValueError("No current data available")
#             return float(current_data['Close'].iloc[-1])
#         except Exception as e:
#             if "Rate limited" in str(e) or "Too Many Requests" in str(e):
#                 logging.error(f"{symbol} is rate limited. Waiting for {delay} seconds before retrying...")
#                 time.sleep(delay)
#             else:
#                 logging.error(f"Error checking price for {symbol}: {e}")
#                 raise
#     raise Exception(f"Failed to get current price for {symbol} after {retries} retries.")
#
#
# def run_prediction(symbol, minutes, results, lock):
#     try:
#         logging.info(f"\n=== Starting prediction for {symbol} ===")
#         predictor = ShortTermPredictor(symbol, minutes)
#
#         logging.info(f"Training model for {symbol}...")
#         predictor.train_model()
#         logging.info(f"Training completed for {symbol}.")
#
#         pred, probs, last_close, pred_time, conf, future_price = predictor.predict_next_movement()
#
#
#         # Only trade on high confidence predictions
#         if conf != "High":
#             logging.info(f"{symbol}: Skipping due to insufficient confidence: {conf}")
#             return None
#
#         logging.info(f"Prediction for {symbol}: {'Up' if pred == 1 else 'Down'} at {pred_time} with confidence {conf}")
#         logging.info(f"Future price: {future_price:.6f}")
#         logging.info(f"{symbol}: Timer started - will check result in {minutes} minutes.")
#
#
#         def check_result():
#             # Wait the specified time before checking the price
#             time.sleep(minutes * 60)
#             current_price = get_current_price(symbol)
#             logging.info(f"{symbol}: Last Close: {last_close:.6f}, Estimated Price: {future_price:.6f}, Current Price: {current_price:.6f}")
#
#             # Calculate percentage change
#             pct_change = ((current_price - last_close) / last_close) * 100
#
#             # Determine if prediction was correct
#             correct = (pred == 1 and current_price > last_close) or (pred == 0 and current_price < last_close)
#
#             with lock:
#                 results.append((symbol, correct, pct_change, conf))
#
#             logging.info(f"{symbol}: Prediction was {'correct' if correct else 'incorrect'}. "
#                          f"Price change: {pct_change:.6f}%")
#
#         timer_thread = threading.Thread(target=check_result, daemon=True)
#         timer_thread.start()
#
#         return timer_thread
#
#     except Exception as e:
#         logging.error(f"Error in run_prediction for {symbol}: {e}")
#         return None
#
#
# def sequential_predictions():
#     # Focus on most liquid crypto assets
#     symbols = ['DOGE-USD', 'BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD',
#                'XRP-USD', 'DOT-USD', 'LTC-USD', 'BNB-USD', 'LINK-USD',
#                'MATIC-USD', 'SHIB-USD', 'AVAX-USD', 'TRX-USD', 'UNI-USD',
#                'XLM-USD', 'ATOM-USD', 'FIL-USD', 'NEAR-USD', 'ALGO-USD']
#
#     minutes = 5  # Prediction horizon (in minutes)
#     results = []  # To store (symbol, correct, pct_change, confidence) tuples
#     lock = threading.Lock()
#     timer_threads = []
#
#     # For each symbol, run prediction and immediately start the next one
#     for sym in symbols:
#         timer_thread = run_prediction(sym, minutes, results, lock)
#         if timer_thread is not None:
#             timer_threads.append(timer_thread)
#         # Add a small delay to avoid rate limiting
#
#     # Wait for all timer threads to finish
#     for t in timer_threads:
#         t.join()
#
#     # Analyze results
#     total_high_confidence = len(results)
#     if total_high_confidence > 0:
#         correct_predictions = sum(1 for r in results if r[1])
#         win_rate = (correct_predictions / total_high_confidence) * 100
#
#         # Calculate average percentage change for correct and incorrect predictions
#         correct_pct_changes = [r[2] for r in results if r[1]]
#         incorrect_pct_changes = [r[2] for r in results if not r[1]]
#
#         avg_correct_change = sum(correct_pct_changes) / len(correct_pct_changes) if correct_pct_changes else 0
#         avg_incorrect_change = sum(incorrect_pct_changes) / len(incorrect_pct_changes) if incorrect_pct_changes else 0
#
#         logging.info(f"\nResults Summary:")
#         logging.info(
#             f"Win Rate for High-Confidence Predictions: {win_rate:.2f}% based on {total_high_confidence} predictions.")
#         logging.info(f"Average % change on correct predictions: {avg_correct_change:.2f}%")
#         logging.info(f"Average % change on incorrect predictions: {avg_incorrect_change:.2f}%")
#
#         # Calculate profit factor
#         profit_factor = abs(sum(correct_pct_changes)) / abs(sum(incorrect_pct_changes)) if sum(
#             incorrect_pct_changes) != 0 else float('inf')
#         logging.info(f"Profit factor: {profit_factor:.2f}")
#     else:
#         logging.info("No high-confidence predictions were made.")
#
#
# def main():
#     logging.basicConfig(level=logging.INFO,
#                         format='%(asctime)s - %(levelname)s - %(message)s')
#     sequential_predictions()
#
#
# if __name__ == '__main__':
#     main()