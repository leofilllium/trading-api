# predictor.py

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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore', category=FutureWarning) # Suppress some common warnings

class ShortTermPredictor:
    def __init__(self, symbol, minutes):
        if not isinstance(minutes, int) or minutes <= 0:
            raise ValueError("Minutes must be a positive integer.")
        self.symbol = symbol
        self.minutes = minutes
        # Modified pipeline with GradientBoostingClassifier
        self.pipeline = Pipeline([
            ('smote', SMOTE(random_state=42, k_neighbors=5)), # k_neighbors might be adjusted in train_model
            ('scaler', RobustScaler()),
            ('classifier', GradientBoostingClassifier(
                n_estimators=100,       # Number of boosting stages
                learning_rate=0.06,     # Step size shrinkage
                max_depth=6,            # Max depth of individual trees
                subsample=0.8,          # Fraction of samples for fitting trees
                random_state=42,
                min_samples_split=50,   # Min samples required to split a node
                min_samples_leaf=20,    # Min samples required at a leaf node
                max_features=0.7,       # Fraction of features to consider for best split
                validation_fraction=0.1,# Fraction of training data for early stopping validation
                n_iter_no_change=10,    # Number of iterations with no improvement to stop training
                tol=1e-4                # Tolerance for stopping criterion
            ))
        ])
        self.feature_columns = None
        self.important_features = None
        self.HIGH_CONFIDENCE_THRESHOLD = 0.70
        self.MEDIUM_CONFIDENCE_THRESHOLD = 0.65
        self.LOW_CONFIDENCE_THRESHOLD = 0.55
        # Attributes to store average price changes
        self.avg_change_up = None
        self.avg_change_down = None
        self.base_predictions = None # Store predictions from CV for analysis


    def prepare_features(self, df):
        """
        Calculates technical indicators and prepares features for the model.
        Handles both single-symbol and multi-symbol (via MultiIndex) DataFrames from yfinance.
        """
        try:
            data = pd.DataFrame()
            # Standardize column access, handle MultiIndex if present
            if isinstance(df.columns, pd.MultiIndex):
                 # If MultiIndex, assume structure like ('Open', 'BTC-USD')
                 logging.debug("MultiIndex detected in DataFrame columns.")
                 for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                      if (col, self.symbol) in df.columns:
                           data[col] = df[(col, self.symbol)]
                      elif col in df.columns: # Fallback if only single level column name exists
                           data[col] = df[col]
                      else:
                           raise ValueError(f"Required column '{col}' not found for symbol '{self.symbol}' in MultiIndex.")
            else:
                # If single-level index, copy necessary columns
                logging.debug("Single-level index detected in DataFrame columns.")
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df.columns for col in required_cols):
                     raise ValueError(f"Missing one or more required columns: {required_cols}. Found: {df.columns.tolist()}")
                data = df[required_cols].copy()

            if data.isnull().values.any():
                logging.warning("NaN values detected in input data before feature engineering. Attempting to fill.")
                # Simple forward fill might be appropriate for time series price data
                data.ffill(inplace=True)
                # Drop any remaining NaNs at the beginning
                data.dropna(axis=0, how='any', inplace=True)
                if data.empty:
                     raise ValueError("DataFrame is empty after handling initial NaNs.")

            # --- Feature Engineering ---
            data['returns'] = data['Close'].pct_change()
            data['vol'] = data['returns'].rolling(window=5).std() * np.sqrt(252) # Annualized volatility (adjust factor if needed)
            data['momentum'] = data['Close'] - data['Close'].shift(5)
            data['SMA5'] = data['Close'].rolling(window=5).mean()
            data['SMA10'] = data['Close'].rolling(window=10).mean()
            data['SMA20'] = data['Close'].rolling(window=20).mean()
            data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
            data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
            data['MA_crossover'] = data['SMA5'] - data['SMA10']
            data['EMA_crossover'] = data['EMA12'] - data['EMA26']
            data['high_low_range'] = data['High'] - data['Low']
            data['close_open_range'] = data['Close'] - data['Open']
            data['ATR'] = data['high_low_range'].rolling(window=14).mean()

            volume_ma5 = data['Volume'].rolling(window=5).mean().replace(0, 1e-6) # Avoid division by zero
            volume_ma10 = data['Volume'].rolling(window=10).mean().replace(0, 1e-6)
            data['volume_ratio'] = data['Volume'].div(volume_ma5).clip(upper=10) # Clip extreme values
            data['volume_ratio_10'] = data['Volume'].div(volume_ma10).clip(upper=10)

            # RSI Calculation
            for window in [7, 14, 21]:
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
                rs = gain / loss.replace(0, 1e-6) # Avoid division by zero
                data[f'RSI_{window}'] = 100 - (100 / (1 + rs))
                data[f'RSI_{window}'].fillna(50, inplace=True) # Fill initial NaNs with neutral 50

            # Bollinger Bands
            data['bollinger_mid'] = data['Close'].rolling(window=20).mean()
            data['bollinger_std'] = data['Close'].rolling(window=20).std()
            data['bollinger_upper'] = data['bollinger_mid'] + (data['bollinger_std'] * 2)
            data['bollinger_lower'] = data['bollinger_mid'] - (data['bollinger_std'] * 2)
            data['bollinger_width'] = (data['bollinger_upper'] - data['bollinger_lower']) / data['bollinger_mid'].replace(0, 1e-6) # Normalized width

            # MACD
            data['MACD'] = data['EMA12'] - data['EMA26']
            data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data['MACD_hist'] = data['MACD'] - data['MACD_signal']

            # Stochastic Oscillator (%K, %D)
            low_min = data['Low'].rolling(window=14).min()
            high_max = data['High'].rolling(window=14).max()
            data['%K'] = 100 * ((data['Close'] - low_min) / (high_max - low_min).replace(0, 1e-6))
            data['%D'] = data['%K'].rolling(window=3).mean()
            data['%K'].fillna(50, inplace=True) # Handle potential NaNs
            data['%D'].fillna(50, inplace=True)

            # Rate of Change (ROC)
            data['ROC'] = data['Close'].pct_change(periods=12) * 100

            # On Balance Volume (OBV) - Simplified calculation
            data['OBV'] = (np.sign(data['Close'].diff()).fillna(0) * data['Volume']).cumsum()

            # Candlestick Patterns (simple examples)
            data['open_equals_low'] = (data['Open'] == data['Low']).astype(int)
            data['open_equals_high'] = (data['Open'] == data['High']).astype(int)

            # Volume Price Trend (VPT)
            data['VPT'] = (data['Volume'] * data['returns']).cumsum()

            # Chaikin Money Flow (CMF)
            mfm = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low']).replace(0, 1e-6)
            mfv = mfm * data['Volume']
            data['CMF'] = mfv.rolling(window=20).sum() / data['Volume'].rolling(window=20).sum().replace(0, 1e-6)
            data['CMF'].fillna(0, inplace=True) # Fill initial NaNs

            # Interaction Features
            data['RSI_vol_ratio'] = data['RSI_14'] * data['volume_ratio']
            data['momentum_volume'] = data['momentum'] * data['volume_ratio']
            data['bollinger_momentum'] = data['bollinger_width'] * data['momentum']

            # Target Variable: 1 if price goes up in `self.minutes`, 0 otherwise
            data['target'] = (data['Close'].shift(-self.minutes) > data['Close']).astype(int)

            # Lagged Features (use adjusted column names if needed)
            for lag in range(1, 4):
                data[f'returns_lag{lag}'] = data['returns'].shift(lag)
                data[f'vol_lag{lag}'] = data['vol'].shift(lag)
                data[f'RSI_14_lag{lag}'] = data['RSI_14'].shift(lag)
                data[f'MACD_lag{lag}'] = data['MACD'].shift(lag)
                data[f'volume_ratio_lag{lag}'] = data['volume_ratio'].shift(lag)

            # Drop rows with NaN values created by rolling windows, diffs, and shifts
            initial_rows = len(data)
            data = data.dropna()
            final_rows = len(data)
            logging.info(f"Feature engineering complete. Dropped {initial_rows - final_rows} rows with NaNs.")

            # Define feature columns *after* all features are created
            self.feature_columns = [
                'returns', 'vol', 'momentum', 'MA_crossover', 'EMA_crossover',
                'high_low_range', 'close_open_range', 'ATR', 'volume_ratio', 'volume_ratio_10',
                'RSI_7', 'RSI_14', 'RSI_21', 'bollinger_width', 'MACD', 'MACD_signal',
                'MACD_hist', '%K', '%D', 'ROC', 'OBV', 'open_equals_low', 'open_equals_high',
                'VPT', 'CMF', 'RSI_vol_ratio', 'momentum_volume', 'bollinger_momentum',
                'returns_lag1', 'returns_lag2', 'returns_lag3',
                'vol_lag1', 'vol_lag2', 'vol_lag3',
                'RSI_14_lag1', 'RSI_14_lag2', 'RSI_14_lag3',
                'MACD_lag1', 'MACD_lag2', 'MACD_lag3',
                'volume_ratio_lag1', 'volume_ratio_lag2', 'volume_ratio_lag3'
            ]

            # Ensure all feature columns exist in the dataframe
            missing_cols = [col for col in self.feature_columns if col not in data.columns]
            if missing_cols:
                raise RuntimeError(f"Internal error: Defined feature columns missing after creation: {missing_cols}")

            if final_rows < 50: # Check after dropping NaNs
                 raise ValueError(f"Insufficient data points ({final_rows}) after feature calculation for reliable prediction. Need at least 50.")

            features = data[self.feature_columns]
            target = data['target']

            # Replace any potential +/- infinity values with large finite numbers or NaN then fill
            features = features.replace([np.inf, -np.inf], np.nan)
            if features.isnull().values.any():
                 logging.warning("NaN or Inf values detected in features AFTER initial dropna. Filling with column median.")
                 # Impute NaNs that might have appeared due to edge cases (e.g., division by zero before replace)
                 features = features.fillna(features.median())


            logging.info(f"Successfully prepared features. Shape: {features.shape}")
            return features, target

        except KeyError as ke:
             logging.error(f"KeyError during feature preparation: Missing column {str(ke)}. Available columns: {df.columns.tolist()}")
             raise ValueError(f"Missing expected column in input data: {str(ke)}") from ke
        except Exception as e:
            logging.exception(f"Error in feature preparation: {str(e)}") # Log full traceback
            raise RuntimeError(f"Feature preparation failed: {str(e)}") from e


    def select_features(self, X, y):
        """Selects the most important features using Gradient Boosting."""
        try:
            # Simpler model for feature selection to avoid overfitting selection process
            feature_selector = GradientBoostingClassifier(
                n_estimators=50, learning_rate=0.1, max_depth=4,
                subsample=0.7, random_state=42
            )
            feature_selector.fit(X, y)
            importances = feature_selector.feature_importances_

            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)

            # Select top 60% of features, but ensure at least 10 features are selected
            top_features_count = max(10, int(len(X.columns) * 0.6))
            top_features = feature_importance['Feature'][:top_features_count].tolist()

            logging.info(f"Selected {len(top_features)} features based on importance.")
            logging.info(f"Top 10 features: {feature_importance['Feature'][:10].tolist()}")
            return top_features
        except Exception as e:
             logging.exception(f"Error during feature selection: {str(e)}")
             # Fallback: Return all features if selection fails
             logging.warning("Feature selection failed. Using all features.")
             return X.columns.tolist()


    def train_model(self, hyperparameter_optimization=False):
        """
        Downloads data, prepares features, selects features, and trains the model.
        Adjusts data download period based on the interval.
        """
        try:
            end = datetime.now()

            # --- Dynamic Data Fetch Period ---
            if self.minutes <= 5:
                days_to_fetch = 7   # Standard for 1m, 5m
            elif self.minutes <= 15:
                days_to_fetch = 25  # ~3-4 weeks for 15m
            elif self.minutes <= 30:
                days_to_fetch = 50  # ~7 weeks for 30m
            else: # Covers 60m and potentially larger intervals
                # yfinance may limit free intraday history (often 60 or 730 days depending on source/interval)
                # Request 90 days, but be aware it might return less.
                days_to_fetch = 90 # ~3 months for 60m
            # --- End Dynamic Data Fetch Period ---

            start = end - timedelta(days=days_to_fetch)
            logging.info(f"Attempting to fetch ~{days_to_fetch} days of data for symbol {self.symbol} with interval {self.minutes}m ending {end.strftime('%Y-%m-%d %H:%M')}.")

            # Fetch data using yfinance
            df = yf.download(self.symbol, start=start, end=end, interval=f'{self.minutes}m', progress=False) # progress=False for cleaner logs

            download_end = datetime.now()
            logging.info(f"Data download completed at {download_end.strftime('%Y-%m-%d %H:%M')}.")
            logging.info(f"Downloaded {len(df)} rows for {self.symbol} from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")

            if df.empty:
                raise ValueError(f"No data downloaded for symbol {self.symbol} from {start.date()} to {end.date()} with interval {self.minutes}m. Check symbol and interval validity.")

            # Check raw data size before feature engineering
            min_required_raw_rows = 60 # Arbitrary minimum to allow for feature calculation lookbacks
            if len(df) < min_required_raw_rows:
                 raise ValueError(f"Insufficient raw data downloaded ({len(df)} rows) for interval {self.minutes}m. Need at least {min_required_raw_rows}. Try a longer fetch period or check symbol/interval.")

            # Prepare features and target
            X, y = self.prepare_features(df) # This method includes its own check for < 50 *processed* rows

            # Check processed data size
            if len(X) < 100: # Minimum required samples for reliable training/splitting
                raise ValueError(f"Insufficient training data ({len(X)} rows) after feature engineering for interval {self.minutes}m. Original download had {len(df)} rows.")

            # Feature Selection
            self.important_features = self.select_features(X, y)
            X = X[self.important_features]

            # --- Handle Class Imbalance and SMOTE ---
            class_counts = y.value_counts()
            logging.info(f"Class distribution before SMOTE: {class_counts.to_dict()}")

            if len(class_counts) < 2:
                raise ValueError(f"Training data only contains one class ({class_counts.index[0]}) after processing. Cannot train.")
            if 0 in class_counts.values:
                 raise ValueError(f"One class has zero samples after processing. Class counts: {class_counts.to_dict()}")

            min_samples_minority = class_counts.min()
            # Dynamically adjust SMOTE k_neighbors based on minority class size
            # k_neighbors must be less than the number of samples in the smallest class
            smote_k = max(1, min(5, min_samples_minority - 1)) # Ensure k >= 1 and k < min_samples
            if smote_k != self.pipeline.steps[0][1].k_neighbors:
                 logging.info(f"Minority class has {min_samples_minority} samples. Adjusting SMOTE k_neighbors to {smote_k}.")
                 self.pipeline.steps[0] = ('smote', SMOTE(random_state=42, k_neighbors=smote_k))#, sampling_strategy='auto')) # Adjust sampling strategy if needed

            # Log imbalance ratio
            imbalance_ratio = class_counts.max() / class_counts.min()
            logging.info(f"Class imbalance ratio: {imbalance_ratio:.2f}")
            if imbalance_ratio > 5: # Log warning for high imbalance
                 logging.warning(f"High class imbalance detected (ratio > 5). SMOTE with k={smote_k} will be applied.")

            # --- TimeSeries Cross-Validation Setup ---
            # Adjust n_splits based on data size? Or keep fixed? Let's keep 4 for consistency unless data is very small.
            n_splits = 4
            min_rows_for_4_splits = 150 # Need enough data for 5 segments (4 train + 1 test)
            if len(X) < min_rows_for_4_splits:
                 n_splits = 3
                 logging.warning(f"Dataset size ({len(X)}) is less than {min_rows_for_4_splits}, reducing TimeSeriesSplit n_splits to {n_splits}")
            if len(X) < 100: # Should have been caught earlier, but safeguard
                 n_splits = 2
                 logging.warning(f"Dataset size ({len(X)}) is very small, reducing TimeSeriesSplit n_splits to {n_splits}")

            tscv = TimeSeriesSplit(n_splits=n_splits)

            # Check if splits are feasible
            min_samples_per_split = len(y) // (n_splits + 1)
            if min_samples_per_split < max(2, smote_k + 1): # Need enough for SMOTE + classification
                raise ValueError(f"Dataset too small ({len(y)} samples) to create {n_splits} valid time series splits with SMOTE k={smote_k}. Need at least {max(2, smote_k + 1)} samples per initial split segment.")


            # --- Cross-Validation Loop ---
            # We perform CV primarily to get a performance estimate and check for bias.
            # The *final* model will be trained on all data afterwards.
            logging.info(f"--- Starting TimeSeries Cross-Validation with {n_splits} splits ---")
            fold_num = 0
            all_y_test = []
            all_y_pred = []
            fold_f1_scores = []

            for train_index, test_index in tscv.split(X):
                fold_num += 1
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                logging.info(f"Fold {fold_num}/{n_splits}: Train indices {train_index[0]}-{train_index[-1]} (size={len(X_train)}), Test indices {test_index[0]}-{test_index[-1]} (size={len(X_test)})")

                # Check k_neighbors against this specific fold's minority class size
                fold_class_counts = y_train.value_counts()
                if len(fold_class_counts) < 2:
                    logging.warning(f"Fold {fold_num}: Skipping fold - training data has only one class.")
                    continue
                min_samples_train_fold = fold_class_counts.min()
                current_k = self.pipeline.steps[0][1].k_neighbors

                temp_pipeline = self.pipeline # Use the main pipeline by default

                if min_samples_train_fold <= current_k:
                    new_k_fold = max(1, min_samples_train_fold - 1)
                    logging.warning(f"Fold {fold_num}: Training data minority class ({min_samples_train_fold} samples) <= SMOTE k ({current_k}). Temporarily adjusting k to {new_k_fold} for this fold.")
                    # Create a temporary pipeline with adjusted k
                    temp_pipeline = Pipeline([
                        ('smote', SMOTE(random_state=42, k_neighbors=new_k_fold)),
                        ('scaler', RobustScaler()),
                        ('classifier', self.pipeline.steps[2][1]) # Re-use classifier instance
                    ])

                # Fit and predict for the fold
                temp_pipeline.fit(X_train, y_train)
                y_pred_fold = temp_pipeline.predict(X_test)
                pred_proba_fold = temp_pipeline.predict_proba(X_test)

                logging.info(f"\nFold {fold_num} Performance Report:")
                report = classification_report(y_test, y_pred_fold, output_dict=True, zero_division=0)
                logging.info(classification_report(y_test, y_pred_fold, zero_division=0))
                fold_f1_scores.append(report['weighted avg']['f1-score']) # Use weighted F1

                all_y_test.extend(y_test)
                all_y_pred.extend(y_pred_fold)

                # Store probabilities from the *last* fold for bias check
                if fold_num == n_splits:
                    self.base_predictions = pred_proba_fold

            # --- Overall Cross-Validation Performance ---
            if not all_y_test:
                 raise RuntimeError("Cross-validation failed - no folds were successfully evaluated.")

            logging.info("\n--- Overall Model Performance (Aggregated CV Results) ---")
            logging.info(classification_report(all_y_test, all_y_pred, zero_division=0))
            logging.info(f"Average cross-validation F1 score (weighted): {np.mean(fold_f1_scores):.4f}")

            # Bias check using the last fold's probabilities
            if self.base_predictions is not None:
                avg_prob = self.base_predictions.mean(axis=0)
                logging.info(f"Average prediction probabilities (from last CV fold): [Down={avg_prob[0]:.4f}, Up={avg_prob[1]:.4f}]")
                if abs(avg_prob[0] - avg_prob[1]) > 0.20: # Stricter bias check?
                    logging.warning("Model shows potential prediction bias based on last CV fold probabilities.")
            else:
                logging.warning("Could not evaluate prediction bias (no probabilities stored from CV).")


            # --- Calculate Average Price Changes for Target Classes ---
            # Use the original DataFrame 'df' and the target 'y' before splitting/SMOTE
            # Ensure alignment between df and y (feature engineering drops initial rows)
            df_aligned = df.reindex(y.index) # Align df rows with y's indices

            future_close_col = df_aligned['Close'].shift(-self.minutes)
            price_change_pct = (future_close_col - df_aligned['Close']) / df_aligned['Close'].replace(0, 1e-9) # Avoid div by zero

            # Ensure price_change_pct and y have the same index for boolean indexing
            price_change_pct = price_change_pct.reindex(y.index).dropna()
            y_aligned = y.reindex(price_change_pct.index) # Align y to the valid price change indices

            up_changes = price_change_pct[y_aligned == 1]
            down_changes = price_change_pct[y_aligned == 0]

            self.avg_change_up = float(up_changes.mean()) if not up_changes.empty and up_changes.notna().any() else 0.0
            self.avg_change_down = float(down_changes.mean()) if not down_changes.empty and down_changes.notna().any() else 0.0

            # Handle potential NaNs if mean results in NaN (e.g., only NaNs in the series)
            if pd.isna(self.avg_change_up): self.avg_change_up = 0.0
            if pd.isna(self.avg_change_down): self.avg_change_down = 0.0

            logging.info(f"Calculated Average % Price Change for Target=1 (Up): {self.avg_change_up:.6f}")
            logging.info(f"Calculated Average % Price Change for Target=0 (Down/No Change): {self.avg_change_down:.6f}")


            # --- Final Model Training ---
            # Train the final pipeline on *all* the prepared data (X, y)
            # Re-check SMOTE k one last time for the full dataset
            final_k = max(1, min(5, y.value_counts().min() - 1))
            if final_k != self.pipeline.steps[0][1].k_neighbors:
                 logging.info(f"Adjusting final SMOTE k_neighbors to {final_k} based on full dataset minority size.")
                 self.pipeline.steps[0] = ('smote', SMOTE(random_state=42, k_neighbors=final_k))

            logging.info(f"--- Fitting final model on all {len(X)} processed data points ---")
            self.pipeline.fit(X, y)
            logging.info("Final model training completed successfully.")

            return self.pipeline

        except ValueError as ve: # Catch specific data/value errors
            logging.error(f"ValueError during model training: {str(ve)}")
            # Re-raise for API handler to potentially return 4xx error
            raise ValueError(f"Training Data Error: {str(ve)}") from ve
        except RuntimeError as re: # Catch specific runtime errors (e.g., during feature prep)
            logging.error(f"RuntimeError during model training: {str(re)}")
            raise RuntimeError(f"Training Runtime Error: {str(re)}") from re
        except Exception as e:
            logging.exception(f"Unexpected error during model training: {str(e)}") # Log full traceback
            # Re-raise for API handler to return 500 error
            raise Exception(f"Internal Server Error during training: {str(e)}") from e


    def predict_next_movement(self):
        """
        Fetches recent data, prepares features, and predicts the next price movement.
        """
        try:
            # --- Fetch Recent Data ---
            # Need enough data points to calculate all features (longest lookback ~ 30 periods)
            # Fetch more than the minimum required for robustness (e.g., 100 periods)
            periods_to_fetch = 100
            fetch_minutes = periods_to_fetch * self.minutes
            # Ensure we fetch at least a few hours, max 2 days (to avoid excessive fetch)
            fetch_hours = max(4, min(48, fetch_minutes / 60.0))

            end = datetime.now()
            start = end - timedelta(hours=fetch_hours)

            logging.info(f"Fetching recent data for prediction: ~{fetch_hours:.1f} hours ({periods_to_fetch} periods) for interval {self.minutes}m.")
            recent_data = yf.download(self.symbol, start=start, end=end, interval=f'{self.minutes}m', progress=False)

            if recent_data.empty:
                raise ValueError(f"No recent data downloaded for prediction ({self.symbol}, {self.minutes}m).")
            # Need at least ~30 rows for feature calculation + 1 row for prediction
            if len(recent_data) < 35:
                 raise ValueError(f"Insufficient recent data ({len(recent_data)} rows) downloaded for feature calculation (need >35).")


            # --- Prepare Features for Recent Data ---
            # We need features for the *latest* point, but context features (ATR, returns mean) use history
            X_recent, _ = self.prepare_features(recent_data.copy()) # Pass copy to avoid modifying original

            if X_recent.empty:
                 raise ValueError("Feature preparation resulted in empty DataFrame from recent data.")

            # Get the features for the single latest data point
            latest_features_full = X_recent.iloc[-1:] # Keep as DataFrame row

            # Ensure the model pipeline is trained and features are selected
            if self.pipeline is None or not hasattr(self.pipeline, 'predict_proba'):
                 raise RuntimeError("Model pipeline is not trained or available. Train the model first.")
            if self.important_features is None:
                 raise RuntimeError("Important features not selected. Train the model first.")

            # Select only the important features for prediction
            missing_cols = [col for col in self.important_features if col not in latest_features_full.columns]
            if missing_cols:
                raise ValueError(f"Missing required features after processing recent data: {missing_cols}")
            X_pred = latest_features_full[self.important_features]


            # --- Make Prediction ---
            probabilities = self.pipeline.predict_proba(X_pred)[0] # [prob_down, prob_up]
            prediction = 1 if probabilities[1] >= 0.5 else 0
            logging.info(f"Raw Prediction: {'Up' if prediction == 1 else 'Down'}, Probabilities: [Down={probabilities[0]:.4f}, Up={probabilities[1]:.4f}]")


            # --- Context Adjustment (Volatility & Trend) ---
            market_volatility_ratio = 1.0 # Default: neutral volatility
            recent_trend = 0.0 # Default: neutral trend

            # Calculate recent volatility using ATR from the historical features (X_recent)
            if 'ATR' in X_recent.columns and len(X_recent) >= 20:
                 # Compare last 5 ATR values to the previous 15
                 recent_atr_mean = X_recent['ATR'].iloc[-5:].mean()
                 previous_atr_mean = X_recent['ATR'].iloc[-20:-5].mean()
                 if pd.notna(recent_atr_mean) and pd.notna(previous_atr_mean) and previous_atr_mean > 1e-9:
                      market_volatility_ratio = recent_atr_mean / previous_atr_mean
                 logging.info(f"Volatility Context: Recent ATR Mean={recent_atr_mean:.6f}, Previous ATR Mean={previous_atr_mean:.6f}, Ratio={market_volatility_ratio:.2f}")
            else:
                 logging.warning("Not enough data or ATR missing for volatility calculation.")

            # Calculate recent trend using returns from historical features (X_recent)
            if 'returns' in X_recent.columns and len(X_recent) >= 5:
                 recent_trend = X_recent['returns'].iloc[-5:].mean() # Avg return over last 5 periods
                 recent_trend = recent_trend if pd.notna(recent_trend) else 0.0
                 logging.info(f"Trend Context: Mean return (last 5 periods) = {recent_trend:.6f}")
            else:
                 logging.warning("Not enough data or returns missing for trend calculation.")

            # Apply adjustments based on context
            confidence_adjustment = 0.0
            # Reduce confidence if volatility is high (e.g., ratio > 1.5)
            if market_volatility_ratio > 1.5:
                 confidence_adjustment -= 0.07 # Slightly smaller penalty
                 logging.info(f"High market volatility detected (ratio={market_volatility_ratio:.2f}), reducing confidence.")
            # Reduce confidence if prediction conflicts with recent trend
            trend_threshold = 0.0 # Check if trend is clearly opposing prediction
            if (prediction == 1 and recent_trend < trend_threshold) or \
               (prediction == 0 and recent_trend > trend_threshold):
                 # Scale penalty by trend magnitude? For now, fixed penalty.
                 confidence_adjustment -= 0.05
                 logging.info(f"Prediction ({'Up' if prediction == 1 else 'Down'}) conflicts with recent trend ({recent_trend:.5f}), reducing confidence.")

            # Apply the adjustment to the probabilities
            adjusted_probabilities = probabilities.copy()
            if confidence_adjustment != 0:
                 logging.info(f"Applying confidence adjustment: {confidence_adjustment:.3f}")
                 if prediction == 1: # Predicted Up
                      prob_up_adjusted = probabilities[1] + confidence_adjustment
                      # Ensure adjusted probability stays within reasonable bounds (e.g., > 0.5 if still predicting up)
                      adjusted_probabilities[1] = max(0.501, min(0.99, prob_up_adjusted))
                      adjusted_probabilities[0] = 1.0 - adjusted_probabilities[1]
                 else: # Predicted Down
                      prob_down_adjusted = probabilities[0] - confidence_adjustment # Adjustment is negative, subtract to increase prob_down
                      adjusted_probabilities[0] = max(0.501, min(0.99, prob_down_adjusted))
                      adjusted_probabilities[1] = 1.0 - adjusted_probabilities[0]
                 logging.info(f"Adjusted Probabilities: [Down={adjusted_probabilities[0]:.4f}, Up={adjusted_probabilities[1]:.4f}]")


            # --- Determine Confidence Level ---
            max_prob = max(adjusted_probabilities)
            if max_prob >= self.HIGH_CONFIDENCE_THRESHOLD:
                confidence_level = "High"
            elif max_prob >= self.MEDIUM_CONFIDENCE_THRESHOLD:
                confidence_level = "Medium"
            elif max_prob >= self.LOW_CONFIDENCE_THRESHOLD:
                confidence_level = "Low"
            else:
                confidence_level = "Very Low"

            # Override confidence if probabilities are very close after adjustment
            if abs(adjusted_probabilities[0] - adjusted_probabilities[1]) < 0.05: # Threshold for "too close"
                 confidence_level = "Very Low"
                 logging.info("Adjusted probabilities are very close (< 0.05 diff), confidence set to Very Low")
            logging.info(f"Final Confidence Level: {confidence_level}")


            # --- Estimate Future Price ---
            # Use the *actual* latest close price from the raw downloaded data
            current_price = float(recent_data['Close'].iloc[-1])

            # Check if average change values are calculated
            if self.avg_change_up is None or self.avg_change_down is None:
                 logging.warning("Average price change data not available from training. Using 0 for future price estimation.")
                 self.avg_change_up = 0.0
                 self.avg_change_down = 0.0

            prob_up = adjusted_probabilities[1]
            prob_down = adjusted_probabilities[0]
            expected_change_pct = (prob_up * float(self.avg_change_up)) + (prob_down * float(self.avg_change_down))
            estimated_future_price = float(current_price * (1 + expected_change_pct))

            logging.info(f"Current Price: {current_price:.6f}")
            logging.info(f"Expected Pct Change (based on probabilities and avg historical moves): {expected_change_pct:.6f}")
            logging.info(f"Estimated Future Price (in {self.minutes} min): {estimated_future_price:.6f}")

            return (
                prediction,                 # 0 or 1
                adjusted_probabilities,     # Numpy array [prob_down, prob_up]
                current_price,              # Float
                datetime.now(),             # Datetime object (prediction time)
                confidence_level,           # String ("High", "Medium", etc.)
                estimated_future_price      # Float
            )

        except ValueError as ve:
            logging.error(f"ValueError during prediction: {str(ve)}")
            # Re-raise for API handler
            raise ValueError(f"Prediction Data Error: {str(ve)}") from ve
        except RuntimeError as re:
             logging.error(f"RuntimeError during prediction: {str(re)}")
             raise RuntimeError(f"Prediction Runtime Error: {str(re)}") from re
        except Exception as e:
            logging.exception(f"Unexpected error during prediction: {str(e)}")
            # Re-raise for API handler
            raise Exception(f"Internal Server Error during prediction: {str(e)}") from e