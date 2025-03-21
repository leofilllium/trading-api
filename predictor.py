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
import warnings
import threading
import time

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class ShortTermPredictor:
    def __init__(self, symbol, minutes):
        self.symbol = symbol
        self.minutes = minutes

        # Modified pipeline with GradientBoostingClassifier instead of XGBoost
        self.pipeline = Pipeline([
            ('smote', SMOTE(random_state=42, k_neighbors=5)),
            ('scaler', RobustScaler()),
            ('classifier', GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.04,
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

    # prepare_features method remains unchanged
    def prepare_features(self, df):
        try:
            # Create a DataFrame with single-level columns
            data = pd.DataFrame()

            if isinstance(df.columns, pd.MultiIndex):
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    data[col] = df[(col, self.symbol)]
            else:
                data = df.copy()

            data['returns'] = data['Close'].pct_change()
            data['vol'] = data['returns'].rolling(window=5).std()
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
            volume_ma5 = data['Volume'].rolling(window=5).mean()
            volume_ma10 = data['Volume'].rolling(window=10).mean()
            data['volume_ratio'] = data['Volume'].div(volume_ma5).clip(upper=10)
            data['volume_ratio_10'] = data['Volume'].div(volume_ma10).clip(upper=10)
            for window in [7, 14, 21]:
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                data[f'RSI_{window}'] = 100 - (100 / (1 + rs))
            data['bollinger_mid'] = data['Close'].rolling(window=20).mean()
            data['bollinger_std'] = data['Close'].rolling(window=20).std()
            data['bollinger_upper'] = data['bollinger_mid'] + (data['bollinger_std'] * 2)
            data['bollinger_lower'] = data['bollinger_mid'] - (data['bollinger_std'] * 2)
            data['bollinger_width'] = data['bollinger_upper'] - data['bollinger_lower']
            data['MACD'] = data['EMA12'] - data['EMA26']
            data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data['MACD_hist'] = data['MACD'] - data['MACD_signal']
            data['%K'] = ((data['Close'] - data['Low'].rolling(window=14).min()) /
                          (data['High'].rolling(window=14).max() - data['Low'].rolling(window=14).min())) * 100
            data['%D'] = data['%K'].rolling(window=3).mean()
            data['ROC'] = data['Close'].pct_change(periods=12) * 100
            data['OBV'] = (data['Volume'] * (~data['Close'].diff().le(0) * 2 - 1)).cumsum()
            data['open_equals_low'] = (data['Open'] == data['Low']).astype(int)
            data['open_equals_high'] = (data['Open'] == data['High']).astype(int)
            data['VPT'] = (data['Volume'] * data['returns']).cumsum()
            money_flow_volume = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (
                        data['High'] - data['Low']) * data['Volume']
            data['CMF'] = money_flow_volume.rolling(window=20).sum() / data['Volume'].rolling(window=20).sum()
            data['RSI_vol_ratio'] = data['RSI_14'] * data['volume_ratio']
            data['momentum_volume'] = data['momentum'] * data['volume_ratio']
            data['bollinger_momentum'] = data['bollinger_width'] * data['momentum']
            data['target'] = (data['Close'].shift(-self.minutes) > data['Close']).astype(int)
            for lag in range(1, 4):
                data[f'returns_lag{lag}'] = data['returns'].shift(lag)
                data[f'vol_lag{lag}'] = data['vol'].shift(lag)
                data[f'RSI_14_lag{lag}'] = data['RSI_14'].shift(lag)
                data[f'MACD_lag{lag}'] = data['MACD'].shift(lag)
                data[f'volume_ratio_lag{lag}'] = data['volume_ratio'].shift(lag)
            data = data.dropna()
            self.feature_columns = [
                'returns', 'vol', 'momentum', 'MA_crossover', 'EMA_crossover',
                'high_low_range', 'close_open_range', 'ATR', 'volume_ratio',
                'volume_ratio_10', 'RSI_7', 'RSI_14', 'RSI_21', 'bollinger_width',
                'MACD', 'MACD_signal', 'MACD_hist', '%K', '%D', 'ROC', 'OBV',
                'open_equals_low', 'open_equals_high', 'VPT', 'CMF',
                'RSI_vol_ratio', 'momentum_volume', 'bollinger_momentum',
                'returns_lag1', 'returns_lag2', 'returns_lag3',
                'vol_lag1', 'vol_lag2', 'vol_lag3',
                'RSI_14_lag1', 'RSI_14_lag2', 'RSI_14_lag3',
                'MACD_lag1', 'MACD_lag2', 'MACD_lag3',
                'volume_ratio_lag1', 'volume_ratio_lag2', 'volume_ratio_lag3'
            ]
            if len(data) < 50:
                raise ValueError("Insufficient data points for reliable prediction")
            features = data[self.feature_columns]
            target = data['target']
            logging.info(f"Successfully prepared features. Shape: {features.shape}")
            return features, target
        except Exception as e:
            logging.error(f"Error in feature preparation: {str(e)}")
            raise

    # select_features method remains unchanged
    def select_features(self, X, y):
        feature_selector = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42
        )
        feature_selector.fit(X, y)
        importances = feature_selector.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        top_features_count = int(len(X.columns) * 0.6)
        top_features = feature_importance['Feature'][:top_features_count].tolist()
        logging.info(f"Selected {len(top_features)} features based on importance")
        logging.info(f"Top 10 features: {top_features[:10]}")
        return top_features

    def train_model(self, hyperparameter_optimization=False):
        try:
            end = datetime.now()
            start = end - timedelta(days=7)
            df = yf.download(self.symbol, start=start, end=end, interval=f'{self.minutes}m')
            logging.info(f"Downloaded {len(df)} rows of data for {self.symbol}")
            if df.empty:
                raise ValueError("No data downloaded")
            X, y = self.prepare_features(df)
            if len(X) < 100:
                raise ValueError("Insufficient training data")
            self.important_features = self.select_features(X, y)
            X = X[self.important_features]
            class_counts = y.value_counts()
            logging.info(f"Class distribution: {class_counts}")
            imbalance_ratio = class_counts.max() / class_counts.min()
            if imbalance_ratio > 3:
                logging.warning(f"Severe class imbalance detected: {imbalance_ratio:.2f}. Adjusting SMOTE parameters.")
                self.pipeline.steps[0] = ('smote', SMOTE(random_state=42, k_neighbors=3, sampling_strategy=0.8))
            tscv = TimeSeriesSplit(n_splits=4)
            if hyperparameter_optimization:
                self.optimize_hyperparameters(X, y, tscv)
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                self.pipeline.fit(X_train, y_train)
                y_pred = self.pipeline.predict(X_test)
                logging.info("\nModel Performance:")
                logging.info(classification_report(y_test, y_pred))
                cv_scores = cross_val_score(self.pipeline, X, y, cv=tscv, scoring='f1')
                logging.info(f"Cross-validation F1 scores: {cv_scores}")
                logging.info(f"Average cross-validation F1 score: {cv_scores.mean()}")
                pred_proba = self.pipeline.predict_proba(X_test)
                avg_prob = pred_proba.mean(axis=0)
                logging.info(f"Average prediction probabilities: {avg_prob}")
                if abs(avg_prob[0] - avg_prob[1]) > 0.15:
                    logging.warning("Model shows bias in predictions")
                self.base_predictions = pred_proba

            # Calculate average percentage price changes for each class
            future_close = df['Close'].shift(-1).loc[X.index]  # Align with next period
            price_change = (future_close - df['Close'].loc[X.index]) / df['Close'].loc[X.index]
            up_changes = price_change[y == 1]
            down_changes = price_change[y == 0]
            self.avg_change_up = float(up_changes.mean()) if not up_changes.empty else 0.0
            self.avg_change_down = float(down_changes.mean()) if not down_changes.empty else 0.0
            # Explicit NaN handling (though .mean() should handle this)
            if pd.isna(self.avg_change_up):
                self.avg_change_up = 0.0
            if pd.isna(self.avg_change_down):
                self.avg_change_down = 0.0
            logging.info(
                f"Average % change (up): {self.avg_change_up:.4f}, (down/no change): {self.avg_change_down:.4f}")

            return self.pipeline
        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise

    def predict_next_movement(self):
        try:
            end = datetime.now()
            start = end - timedelta(hours=24)
            recent_data = yf.download(self.symbol, start=start, end=end, interval=f'{self.minutes}m')
            if recent_data.empty:
                raise ValueError("No recent data downloaded")
            X, _ = self.prepare_features(recent_data)
            X_full = X.copy()
            if self.important_features:
                X_pred = X[self.important_features]
            else:
                X_pred = X
            if len(X_pred) == 0:
                raise ValueError("No valid features generated from recent data")
            # Keep as DataFrame to preserve feature names
            latest_features = X_pred.iloc[-1:]  # Do not use .values
            probabilities = self.pipeline.predict_proba(latest_features)[0]
            prediction = 1 if probabilities[1] >= 0.5 else 0
            market_volatility = X_full['ATR'].iloc[-5:].mean() / X_full['ATR'].iloc[-20:].mean()
            recent_trend = X_full['returns'].iloc[-5:].mean()
            confidence_adjustment = 0
            if market_volatility > 1.5:
                confidence_adjustment -= 0.1
                logging.info(f"High market volatility detected ({market_volatility:.2f}), reducing confidence")
            if (prediction == 1 and recent_trend < 0) or (prediction == 0 and recent_trend > 0):
                confidence_adjustment -= 0.05
                logging.info("Prediction conflicts with recent trend, reducing confidence")
            adjusted_probabilities = probabilities.copy()
            if confidence_adjustment != 0:
                if prediction == 1:
                    adjusted_probabilities[1] = max(0.5, probabilities[1] + confidence_adjustment)
                    adjusted_probabilities[0] = 1 - adjusted_probabilities[1]
                else:
                    adjusted_probabilities[0] = max(0.5, probabilities[0] + confidence_adjustment)
                    adjusted_probabilities[1] = 1 - adjusted_probabilities[0]
            max_prob = max(adjusted_probabilities)
            if max_prob >= self.HIGH_CONFIDENCE_THRESHOLD:
                confidence_level = "High"
            elif max_prob >= self.MEDIUM_CONFIDENCE_THRESHOLD:
                confidence_level = "Medium"
            elif max_prob >= self.LOW_CONFIDENCE_THRESHOLD:
                confidence_level = "Low"
            else:
                confidence_level = "Very Low"
            if abs(adjusted_probabilities[0] - adjusted_probabilities[1]) < 0.1:
                confidence_level = "Very Low"
                logging.info("Prediction uncertainty is high, confidence set to Very Low")

            # Calculate estimated future price with explicit scalar conversion
            current_price = float(recent_data['Close'].iloc[-1].item())  # Use .item() for scalar
            prob_up = adjusted_probabilities[1]
            prob_down = adjusted_probabilities[0]
            expected_change = prob_up * self.avg_change_up + prob_down * self.avg_change_down
            estimated_future_price = float(current_price * (1 + expected_change))  # Ensure scalar

            logging.info(f"Prediction: {'Up' if prediction == 1 else 'Down'}, "
                         f"Probabilities: {probabilities}, Adjusted: {adjusted_probabilities}, "
                         f"Confidence: {confidence_level}, "
                         f"Estimated Future Price: {estimated_future_price:.6f}")

            return prediction, adjusted_probabilities, current_price, datetime.now(), confidence_level, estimated_future_price
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise

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