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


warnings.filterwarnings('ignore')


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
        self.HIGH_CONFIDENCE_THRESHOLD = 0.70  # Increased threshold for high confidence
        self.MEDIUM_CONFIDENCE_THRESHOLD = 0.65  # Added medium confidence threshold
        self.LOW_CONFIDENCE_THRESHOLD = 0.55

    def prepare_features(self, df):
        try:
            # Create a DataFrame with single-level columns
            data = pd.DataFrame()

            # Handle both multi-index and single-index DataFrames
            if isinstance(df.columns, pd.MultiIndex):
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    data[col] = df[(col, self.symbol)]
            else:
                data = df.copy()

            # Calculate returns
            data['returns'] = data['Close'].pct_change()

            # Calculate technical features
            data['vol'] = data['returns'].rolling(window=5).std()
            data['momentum'] = data['Close'] - data['Close'].shift(5)

            # Moving averages with different windows
            data['SMA5'] = data['Close'].rolling(window=5).mean()
            data['SMA10'] = data['Close'].rolling(window=10).mean()
            data['SMA20'] = data['Close'].rolling(window=20).mean()
            data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
            data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
            data['MA_crossover'] = data['SMA5'] - data['SMA10']
            data['EMA_crossover'] = data['EMA12'] - data['EMA26']

            # Price volatility
            data['high_low_range'] = data['High'] - data['Low']
            data['close_open_range'] = data['Close'] - data['Open']
            data['ATR'] = data['high_low_range'].rolling(window=14).mean()

            # Volume ratio with different windows
            volume_ma5 = data['Volume'].rolling(window=5).mean()
            volume_ma10 = data['Volume'].rolling(window=10).mean()
            data['volume_ratio'] = data['Volume'].div(volume_ma5).clip(upper=10)
            data['volume_ratio_10'] = data['Volume'].div(volume_ma10).clip(upper=10)

            # RSI with different windows
            for window in [7, 14, 21]:
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                data[f'RSI_{window}'] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            data['bollinger_mid'] = data['Close'].rolling(window=20).mean()
            data['bollinger_std'] = data['Close'].rolling(window=20).std()
            data['bollinger_upper'] = data['bollinger_mid'] + (data['bollinger_std'] * 2)
            data['bollinger_lower'] = data['bollinger_mid'] - (data['bollinger_std'] * 2)
            data['bollinger_width'] = data['bollinger_upper'] - data['bollinger_lower']

            # MACD
            data['MACD'] = data['EMA12'] - data['EMA26']
            data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data['MACD_hist'] = data['MACD'] - data['MACD_signal']

            # Stochastic Oscillator
            data['%K'] = ((data['Close'] - data['Low'].rolling(window=14).min()) /
                          (data['High'].rolling(window=14).max() - data['Low'].rolling(window=14).min())) * 100
            data['%D'] = data['%K'].rolling(window=3).mean()

            # Rate of Change (ROC)
            data['ROC'] = data['Close'].pct_change(periods=12) * 100

            # On-Balance Volume (OBV)
            data['OBV'] = (data['Volume'] * (~data['Close'].diff().le(0) * 2 - 1)).cumsum()

            # Additional features based on Open, High, Low prices
            data['open_equals_low'] = (data['Open'] == data['Low']).astype(int)
            data['open_equals_high'] = (data['Open'] == data['High']).astype(int)

            # New features based on the evaluation report
            # Volume Price Trend
            data['VPT'] = (data['Volume'] * data['returns']).cumsum()

            # Chaikin Money Flow (CMF)
            money_flow_volume = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (
                        data['High'] - data['Low']) * data['Volume']
            data['CMF'] = money_flow_volume.rolling(window=20).sum() / data['Volume'].rolling(window=20).sum()

            # Feature interactions
            data['RSI_vol_ratio'] = data['RSI_14'] * data['volume_ratio']
            data['momentum_volume'] = data['momentum'] * data['volume_ratio']
            data['bollinger_momentum'] = data['bollinger_width'] * data['momentum']

            # Target (future price movement)
            data['target'] = (data['Close'].shift(-self.minutes) > data['Close']).astype(int)

            # Add lagged features with different lags for selected indicators
            for lag in range(1, 4):  # Reduced from 5 to 3 lags
                data[f'returns_lag{lag}'] = data['returns'].shift(lag)
                data[f'vol_lag{lag}'] = data['vol'].shift(lag)
                data[f'RSI_14_lag{lag}'] = data['RSI_14'].shift(lag)
                data[f'MACD_lag{lag}'] = data['MACD'].shift(lag)
                data[f'volume_ratio_lag{lag}'] = data['volume_ratio'].shift(lag)

            # Drop NaN values
            data = data.dropna()

            # Initial feature set
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

            # Check if we have enough data
            if len(data) < 50:
                raise ValueError("Insufficient data points for reliable prediction")

            # Extract only the features we need
            features = data[self.feature_columns]
            target = data['target']

            logging.info(f"Successfully prepared features. Shape: {features.shape}")
            return features, target

        except Exception as e:
            logging.error(f"Error in feature preparation: {str(e)}")
            raise

    def select_features(self, X, y):
        # Train a model to get feature importances
        feature_selector = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42
        )
        feature_selector.fit(X, y)

        # Get feature importances
        importances = feature_selector.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        # Select top features (approximately 60% of original features)
        top_features_count = int(len(X.columns) * 0.6)
        top_features = feature_importance['Feature'][:top_features_count].tolist()

        logging.info(f"Selected {len(top_features)} features based on importance")
        logging.info(f"Top 10 features: {top_features[:10]}")

        return top_features

    def train_model(self, hyperparameter_optimization=False):
        try:
            # Download historical data with more history
            end = datetime.now()
            start = end - timedelta(days=7)  # Increased for more training data
            df = yf.download(self.symbol, start=start, end=end, interval=f'{self.minutes}m')
            logging.info(f"Downloaded {len(df)} rows of data for {self.symbol}")

            if df.empty:
                raise ValueError("No data downloaded")

            X, y = self.prepare_features(df)

            if len(X) < 100:
                raise ValueError("Insufficient training data")

            # Use feature selection to find the most important features
            self.important_features = self.select_features(X, y)
            X = X[self.important_features]  # Keep only important features

            # Handle class imbalance check
            class_counts = y.value_counts()
            logging.info(f"Class distribution: {class_counts}")
            imbalance_ratio = class_counts.max() / class_counts.min()

            if imbalance_ratio > 3:
                logging.warning(f"Severe class imbalance detected: {imbalance_ratio:.2f}. Adjusting SMOTE parameters.")
                # Adjust SMOTE parameters for severe imbalance
                self.pipeline.steps[0] = ('smote', SMOTE(random_state=42, k_neighbors=3, sampling_strategy=0.8))

            # Split the data using TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=4)

            # Perform hyperparameter optimization if requested
            if hyperparameter_optimization:
                self.optimize_hyperparameters(X, y, tscv)

            # Train and evaluate the model
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                # Train the pipeline
                self.pipeline.fit(X_train, y_train)

                # Evaluate
                y_pred = self.pipeline.predict(X_test)
                logging.info("\nModel Performance:")
                logging.info(classification_report(y_test, y_pred))

                # Cross-validation for a more robust evaluation
                cv_scores = cross_val_score(self.pipeline, X, y, cv=tscv, scoring='f1')
                logging.info(f"Cross-validation F1 scores: {cv_scores}")
                logging.info(f"Average cross-validation F1 score: {cv_scores.mean()}")

                # Verify model isn't biased
                pred_proba = self.pipeline.predict_proba(X_test)
                avg_prob = pred_proba.mean(axis=0)
                logging.info(f"Average prediction probabilities: {avg_prob}")

                if abs(avg_prob[0] - avg_prob[1]) > 0.15:  # Reduced threshold for bias detection
                    logging.warning("Model shows bias in predictions")

                # Store trained model's base predictions for calibration
                self.base_predictions = pred_proba

            return self.pipeline

        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise

    def optimize_hyperparameters(self, X, y, tscv):
        """Simple hyperparameter optimization using grid search"""
        from sklearn.model_selection import GridSearchCV

        # Define parameter grid for GradientBoostingClassifier
        param_grid = {
            'classifier__n_estimators': [300, 500, 700],
            'classifier__learning_rate': [0.003, 0.005, 0.01],
            'classifier__max_depth': [5, 6, 8],
            'classifier__subsample': [0.7, 0.8, 0.9],
            'classifier__min_samples_split': [30, 50, 70],
            'classifier__min_samples_leaf': [10, 20, 30],
            'classifier__max_features': [0.6, 0.7, 0.8]
        }

        # Create pipeline for grid search
        search_pipeline = Pipeline([
            ('smote', SMOTE(random_state=42, k_neighbors=5)),
            ('scaler', RobustScaler()),
            ('classifier', GradientBoostingClassifier(
                random_state=42
            ))
        ])

        # Perform grid search
        grid_search = GridSearchCV(
            search_pipeline,
            param_grid,
            cv=tscv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        logging.info("Starting hyperparameter optimization...")
        grid_search.fit(X, y)
        logging.info(f"Best parameters: {grid_search.best_params_}")

        # Update pipeline with best parameters
        best_params = grid_search.best_params_
        self.pipeline.set_params(**{k: v for k, v in best_params.items()})

        return best_params

    def predict_next_movement(self):
        try:
            # Download more historical data for feature calculation
            end = datetime.now()
            start = end - timedelta(hours=24)  # Increased for more context
            recent_data = yf.download(self.symbol, start=start, end=end, interval=f'{self.minutes}m')

            if recent_data.empty:
                raise ValueError("No recent data downloaded")

            # Generate full set of features
            X, _ = self.prepare_features(recent_data)
            X_full = X.copy()  # Preserve full feature set for market calculations

            # Use only the important features for prediction if they have been selected
            if self.important_features:
                X_pred = X[self.important_features]
            else:
                X_pred = X

            if len(X_pred) == 0:
                raise ValueError("No valid features generated from recent data")

            latest_features = X_pred.iloc[-1:].values

            # Get prediction and probabilities using the pipeline
            probabilities = self.pipeline.predict_proba(latest_features)[0]
            prediction = 1 if probabilities[1] >= 0.5 else 0

            # Calculate market conditions using the full feature set
            market_volatility = X_full['ATR'].iloc[-5:].mean() / X_full['ATR'].iloc[-20:].mean()
            recent_trend = X_full['returns'].iloc[-5:].mean()

            # Adjust confidence based on market conditions
            confidence_adjustment = 0

            # High volatility should reduce confidence
            if market_volatility > 1.5:
                confidence_adjustment -= 0.1
                logging.info(f"High market volatility detected ({market_volatility:.2f}), reducing confidence")

            # Conflicting signals should reduce confidence
            if (prediction == 1 and recent_trend < 0) or (prediction == 0 and recent_trend > 0):
                confidence_adjustment -= 0.05
                logging.info("Prediction conflicts with recent trend, reducing confidence")

            # Adjust probabilities based on market conditions
            adjusted_probabilities = probabilities.copy()
            if confidence_adjustment != 0:
                # Reduce probability of predicted class and increase other class
                if prediction == 1:
                    adjusted_probabilities[1] = max(0.5, probabilities[1] + confidence_adjustment)
                    adjusted_probabilities[0] = 1 - adjusted_probabilities[1]
                else:
                    adjusted_probabilities[0] = max(0.5, probabilities[0] + confidence_adjustment)
                    adjusted_probabilities[1] = 1 - adjusted_probabilities[0]

            # Determine confidence level based on adjusted probabilities
            max_prob = max(adjusted_probabilities)
            if max_prob >= self.HIGH_CONFIDENCE_THRESHOLD:
                confidence_level = "High"
            elif max_prob >= self.MEDIUM_CONFIDENCE_THRESHOLD:
                confidence_level = "Medium"
            elif max_prob >= self.LOW_CONFIDENCE_THRESHOLD:
                confidence_level = "Low"
            else:
                confidence_level = "Very Low"

            # Additional check for prediction uncertainty
            if abs(adjusted_probabilities[0] - adjusted_probabilities[1]) < 0.1:
                confidence_level = "Very Low"
                logging.info("Prediction uncertainty is high, confidence set to Very Low")

            logging.info(f"Prediction: {'Up' if prediction == 1 else 'Down'}, "
                        f"Probabilities: {probabilities}, Adjusted: {adjusted_probabilities}, "
                        f"Confidence: {confidence_level}")

            return prediction, adjusted_probabilities, recent_data['Close'].iloc[-1], datetime.now(), confidence_level

        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise