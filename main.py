# main.py (or your FastAPI application file)

from fastapi import FastAPI, HTTPException, Query
import logging
import numpy as np
import time # Import time for logging durations

# Assuming predictor.py is in the same directory or accessible via Python path
# This predictor.py should be the one with the dynamic data fetching logic from the previous response.
from predictor import ShortTermPredictor
from datetime import datetime # Keep datetime import

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

# Consider predictor caching or management if resource usage becomes an issue
# For simplicity, we create a new instance per request here.

@app.get("/predict")
def get_prediction(
    symbol: str = Query("BTC-USD", description="Trading currency symbol (e.g., BTC-USD, AAPL)"),
    minutes: int = Query(5, description="Time interval in minutes (e.g., 1, 5, 15, 30, 60)")
):
    """
    Trains a model (or uses a cached one in a real scenario) and predicts
    the price movement for the next interval, returning data in the original format.
    """
    request_start_time = time.time()
    logging.info(f"Received prediction request for {symbol} at {minutes}m interval.")
    try:
        # Optional: Validate minutes input (can add specific allowed values if needed)
        if minutes <= 0:
            raise HTTPException(status_code=400, detail="Minutes must be a positive integer.")
        # You might want to log a warning for non-standard intervals supported by yfinance
        # if minutes not in [1, 5, 15, 30, 60]:
        #     logging.warning(f"Requested interval {minutes}m might not be standard or supported by yfinance.")

        # Instantiate the predictor (using the improved class)
        predictor = ShortTermPredictor(symbol, minutes)

        # --- Training ---
        # As before, training on every request is slow and resource-intensive.
        # Consider pre-training or offline training in production.
        logging.info(f"[{symbol}-{minutes}m] Starting model training...")
        train_start_time = time.time()
        predictor.train_model() # Call the improved training method
        train_duration = time.time() - train_start_time
        logging.info(f"[{symbol}-{minutes}m] Model training finished in {train_duration:.2f} seconds.")

        # --- Prediction ---
        logging.info(f"[{symbol}-{minutes}m] Generating prediction...")
        predict_start_time = time.time()
        # Call the improved prediction method - variable names match the original return keys
        (prediction, adjusted_probabilities, current_price,
         prediction_time, confidence_level, future_price) = predictor.predict_next_movement()
        predict_duration = time.time() - predict_start_time
        logging.info(f"[{symbol}-{minutes}m] Prediction generated in {predict_duration:.2f} seconds.")

        request_duration = time.time() - request_start_time
        logging.info(f"[{symbol}-{minutes}m] Total request duration: {request_duration:.2f} seconds.")

        # --- Format Response (Using the ORIGINAL structure) ---
        return {
            "symbol": symbol,
            "minutes": minutes,
            "prediction": "Up" if prediction == 1 else "Down",
            # Ensure probabilities are JSON serializable (list)
            "probabilities": adjusted_probabilities.tolist() if isinstance(adjusted_probabilities, np.ndarray) else adjusted_probabilities,
            "current_price": current_price, # No rounding applied here as per original
            "prediction_time": prediction_time.isoformat(),
            "confidence_level": confidence_level,
            "future_price": future_price # No rounding applied here as per original
        }

    except (ValueError, RuntimeError) as data_err:
        # Handle specific errors raised from predictor for data/runtime issues
        logging.error(f"[{symbol}-{minutes}m] Prediction failed due to data/runtime error: {str(data_err)}")
        # Use 400 for client-side/data issues potentially triggered by input
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(data_err)}")
    except Exception as e:
        # Catch any other unexpected errors during the process
        logging.exception(f"[{symbol}-{minutes}m] An unexpected error occurred during prediction: {str(e)}")
        # Use 500 for internal server errors
        raise HTTPException(status_code=500, detail="Internal server error during prediction process.")
