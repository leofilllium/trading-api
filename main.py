from fastapi import FastAPI, HTTPException, Query
import logging
import numpy as np
from predictor import ShortTermPredictor  # Ensure predictor.py contains your class definition

app = FastAPI()


@app.get("/predict")
def get_prediction(
        symbol: str = Query("AAPL", description="Trading currency symbol"),
        minutes: int = Query(15, description="Interval in minutes")
):
    try:
        # Instantiate the predictor with parameters from the query
        predictor = ShortTermPredictor(symbol, minutes)

        # Train the model (in production you might load a pre-trained model instead)
        predictor.train_model()

        # Get prediction details
        prediction, adjusted_probabilities, current_price, prediction_time, confidence_level = predictor.predict_next_movement()

        return {
            "symbol": symbol,
            "minutes": minutes,
            "prediction": "Up" if prediction == 1 else "Down",
            "probabilities": adjusted_probabilities.tolist() if isinstance(adjusted_probabilities,
                                                                           np.ndarray) else adjusted_probabilities,
            "current_price": current_price,
            "prediction_time": prediction_time.isoformat(),
            "confidence_level": confidence_level
        }
    except Exception as e:
        logging.error("Error while predicting: " + str(e))
        raise HTTPException(status_code=500, detail="Prediction failed")
