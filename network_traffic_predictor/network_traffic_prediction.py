# Network Traffic Prediction Project
# This project uses machine learning to predict network traffic based on historical data
# It includes data preprocessing, model training, evaluation, and deployment components

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from flask import Flask, request, jsonify
import os
import json
import datetime

# ========== DATA PROCESSING MODULE ==========

class DataProcessor:
    def __init__(self, seq_length=24):
        """Initialize the data processor with sequence length for time series."""
        self.seq_length = seq_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def load_data(self, filepath):
        """Load network traffic data from CSV file."""
        try:
            # Expected format: timestamp, traffic_volume (bytes/packets), additional_features
            data = pd.read_csv("C:/Users/hp/Documents/network_traffic_predictor/web_traffic.csv")
            
            # Convert timestamp to datetime
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                # Extract time-based features
                data['hour'] = data['timestamp'].dt.hour
                data['day_of_week'] = data['timestamp'].dt.dayofweek
                data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
                
            print(f"Data loaded successfully with {data.shape[0]} records and {data.shape[1]} features")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            # Generate synthetic data for demo purposes
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self, n_samples=8760):  # ~1 year of hourly data
        """Generate synthetic network traffic data for demonstration."""
        print("Generating synthetic network traffic data...")
        
        # Create timestamp index (hourly for one year)
        base = datetime.datetime(2023, 1, 1)
        timestamps = [base + datetime.timedelta(hours=i) for i in range(n_samples)]
        
        # Generate synthetic traffic patterns
        # Base traffic with daily and weekly patterns
        hourly_pattern = np.sin(np.linspace(0, 2*np.pi, 24)) * 0.4 + 0.6  # Daily pattern
        weekly_multiplier = np.array([0.8, 1.0, 1.1, 1.0, 1.1, 0.7, 0.6])  # Weekly pattern
        
        traffic = []
        for i, ts in enumerate(timestamps):
            # Get hour and day of week
            hour = ts.hour
            day = ts.weekday()
            
            # Base traffic level with some randomness
            base_traffic = hourly_pattern[hour] * weekly_multiplier[day]
            
            # Add random noise and trends
            noise = np.random.normal(0, 0.05)
            trend = i / n_samples * 0.3  # Gradual increase over time
            
            # Calculate final traffic value (in Gbps)
            traffic_value = (base_traffic + noise + trend) * 10  # Scale to reasonable network traffic values
            
            traffic.append(max(0, traffic_value))  # Ensure non-negative
        
        # Create DataFrame
        data = pd.DataFrame({
            'timestamp': timestamps,
            'traffic_gbps': traffic
        })
        
        # Extract time features
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        return data
    
    def preprocess_data(self, data, target_col='traffic_gbps'):
        """Preprocess the data for LSTM model training."""
        # Drop any missing values
        data = data.dropna()
        
        # Scale the traffic data
        traffic_data = data[target_col].values.reshape(-1, 1)
        scaled_traffic = self.scaler.fit_transform(traffic_data)
        
        # Create sequences for LSTM
        X, y = [], []
        for i in range(len(scaled_traffic) - self.seq_length):
            X.append(scaled_traffic[i:i + self.seq_length, 0])
            y.append(scaled_traffic[i + self.seq_length, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Reshape for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def create_train_test_split(self, X, y, test_size=0.2):
        """Split data into training and testing sets."""
        return train_test_split(X, y, test_size=test_size, shuffle=False)
    
    def inverse_transform(self, scaled_data):
        """Convert scaled predictions back to original scale."""
        return self.scaler.inverse_transform(scaled_data.reshape(-1, 1))
    
    def save_scaler(self, path="models/scaler.pkl"):
        """Save the scaler for later use in predictions."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)
        print(f"Scaler saved to {path}")
    
    def load_scaler(self, path="models/scaler.pkl"):
        """Load the scaler for making predictions."""
        self.scaler = joblib.load(path)
        print(f"Scaler loaded from {path}")

# ========== MODEL TRAINING MODULE ==========

class TrafficPredictor:
    def __init__(self, seq_length=24):
        """Initialize the predictor with sequence length."""
        self.seq_length = seq_length
        self.model = None
    
    def build_model(self, input_shape):
        """Build and compile LSTM model."""
        model = Sequential()
        
        # LSTM layers with dropout to prevent overfitting
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model with mean squared error loss and Adam optimizer
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        self.model = model
        print("Model built successfully")
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the LSTM model."""
        if self.model is None:
            self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        print("Model training completed")
        return history
    
    def evaluate_model(self, X_test, y_test, scaler):
        """Evaluate the model performance."""
        # Make predictions
        predictions = self.model.predict(X_test)
        
        # Inverse transform the scaled predictions
        predictions = scaler.inverse_transform(predictions)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate error metrics
        mse = mean_squared_error(y_test_actual, predictions)
        mae = mean_absolute_error(y_test_actual, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_actual, predictions)
        
        evaluation = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
        print(f"Model Evaluation: MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
        return predictions, y_test_actual, evaluation
    
    def save_model(self, path="models/traffic_model.h5"):
        """Save the trained model for deployment."""
        if self.model is None:
            print("No model to save")
            return
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path="models/traffic_model.h5"):
        """Load a trained model for prediction."""
        self.model = load_model(path)
        print(f"Model loaded from {path}")
    
    def plot_results(self, actual, predicted, title="Actual vs Predicted Network Traffic"):
        """Plot actual vs predicted values."""
        plt.figure(figsize=(12, 6))
        plt.plot(actual, label='Actual Traffic')
        plt.plot(predicted, label='Predicted Traffic')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Traffic Volume (Gbps)')
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/prediction_results.png')
        plt.close()
        
        print("Results plot saved to 'results/prediction_results.png'")
    
    def predict_future(self, last_sequence, steps_ahead=24, scaler=None):
        """Predict future traffic for a number of steps ahead."""
        if self.model is None:
            print("No model loaded for prediction")
            return None
        
        # Make a copy of the last sequence for predictions
        curr_sequence = last_sequence.copy()
        future_predictions = []
        
        # Predict one step at a time
        for _ in range(steps_ahead):
            # Reshape for prediction
            curr_reshaped = curr_sequence.reshape(1, self.seq_length, 1)
            
            # Predict next value
            next_pred = self.model.predict(curr_reshaped)[0, 0]
            
            # Add to predictions
            future_predictions.append(next_pred)
            
            # Update sequence (remove first, append prediction)
            curr_sequence = np.append(curr_sequence[1:], next_pred)
        
        # Inverse transform if scaler provided
        if scaler:
            future_predictions = scaler.inverse_transform(
                np.array(future_predictions).reshape(-1, 1)
            ).flatten()
        
        return future_predictions

# ========== DEPLOYMENT MODULE ==========

class TrafficPredictionService:
    def __init__(self):
        """Initialize the prediction service."""
        self.data_processor = DataProcessor()
        self.predictor = TrafficPredictor()
        
    def load_models(self, model_path="models/traffic_model.h5", scaler_path="models/scaler.pkl"):
        """Load the trained model and scaler."""
        try:
            self.predictor.load_model(model_path)
            self.data_processor.load_scaler(scaler_path)
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def predict_traffic(self, input_data, hours_ahead=24):
        """Make traffic predictions from input sequence."""
        try:
            # Prepare input data
            if isinstance(input_data, list):
                # Convert to numpy array if a list is provided
                sequence = np.array(input_data)
            else:
                # Assume it's already a numpy array or similar
                sequence = input_data
                
            # Scale the input sequence
            scaled_sequence = self.data_processor.scaler.transform(
                sequence.reshape(-1, 1)
            ).flatten()
            
            # Make prediction
            predictions = self.predictor.predict_future(
                scaled_sequence, 
                steps_ahead=hours_ahead,
                scaler=self.data_processor.scaler
            )
            
            return predictions
        except Exception as e:
            print(f"Prediction error: {e}")
            return None



# ========== MAIN EXECUTION ==========

def main():
    """Main function to run the entire pipeline."""
    print("Starting Network Traffic Prediction project...")
    
    # Create data processor and load data
    data_processor = DataProcessor(seq_length=24)
    # data = data_processor.load_data("path/to/your/data.csv")  # Use this for real data
    data = data_processor.load_data(None)  # This will generate synthetic data
    
    # Preprocess data
    X, y = data_processor.preprocess_data(data)
    X_train, X_test, y_train, y_test = data_processor.create_train_test_split(X, y)
    
    # Further split test data into validation and test sets
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=False)
    
    # Build and train the model
    predictor = TrafficPredictor(seq_length=24)
    predictor.build_model((X_train.shape[1], 1))
    history = predictor.train_model(X_train, y_train, X_val, y_val, epochs=20) 
    # Evaluate the model
    predictions, actual, metrics = predictor.evaluate_model(X_test, y_test, data_processor.scaler)
    predictor.plot_results(actual, predictions)
    
    # Save model and scaler for deployment
    predictor.save_model()
    data_processor.save_scaler()
    
    # Example of future prediction (next 24 hours)
    last_sequence = X_test[-1]
    future_pred = predictor.predict_future(
        last_sequence, 
        steps_ahead=24, 
        scaler=data_processor.scaler
    )
    
    print("\nFuture 24-hour traffic prediction (Gbps):")
    for i, pred in enumerate(future_pred):
        print(f"Hour {i+1}: {pred:.3f} Gbps")
    
    # Save prediction metrics to file
    os.makedirs('results', exist_ok=True)
    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("\nProject execution completed successfully")
    print("Models and results saved in 'models/' and 'results/' directories")
    print("\nTo deploy the API service, run:")
    print("    from network_traffic_prediction import app")
    print("    app.run(host='0.0.0.0', port=5000)")

if __name__ == "__main__":
    main()
    
    