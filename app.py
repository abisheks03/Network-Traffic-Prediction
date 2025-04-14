from flask import Flask, request, jsonify
import json
import datetime
from network_traffic_prediction import TrafficPredictionService

# Flask API for deployment
app = Flask(__name__)
prediction_service = TrafficPredictionService()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    try:
        # Get input data from request
        data = request.json
        input_sequence = data.get('sequence', [])
        hours_ahead = data.get('hours_ahead', 24)
        
        # Validate input
        if not input_sequence or len(input_sequence) != prediction_service.predictor.seq_length:
            return jsonify({
                'error': f'Input sequence must contain exactly {prediction_service.predictor.seq_length} values'
            }), 400
        
        # Make prediction
        predictions = prediction_service.predict_traffic(input_sequence, hours_ahead)
        
        if predictions is None:
            return jsonify({'error': 'Failed to generate prediction'}), 500
        
        # Create timestamps for predictions
        now = datetime.datetime.now()
        timestamps = [(now + datetime.timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S') 
                     for i in range(1, hours_ahead + 1)]
        
        # Return predictions with timestamps
        result = {
            'timestamps': timestamps,
            'predictions': predictions.tolist()
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == "__main__":
    prediction_service.load_models()
    app.run(debug=True)
