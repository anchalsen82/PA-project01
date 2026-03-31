"""
Flask API Server for Pump Failure Prediction Dashboard
Serves real-time predictions and sensor data to the web interface.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import threading
import time

from data.generate_data import generate_pump_sensor_data, save_data
from models.predictor import PumpFailurePredictor

app = Flask(__name__, static_folder='dashboard', static_url_path='')
CORS(app)

# Global state
predictor = PumpFailurePredictor()
sensor_data = None
is_model_ready = False
live_readings = []
alerts_log = []


def initialize_system():
    """Generate data, train model on startup if not already there."""
    global sensor_data, is_model_ready, predictor

    print("[INIT] Initializing Predictive Maintenance System...")

    # For Vercel, we can try loading a pre-trained model if it exists
    try:
        if os.path.exists('models/saved/random_forest_model.pkl'):
            predictor.load_model(model_name='random_forest', directory='models/saved')
            # Use small dummy data for len(sensor_data) if it's missing
            sensor_data = pd.DataFrame(columns=predictor.FEATURE_COLUMNS + ['failure'])
            is_model_ready = True
            print("[OK] Loaded pre-trained model.")
            return
    except Exception as e:
        print(f"[WARN] Could not load model: {e}")

    # Generate synthetic data
    print("[DATA] Generating sensor data...")
    sensor_data = generate_pump_sensor_data(n_samples=1000, failure_ratio=0.15) # Smaller for serverless
    save_data(sensor_data)

    # Train model
    print("[ML] Training Random Forest model...")
    metrics = predictor.train(sensor_data, model_name='random_forest')
    
    # Skip saving on Vercel as it is read-only
    try:
        predictor.save_model()
    except:
        pass

    is_model_ready = True
    print(f"[OK] System ready! Model accuracy: {metrics['test_accuracy']:.2%}")


def generate_live_reading():
    """Simulate a live sensor reading."""
    pump_id = np.random.choice(['PUMP-001', 'PUMP-002', 'PUMP-003', 'PUMP-004', 'PUMP-005'])

    # 85% chance of normal, 15% chance of anomaly
    is_anomaly = np.random.random() < 0.15

    if is_anomaly:
        reading = {
            'timestamp': datetime.now().isoformat(),
            'pump_id': pump_id,
            'temperature_C': round(np.random.normal(92, 8), 2),
            'vibration_mm_s': round(np.random.normal(7.0, 1.5), 2),
            'pressure_bar': round(np.random.normal(2.0, 0.4), 2),
            'flow_rate_L_min': round(np.random.normal(75, 12), 2),
            'rpm': round(np.random.normal(1220, 60), 1),
            'power_kW': round(np.random.normal(21, 2.5), 2),
            'humidity_pct': round(np.random.normal(62, 8), 2),
            'noise_level_dB': round(np.random.normal(90, 5), 2)
        }
    else:
        reading = {
            'timestamp': datetime.now().isoformat(),
            'pump_id': pump_id,
            'temperature_C': round(np.random.normal(65, 5), 2),
            'vibration_mm_s': round(np.random.normal(2.5, 0.8), 2),
            'pressure_bar': round(np.random.normal(3.5, 0.3), 2),
            'flow_rate_L_min': round(np.random.normal(120, 10), 2),
            'rpm': round(np.random.normal(1450, 30), 1),
            'power_kW': round(np.random.normal(15, 1.5), 2),
            'humidity_pct': round(np.random.normal(45, 8), 2),
            'noise_level_dB': round(np.random.normal(72, 4), 2)
        }

    return reading


# ===== API Routes =====

@app.route('/')
def serve_dashboard():
    return send_from_directory('dashboard', 'index.html')


@app.route('/api/status')
def system_status():
    return jsonify({
        'status': 'ready' if is_model_ready else 'initializing',
        'model': predictor.model_name if is_model_ready else None,
        'total_samples': len(sensor_data) if sensor_data is not None else 0,
        'uptime': datetime.now().isoformat()
    })


@app.route('/api/metrics')
def get_metrics():
    if not is_model_ready:
        return jsonify({'error': 'Model not ready'}), 503
    return jsonify(predictor.training_metrics)


@app.route('/api/predict', methods=['POST'])
def predict():
    if not is_model_ready:
        return jsonify({'error': 'Model not ready'}), 503

    data = request.json
    result = predictor.predict(data)

    # Log alert if needed
    if result.get('alert'):
        alert_entry = {
            **result,
            'pump_id': data.get('pump_id', 'Unknown'),
            'sensor_data': data
        }
        alerts_log.append(alert_entry)
        # Keep only last 100 alerts
        if len(alerts_log) > 100:
            alerts_log.pop(0)

    return jsonify(result)


@app.route('/api/live-reading')
def get_live_reading():
    """Get a simulated live sensor reading with prediction."""
    if not is_model_ready:
        return jsonify({'error': 'Model not ready'}), 503

    reading = generate_live_reading()
    prediction = predictor.predict(reading)

    response = {
        'sensor_data': reading,
        'prediction': prediction
    }

    # Store live reading
    live_readings.append(response)
    if len(live_readings) > 200:
        live_readings.pop(0)

    # Log alert
    if prediction.get('alert'):
        alerts_log.append({
            **prediction,
            'pump_id': reading['pump_id'],
            'sensor_data': reading
        })
        if len(alerts_log) > 100:
            alerts_log.pop(0)

    return jsonify(response)


@app.route('/api/live-history')
def get_live_history():
    """Get recent live readings history."""
    count = request.args.get('count', 50, type=int)
    return jsonify(live_readings[-count:])


@app.route('/api/alerts')
def get_alerts():
    """Get recent alerts."""
    count = request.args.get('count', 20, type=int)
    return jsonify(alerts_log[-count:])


@app.route('/api/sensor-stats')
def get_sensor_stats():
    """Get statistical summary of sensor data."""
    if sensor_data is None:
        return jsonify({'error': 'No data available'}), 503

    features = predictor.FEATURE_COLUMNS
    stats = {}
    for feature in features:
        col = sensor_data[feature].dropna()
        stats[feature] = {
            'mean': round(float(col.mean()), 2),
            'std': round(float(col.std()), 2),
            'min': round(float(col.min()), 2),
            'max': round(float(col.max()), 2),
            'median': round(float(col.median()), 2)
        }

    # Distribution of failure by pump
    pump_failure = sensor_data.groupby('pump_id')['failure'].mean().round(3).to_dict()

    return jsonify({
        'feature_stats': stats,
        'pump_failure_rates': pump_failure,
        'total_records': len(sensor_data),
        'failure_count': int(sensor_data['failure'].sum()),
        'normal_count': int((sensor_data['failure'] == 0).sum())
    })


@app.route('/api/data-distribution')
def get_data_distribution():
    """Get data distribution for charts."""
    if sensor_data is None:
        return jsonify({'error': 'No data available'}), 503

    features = predictor.FEATURE_COLUMNS
    distributions = {}

    for feature in features:
        normal_data = sensor_data[sensor_data['failure'] == 0][feature].dropna()
        failure_data = sensor_data[sensor_data['failure'] == 1][feature].dropna()

        # Create histogram bins
        all_data = sensor_data[feature].dropna()
        bins = np.linspace(all_data.min(), all_data.max(), 30)
        
        normal_hist, _ = np.histogram(normal_data, bins=bins)
        failure_hist, _ = np.histogram(failure_data, bins=bins)

        distributions[feature] = {
            'bins': np.round((bins[:-1] + bins[1:]) / 2, 2).tolist(),
            'normal': normal_hist.tolist(),
            'failure': failure_hist.tolist()
        }

    return jsonify(distributions)


@app.route('/api/feature-importance')
def get_feature_importance():
    """Get feature importance from the trained model."""
    if not is_model_ready or predictor.feature_importances is None:
        return jsonify({'error': 'Feature importances not available'}), 503

    sorted_features = sorted(
        predictor.feature_importances.items(),
        key=lambda x: x[1], reverse=True
    )

    return jsonify({
        'features': [f[0] for f in sorted_features],
        'importances': [round(f[1], 4) for f in sorted_features]
    })


@app.route('/api/compare-models', methods=['POST'])
def compare_models():
    """Compare all available ML models."""
    if sensor_data is None:
        return jsonify({'error': 'No data available'}), 503

    comparison = predictor.compare_models(sensor_data)
    return jsonify(comparison.to_dict(orient='records'))


@app.route('/api/pump-status')
def get_pump_status():
    """Get current status of all pumps."""
    if not is_model_ready:
        return jsonify({'error': 'Model not ready'}), 503

    pumps = ['PUMP-001', 'PUMP-002', 'PUMP-003', 'PUMP-004', 'PUMP-005']
    statuses = []

    for pump_id in pumps:
        reading = generate_live_reading()
        reading['pump_id'] = pump_id
        prediction = predictor.predict(reading)

        statuses.append({
            'pump_id': pump_id,
            'sensor_data': reading,
            'prediction': prediction
        })

    return jsonify(statuses)


# Initialize early for Vercel
initialize_system()

if __name__ == '__main__':
    print("\n[START] Starting Predictive Maintenance Dashboard...")
    print("        Open http://localhost:5000 in your browser\n")
    app.run(debug=False, host='0.0.0.0', port=5000)
