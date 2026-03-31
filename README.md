# 🏭 PumpGuard AI — Predictive Maintenance System

A **Machine Learning-based predictive maintenance system** for industrial pump failure detection. The system collects sensor data, trains ML models, and provides real-time failure predictions through a premium web dashboard.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)
![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)

---

## ✨ Features

- **8 Sensor Inputs**: Temperature, Vibration, Pressure, Flow Rate, RPM, Power, Humidity, Noise Level
- **4 ML Models**: Logistic Regression, Random Forest, SVM, Gradient Boosting
- **Real-time Dashboard**: Live sensor monitoring with auto-refresh
- **Failure Prediction**: Binary classification with probability and risk levels
- **Alert System**: Automatic alerts when failure is predicted (LOW/MEDIUM/HIGH/CRITICAL)
- **Data Visualization**: Distribution charts, ROC curves, feature importance, correlation heatmaps
- **Manual Prediction**: Input custom sensor values and get instant predictions
- **Model Comparison**: Side-by-side evaluation of all trained models

---

## 📁 Project Structure

```
PA/
├── app.py                               # Flask API server
├── requirements.txt                     # Python dependencies
├── Predictive_Maintenance_Pipeline.ipynb # Jupyter Notebook (full ML pipeline)
├── README.md
├── data/
│   ├── __init__.py
│   ├── generate_data.py                 # Synthetic data generator
│   └── pump_sensor_data.csv             # Generated sensor data (created at runtime)
├── models/
│   ├── __init__.py
│   ├── predictor.py                     # ML prediction engine
│   └── saved/                           # Saved model files (created at runtime)
└── dashboard/
    ├── index.html                       # Web dashboard
    ├── style.css                        # Premium dark theme styles
    └── app.js                           # Dashboard logic & charts
```

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Jupyter Notebook (Optional — for analysis)

```bash
jupyter notebook Predictive_Maintenance_Pipeline.ipynb
```

### 3. Launch the Web Dashboard

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| **Dashboard** | KPI overview, pump fleet status, failure distribution, feature importance |
| **Live Monitor** | Real-time sensor gauges, live predictions, temperature/vibration timeline |
| **Analytics** | Data distributions, ROC curve, sensor statistics table |
| **Predictions** | Manual prediction form — input sensor values, get instant results |
| **Alerts** | Alert history log with risk levels and timestamps |
| **Model Info** | Model performance metrics, confusion matrix, classification report |

---

## 🤖 Machine Learning Pipeline

1. **Data Generation**: 5000 synthetic sensor readings with 15% failure rate
2. **Preprocessing**: Median imputation, outlier removal, StandardScaler normalization
3. **Feature Selection**: 8 sensor features
4. **Model Training**: 4 models with hyperparameter tuning
5. **Evaluation**: Accuracy, Precision, Recall, F1, ROC AUC, 5-fold Cross-Validation
6. **Prediction**: Binary classification + failure probability + 4-level risk assessment

---

## 🔮 Optional Extensions

- **IoT Integration**: Connect Arduino/Raspberry Pi sensors via serial port
- **Cloud Deployment**: Deploy on AWS/Azure/GCP for remote monitoring
- **Time-Series Forecasting**: Add LSTM/ARIMA for trend prediction
- **Anomaly Detection**: Unsupervised methods (Isolation Forest, Autoencoders)

---

## 📦 Technologies Used

- **Python 3.8+**
- **Pandas & NumPy** — Data processing
- **Scikit-learn** — Machine learning models
- **Matplotlib & Seaborn** — Visualizations (notebook)
- **Chart.js** — Interactive web charts
- **Flask** — API server
- **HTML/CSS/JS** — Premium web dashboard

---

## 📄 License

This project is for educational purposes. MIT License.
