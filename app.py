from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

try:
    # Load and preprocess data
    df = pd.read_csv("FINAL_USO.csv")
    df.columns = [col.strip() for col in df.columns]
    df = df.dropna()
    logger.info("Dataset loaded and preprocessed successfully")
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    raise

# Regression model for price prediction
X_reg = df[['Open', 'Volume', 'SP_close', 'DJ_close']]
y_reg = df['Close']
regression_model = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
regression_model.fit(X_train_reg, y_train_reg)
logger.info("Regression model trained successfully")

# Classification model for trend prediction
X_clf = df[['Open', 'Close', 'Volume']]
y_clf = df['SF_Trend']
scaler_clf = StandardScaler()
X_clf_scaled = scaler_clf.fit_transform(X_clf)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_clf_scaled, y_clf, test_size=0.2, random_state=42)
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(Xc_train, yc_train)
logger.info("Classification model trained successfully")

@app.route('/')
def home():
    logger.info("Rendering main home page")
    return render_template('home.html')

@app.route('/price')
def price():
    logger.info("Rendering price prediction page")
    return render_template('index.html')

@app.route('/trend')
def trend():
    logger.info("Rendering trend prediction page")
    return render_template('index1.html')

@app.route('/predict_price', methods=['POST'])
def predict_price():
    try:
        open_val = float(request.form['open'])
        volume = float(request.form['volume'])
        sp_close = float(request.form['sp_close'])
        dj_close = float(request.form['dj_close'])
        currency = request.form['currency']

        if any(val < 0 for val in [open_val, volume, sp_close, dj_close]):
            logger.warning("Negative input values detected")
            return render_template('result.html', error="Input tidak boleh negatif")

        input_df = pd.DataFrame([[open_val, volume, sp_close, dj_close]],
                                columns=['Open', 'Volume', 'SP_close', 'DJ_close'])
        pred = regression_model.predict(input_df)[0]
        pred = round(pred, 2)

        if currency == "IDR":
            pred *= 15000
            result = f"Harga Prediksi: Rp {pred:,.2f}"
        else:
            result = f"Harga Prediksi: ${pred:.2f}"

        logger.info(f"Price prediction successful: {result}")
        return render_template('result.html', prediction=result)

    except ValueError:
        logger.error("Invalid input format")
        return render_template('result.html', error="Masukkan nilai numerik yang valid")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return render_template('result.html', error="Terjadi kesalahan dalam prediksi")

@app.route('/predict_trend', methods=['POST'])
def predict_trend():
    try:
        open_val = float(request.form['open'])
        close = float(request.form['close'])
        volume = float(request.form['volume'])

        if any(val < 0 for val in [open_val, close, volume]):
            logger.warning("Negative input values detected")
            return render_template('result.html', error="Input tidak boleh negatif")

        input_data = np.array([[open_val, close, volume]])
        input_scaled = scaler_clf.transform(input_data)
        prediction = decision_tree_model.predict(input_scaled)[0]

        if prediction == 1:
            result = "Prediksi Tren: Bullish 🚀 (Naik)"
        else:
            result = "Prediksi Tren: Bearish 📉 (Turun)"

        logger.info(f"Trend prediction successful: {result}")
        return render_template('result.html', prediction=result)

    except ValueError:
        logger.error("Invalid input format")
        return render_template('result.html', error="Masukkan nilai numerik yang valid")
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return render_template('result.html', error="Terjadi kesalahan dalam prediksi")

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=True)