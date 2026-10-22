from flask import Flask, render_template_string, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained linear regression model (ensure your file is named 'model.pkl' or change path below)
MODEL_PATH = 'SVC.pkl'
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# HTML template with modern design, soft shadows, and responsive layout
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Insurance Cost Predictor</title>
    <style>
        :root {
            --primary: #2563eb;
            --primary-hover: #1d4ed8;
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #1e293b;
            --text-muted: #64748b;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 40px 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            width: 100%;
            max-width: 650px;
        }
        .card {
            background: var(--card-bg);
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.05), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        }
        h2 {
            margin-top: 0;
            text-align: center;
            color: var(--text-main);
            font-size: 1.8rem;
            margin-bottom: 8px;
        }
        .subtitle {
            text-align: center;
            color: var(--text-muted);
            margin-bottom: 30px;
            font-size: 0.95rem;
        }
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media(max-width: 600px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
        }
        .form-group {
            margin-bottom: 16px;
        }
        label {
            display: block;
            margin-bottom: 6px;
            font-weight: 600;
            font-size: 0.85rem;
            color: #334155;
        }
        input, select {
            width: 100%;
            padding: 12px;
            border: 1px solid #cbd5e1;
            border-radius: 10px;
            font-size: 0.95rem;
            box-sizing: border-box;
            transition: all 0.2s ease;
            background-color: #f8fafc;
        }
        input:focus, select:focus {
            outline: none;
            border-color: var(--primary);
            background-color: #fff;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15);
        }
        .btn-submit {
            width: 100%;
            background-color: var(--primary);
            color: white;
            padding: 14px;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
            transition: all 0.2s ease;
            margin-top: 10px;
        }
        .btn-submit:hover {
            background-color: var(--primary-hover);
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45);
            transform: translateY(-1px);
        }
        .result-card {
            background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
            color: white;
            padding: 24px;
            border-radius: 14px;
            text-align: center;
            box-shadow: 0 10px 25px rgba(37, 99, 235, 0.3);
            margin-top: 25px;
        }
        .result-card h3 {
            margin: 0;
            font-weight: 400;
            font-size: 0.95rem;
            opacity: 0.9;
        }
        .result-card h1 {
            margin: 10px 0 0 0;
            font-size: 2.2rem;
            font-weight: 700;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h2>🏥 Insurance Cost Predictor</h2>
            <p class="subtitle">Enter your details below to estimate medical charges</p>
            
            <form method="POST">
                <div class="form-grid">
                    <div class="form-group">
                        <label>Age</label>
                        <input type="number" name="age" value="{{ request.form.get('age', 30) }}" required>
                    </div>
                    <div class="form-group">
                        <label>Sex (1: Male, 0: Female)</label>
                        <select name="sex">
                            <option value="1" {% if request.form.get('sex') == '1' %}selected{% endif %}>Male (1)</option>
                            <option value="0" {% if request.form.get('sex') == '0' %}selected{% endif %}>Female (0)</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>BMI</label>
                        <input type="number" step="0.1" name="bmi" value="{{ request.form.get('bmi', 25.0) }}" required>
                    </div>
                    <div class="form-group">
                        <label>Children</label>
                        <input type="number" name="children" value="{{ request.form.get('children', 0) }}" required>
                    </div>
                    <div class="form-group">
                        <label>Smoker (1: Yes, 0: No)</label>
                        <select name="smoker">
                            <option value="0" {% if request.form.get('smoker') == '0' %}selected{% endif %}>No (0)</option>
                            <option value="1" {% if request.form.get('smoker') == '1' %}selected{% endif %}>Yes (1)</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Region (0-3)</label>
                        <input type="number" min="0" max="3" name="region" value="{{ request.form.get('region', 0) }}" required>
                    </div>
                </div>
                
                <button type="submit" class="btn-submit">Calculate Prediction</button>
            </form>

            {% if prediction is not none %}
            <div class="result-card">
                <h3>Estimated Charges</h3>
                <h1>₹ {{ "{:,.2f}".format(prediction) }}</h1>
            </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Features mapped directly in order: age, sex, bmi, children, smoker, region[cite: 4]
            features = np.array([[
                float(request.form['age']),
                float(request.form['sex']),
                float(request.form['bmi']),
                float(request.form['children']),
                float(request.form['smoker']),
                float(request.form['region'])
            ]])
            
            prediction = model.predict(features)[0]
        except Exception as e:
            prediction = None

    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
