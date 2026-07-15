from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

MODEL_DIR = 'models'
RESULTS_CSV = os.path.join(MODEL_DIR, 'model_evaluation_results.csv')

def get_available_models():
    if not os.path.exists(MODEL_DIR):
        return []
    return [f.replace('.pkl', '') for f in os.listdir(MODEL_DIR) if f.endswith('.pkl')]

def get_feature_names():
    try:
        data = pd.read_csv('clean_dataset.csv')
        if 'Unnamed: 0' in data.columns:
            data = data.drop(['Unnamed: 0'], axis=1)
        features = data.drop(['quality'], axis=1).columns.tolist()
        features = [pd.Series(f).str.replace('[^A-Za-z0-9_]+', '_', regex=True).iloc[0] for f in features]
        return features
    except Exception:
        # Fallback default features if clean_dataset.csv isn't found
        return ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 
                'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 
                'pH', 'sulphates', 'alcohol']

@app.route('/')
def index():
    # Dashboard home / summary
    models = get_available_models()
    metrics = []
    if os.path.exists(RESULTS_CSV):
        try:
            df = pd.read_csv(RESULTS_CSV)
            metrics = df.to_dict(orient='records')
            # Sort by R2 descending for best performance ranking
            metrics = sorted(metrics, key=lambda x: x.get('R2', -999), reverse=True)
        except Exception:
            pass
    return render_template('dashboard.html', total_models=len(models), metrics=metrics)

def get_feature_metadata():
    """
    Extracts feature names along with their min, max, and median values
    dynamically from the clean dataset to enforce bounds in the UI.
    """
    try:
        data = pd.read_csv('clean_dataset.csv')
        if 'Unnamed: 0' in data.columns:
            data = data.drop(['Unnamed: 0'], axis=1)
            
        features_df = data.drop(['quality'], axis=1)
        
        metadata = []
        for col in features_df.columns:
            # Match the training clean-up step logic
            clean_name = pd.Series(col).str.replace('[^A-Za-z0-9_]+', '_', regex=True).iloc[0]
            
            # Extract statistics rounded to comfortable UI decimals
            metadata.append({
                'name': clean_name,
                'min': float(np.floor(features_df[col].min())),
                'max': float(np.ceil(features_df[col].max())),
                'default': float(round(features_df[col].median(), 2))
            })
        return metadata
    except Exception:
        # Fallback dataset metadata configurations if clean_dataset.csv isn't found
        fallback_features = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 
                             'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 
                             'pH', 'sulphates', 'alcohol']
        return [{'name': f, 'min': 0, 'max': 100, 'default': 0} for f in fallback_features]

@app.route('/predict-page')
def predict_page():
    models = get_available_models()
    features = get_feature_metadata() # <-- Call the updated metadata function
    return render_template('predict.html', models=models, features=features)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        selected_model_name = request.form.get('model_name')
        if not selected_model_name:
            return jsonify({'success': False, 'error': 'No model selected'}), 400

        model_path = os.path.join(MODEL_DIR, f'{selected_model_name}.pkl')
        if not os.path.exists(model_path):
            return jsonify({'success': False, 'error': f'Model {selected_model_name} not found.'}), 404
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        features = get_feature_names()
        input_data = []
        for feature in features:
            val = request.form.get(feature)
            if val is None or val == '':
                return jsonify({'success': False, 'error': f'Missing value for feature: {feature}'}), 400
            input_data.append(float(val))

        input_df = pd.DataFrame([input_data], columns=features)
        prediction = model.predict(input_df)
        output = float(prediction[0])

        return jsonify({
            'success': True,
            'model_used': selected_model_name,
            'prediction': round(output, 3)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/compare')
def compare():
    metrics = []
    if os.path.exists(RESULTS_CSV):
        try:
            df = pd.read_csv(RESULTS_CSV)
            metrics = df.to_dict(orient='records')
        except Exception:
            pass
    return render_template('compare.html', metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)