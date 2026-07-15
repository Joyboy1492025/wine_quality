# Red Wine Quality Prediction System

A comprehensive machine learning project for predicting red wine quality from physicochemical properties. Features multiple regression models, model comparison, and an interactive web dashboard built with Flask.

---

## 📋 Project Overview

This project demonstrates the complete ML pipeline:
- **Data Cleaning & EDA** in Jupyter notebook
- **Model Training** using 10+ regression algorithms
- **Model Comparison** with metrics tracking
- **Interactive Dashboard** for predictions and model analytics

Predicts wine quality scores (0–10) based on 11 physicochemical features like acidity, alcohol content, sulfur dioxide levels, and more.

---

## 📁 Project Structure

```
Red Wine/
├── Main_app.ipynb                 # Data cleaning, EDA, and model training
├── app.py                          # Flask web application
├── model.py                        # Model training & evaluation script
├── requirements.txt                # Python dependencies
├── red_wine_quality.csv            # Raw dataset from UCI
├── clean_dataset.csv               # Processed dataset
├── README.md                       # This file
├── models/                         # Trained model artifacts
│   ├── *.pkl                       # Pickled trained models
│   └── model_evaluation_results.csv # Performance metrics comparison
└── templates/                      # Flask HTML templates
    ├── base.html                   # Base template
    ├── dashboard.html              # Model comparison & stats
    ├── predict.html                # Prediction form
    └── compare.html                # Model performance comparison
```

---

## 🎯 Models Included

The project trains and compares 10+ regression models:

- **LinearRegression** — Baseline linear model
- **Ridge & Lasso** — Regularized linear regression
- **ElasticNet** — Combined L1/L2 regularization
- **Polynomial Regression** — Degree 2 polynomial features
- **Robust Regression (HuberRegressor)** — Resistant to outliers
- **SGDRegressor** — Stochastic gradient descent
- **Artificial Neural Network (MLP)** — Deep learning approach
- **Random Forest** — Ensemble method
- **Support Vector Regression (SVR)** — Kernel-based method
- **XGBoost** — Gradient boosting
- **LightGBM** — Fast gradient boosting

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd "Red Wine"
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Workflow

### Step 1: Data Preparation & Model Training

Run the training script to clean data and train all models:

```bash
python model.py
```

This will:
- Load `red_wine_quality.csv`
- Clean and preprocess the data
- Split into train/test sets
- Train all 10+ models
- Save trained models to `models/` directory
- Generate `model_evaluation_results.csv` with performance metrics
- Create `clean_dataset.csv` for the web app

**Output:**
- `clean_dataset.csv` — Cleaned and processed dataset
- `models/*.pkl` — Trained model files
- `models/model_evaluation_results.csv` — Comparison metrics (R², MAE, MSE)

### Step 2: Explore Data in Notebook (Optional)

Open `Main_app.ipynb` in Jupyter for detailed exploratory data analysis:

```bash
jupyter notebook Main_app.ipynb
```

The notebook includes:
- Data cleaning and preprocessing
- Correlation analysis and visualizations
- Feature distributions and relationships
- Model training walkthrough
- Evaluation metrics and plots

### Step 3: Launch the Web Application

Run the Flask app:

```bash
python app.py
```

Open your browser to **http://127.0.0.1:5000**

---

## 🎨 Web Application Features

### Dashboard (`/`)
- View total number of trained models
- Compare model performance metrics (R², MAE, MSE)
- Sort models by best performance

### Prediction (`/predict`)
- Interactive form for all 11 wine features
- Input validation with min/max ranges
- Real-time quality predictions
- Supports predictions from any trained model

### Model Comparison (`/compare`)
- Side-by-side model performance metrics
- Visual comparison of accuracy scores
- Identify best-performing model

---

## 📈 Data Features

The model uses 11 physicochemical properties to predict wine quality:

1. **Fixed Acidity** — Mostly non-volatile acids
2. **Volatile Acidity** — Amount of acetic acid
3. **Citric Acid** — Adds freshness and flavor
4. **Residual Sugar** — Remaining after fermentation
5. **Chlorides** — Salt content
6. **Free Sulfur Dioxide** — Prevents spoilage
7. **Total Sulfur Dioxide** — Total SO₂ content
8. **Density** — Physical density (g/mL)
9. **pH** — Acidity/basicity level
10. **Sulphates** — Wine additive for preservation
11. **Alcohol** — Alcohol content by volume (%)

**Target:** Quality score (0–10)

---

## 📊 Model Evaluation

Models are evaluated on:
- **R² Score** — Proportion of variance explained
- **Mean Absolute Error (MAE)** — Average prediction error
- **Mean Squared Error (MSE)** — Squared prediction error

Results are saved to `models/model_evaluation_results.csv` and displayed in the dashboard.

---

## 🔧 Configuration

Edit `model.py` to:
- Adjust train/test split ratio
- Change random seed for reproducibility
- Modify hyperparameters for each model
- Add or remove models from training

Edit `app.py` to:
- Change Flask port (default: 5000)
- Modify input validation ranges
- Add new prediction routes

---

## 📚 Dataset

The UCI Wine Quality dataset contains 1,599 red wine samples with physicochemical tests and quality ratings. Download from:
https://archive.ics.uci.edu/ml/datasets/Wine+Quality

- File: `red_wine_quality.csv`
- Size: ~100 KB
- Samples: 1,599
- Features: 11 physicochemical properties
- Target: Quality (0–10)

---

## 📝 Files Reference

| File | Purpose |
|------|---------|
| `model.py` | Train all models and save artifacts |
| `app.py` | Flask web application and API |
| `Main_app.ipynb` | Exploratory data analysis & training notebook |
| `clean_dataset.csv` | Preprocessed dataset (generated) |
| `models/*.pkl` | Trained model files (generated) |
| `models/model_evaluation_results.csv` | Performance comparison (generated) |

---

## 🐛 Troubleshooting

**Models not found:**
- Ensure `model.py` has been run to generate model files
- Check that `models/` directory exists with `.pkl` files

**Dataset not found:**
- Place `red_wine_quality.csv` in the project root
- Run `python model.py` to process the raw dataset

**Feature mismatch errors:**
- Clear the `models/` directory
- Re-run `python model.py` to retrain with current features

**Port already in use:**
- Edit the port in `app.py`: `app.run(port=5001)`
- Or stop the process using port 5000

---

## 📄 License

This is a practice project using the UCI Wine Quality dataset.

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Wine Quality dataset
- Flask for the web framework
- Scikit-learn, XGBoost, and LightGBM for ML algorithms
it on for local development:

```bash
FLASK_DEBUG=1 python app.py
```

---

## Input ranges

The form enforces the same ranges the model was trained on, so predictions
stay within the space the model has actually seen:

| Feature | Range |
|---|---|
| Fixed Acidity | 4.0 – 16.0 g/L |
| Volatile Acidity | 0.1 – 2.0 g/L |
| Citric Acid | 0.0 – 1.0 g/L |
| Residual Sugar | 0.5 – 20.0 g/L |
| Chlorides | 0.01 – 0.6 g/L |
| Free Sulfur Dioxide | 1 – 75 mg/L |
| Total Sulfur Dioxide | 5 – 300 mg/L |
| Density | 0.9850 – 1.0050 g/cm³ |
| pH | 2.7 – 4.0 |
| Sulphates | 0.3 – 2.0 g/L |
| Alcohol | 8.0 – 15.0 % vol |

Both sides validate: the browser blocks obviously invalid values via HTML
`min`/`max`, and `app.py` re-checks everything server-side (never trust the
client alone), returning a clear message naming exactly which field is wrong.

---

## Known bugs that were fixed in this version of `app.py`

| Issue | Fix |
|---|---|
| Raw (unscaled) inputs were fed straight into a model trained on standardized data, silently producing wrong predictions | Inputs are now transformed with the same fitted `scaler` before prediction |
| A validation error wiped the user's typed values back to hard-coded defaults | Submitted values are now echoed back into the form |
| Missing form field raised an uncaught `KeyError` | Explicit, friendly "X is required" message |
| Non-numeric input raised a raw Python conversion error | Friendly "X must be a valid number" message |
| Internal exception text was shown directly to the browser | Logged server-side only; user sees a safe generic message |
| Predicted quality could render as an impossible value (e.g. `-1` or `11`) | Clamped to the valid 0–10 range |
| `debug=True` left on unconditionally (security risk) | Off by default, opt-in via `FLASK_DEBUG=1` |
| App crashed at import if `model.pkl` was missing | Graceful startup check with a clear on-page message instead of a crash |

---

## Notes & limitations

- This is an ordinary least-squares linear regression — a solid, interpretable
  baseline, but wine quality is inherently noisy (it comes from subjective
  human tasting panels), so don't expect a very high R² or perfectly precise
  predictions.
- Predictions outside the trained input ranges are extrapolation and are not
  reliable — that's why the form enforces the ranges above.
- The dataset itself isn't bundled here; if you don't have it, search for the
  "Wine Quality" dataset (red wine variant) from the UCI Machine Learning
  Repository.
