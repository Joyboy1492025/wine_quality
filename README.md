# Wine Quality Predictor

A small end-to-end project that predicts red wine quality (score 0–10) from its
physicochemical properties, using a Multiple Linear Regression model trained
in a Jupyter notebook and served through a Flask web app.

---

## Project structure

```
.
├── Main_app.ipynb        # Data cleaning, EDA, and model training (annotated)
├── app.py                 # Flask backend that loads the model + scaler and serves predictions
├── templates/
│   └── index.html          # Web form (Jinja2 template rendered by Flask)
├── model.pkl               # Trained LinearRegression model  (you generate this)
├── scaler.pkl               # Fitted StandardScaler used at training time (you generate this)
└── red_wine_quality.csv     # Source dataset (not included — see Dataset section)
```

`model.pkl` and `scaler.pkl` are **not shipped** — you generate them yourself by
running the notebook end-to-end (see below). `red_wine_quality.csv` is likewise
not included; place it next to the notebook before running.

---

## How it works

1. **`Main_app.ipynb`** loads the raw wine dataset, cleans it (strips units like
   `g/L`, `mg/L`, `% vol` from the raw text columns via regex), explores it
   (correlation plots, box plots, a pairplot), then:
   - splits it into train/test sets,
   - fits a `StandardScaler` on the training features,
   - trains a `LinearRegression` model on the **scaled** training features,
   - evaluates it on the test set (MSE, R², a residual plot).

   Every code cell has a markdown explanation directly beneath it describing
   what it does, and a few cells call out things worth double-checking (e.g. a
   dtype comparison in the cleaning step, and the fact that the scaler needs
   to be saved for deployment — see below).

2. **`app.py`** loads the trained `model.pkl` **and** the fitted `scaler.pkl`,
   validates form input against sensible physicochemical ranges, scales the
   input exactly the way the training data was scaled, and returns a predicted
   quality score.

3. **`templates/index.html`** is the form the user fills in — one field per
   physicochemical property, each with a min/max range enforced both in the
   browser (HTML `min`/`max`) and again on the server (defense in depth).

---

## Setup

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install flask numpy pandas scikit-learn matplotlib seaborn jupyter
```

### Step 1 — Get the dataset

Place `red_wine_quality.csv` in the same folder as `Main_app.ipynb`
(this is the classic UCI "Wine Quality" red wine dataset — 11 physicochemical
features + a `quality` score).

### Step 2 — Run the notebook and save the model + scaler

Open and run `Main_app.ipynb` top to bottom. It does **not** currently save
`model.pkl` / `scaler.pkl` on its own — add this as a final cell (or run it
manually right after cell 26/28) and execute it:

```python
import pickle
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
```

This is the single most important step — the model was trained on
**standardized** features (via `StandardScaler`), not raw units, so `app.py`
needs that exact fitted scaler to transform new inputs the same way before
prediction. Without `scaler.pkl`, the app will refuse to serve predictions
and will tell you why.

Move both `model.pkl` and `scaler.pkl` into the same folder as `app.py`.

### Step 3 — Run the app

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

By default the app runs with Flask's debugger **off** (safe default). To turn
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
