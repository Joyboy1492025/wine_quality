import os
import pickle
import logging

import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# --------------------------------------------------------------------------
# Model / scaler loading
# --------------------------------------------------------------------------
# BUG FIX (critical): the notebook trains LinearRegression on features that
# were transformed with StandardScaler (scaler.fit_transform(x_train)).
# The original app.py fed RAW, un-scaled numbers straight into model.predict(),
# which silently produces wrong predictions (a LinearRegression trained on
# z-scores has completely different coefficients than one trained on raw
# units). We now load the fitted scaler and transform inputs the exact same
# way before calling predict(). If scaler.pkl is missing, we fail fast with a
# clear message instead of quietly returning bad numbers.
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")

model = None
scaler = None
startup_error = None

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    startup_error = f"Model file not found at '{MODEL_PATH}'. Train and pickle the model first."
except Exception as e:  # pragma: no cover - defensive
    startup_error = f"Failed to load model.pkl: {e}"

try:
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    startup_error = (startup_error + " " if startup_error else "") + (
        f"Scaler file not found at '{SCALER_PATH}'. In the notebook, after "
        "`scaler = StandardScaler(); scaler.fit_transform(x_train)`, add "
        "`pickle.dump(scaler, open('scaler.pkl', 'wb'))` and re-export it "
        "next to model.pkl, otherwise predictions will be wrong."
    )
except Exception as e:  # pragma: no cover - defensive
    startup_error = (startup_error + " " if startup_error else "") + f"Failed to load scaler.pkl: {e}"

if startup_error:
    logging.error(startup_error)

# --------------------------------------------------------------------------
# Feature configuration
# --------------------------------------------------------------------------
# Order matters: it must match the column order the model was trained on
# (x = data_clean.drop(columns='quality') in the notebook -> the original
# CSV column order). Kept identical to the previous version.
FEATURE_LIMITS = {
    "fixed_acidity": {"min": 4.0, "max": 16.0, "label": "Fixed Acidity", "default": 7.4, "step": 0.1},
    "volatile_acidity": {"min": 0.1, "max": 2.0, "label": "Volatile Acidity", "default": 0.70, "step": 0.01},
    "citric_acid": {"min": 0.0, "max": 1.0, "label": "Citric Acid", "default": 0.00, "step": 0.01},
    "residual_sugar": {"min": 0.5, "max": 20.0, "label": "Residual Sugar", "default": 1.9, "step": 0.1},
    "chlorides": {"min": 0.01, "max": 0.6, "label": "Chlorides", "default": 0.076, "step": 0.001},
    "free_sulfur_dioxide": {"min": 1.0, "max": 75.0, "label": "Free Sulfur Dioxide", "default": 11, "step": 1},
    "total_sulfur_dioxide": {"min": 5.0, "max": 300.0, "label": "Total Sulfur Dioxide", "default": 34, "step": 1},
    "density": {"min": 0.9850, "max": 1.0050, "label": "Density", "default": 0.9978, "step": 0.0001},
    "ph": {"min": 2.7, "max": 4.0, "label": "pH", "default": 3.51, "step": 0.01},
    "sulphates": {"min": 0.3, "max": 2.0, "label": "Sulphates", "default": 0.56, "step": 0.01},
    "alcohol": {"min": 8.0, "max": 15.0, "label": "Alcohol", "default": 9.4, "step": 0.1},
}

# Quality scores in the underlying dataset run 0-10; clamp display so a
# borderline linear-regression output (e.g. 10.4 or -0.3) never shows an
# impossible class to the user.
QUALITY_MIN, QUALITY_MAX = 0, 10


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    rounded_prediction = None
    error_message = None

    # BUG FIX: previously, after a POST the form always redrew the hard-coded
    # default values (7.4, 0.70, ...) instead of what the user typed, so a
    # validation error wiped out their input. We now always echo back
    # whatever was submitted (or the defaults on first GET).
    form_values = {key: str(bounds.get("default", "")) for key, bounds in FEATURE_LIMITS.items()}

    if startup_error:
        error_message = "The prediction service is not available: " + startup_error
        return render_template(
            "index.html",
            prediction=None,
            rounded_prediction=None,
            error_message=error_message,
            form_values=form_values,
            limits=FEATURE_LIMITS,
        )

    if request.method == "POST":
        # Keep whatever the user typed, even on error, so the form doesn't
        # silently reset.
        form_values = {key: request.form.get(key, "") for key in FEATURE_LIMITS}

        try:
            feature_values = []
            for key, bounds in FEATURE_LIMITS.items():
                raw = request.form.get(key)

                # BUG FIX: request.form[key] raised an uncaught KeyError
                # (surfaced as a generic "System Error") if a field was
                # missing from the submission. Validate presence explicitly.
                if raw is None or raw.strip() == "":
                    raise ValueError(f"{bounds['label']} is required.")

                # BUG FIX: bare float(raw) raised a raw Python
                # "could not convert string to float: '...'" message on
                # non-numeric input. Give a user-friendly message instead.
                try:
                    val = float(raw)
                except ValueError:
                    raise ValueError(f"{bounds['label']} must be a valid number.")

                if val < bounds["min"] or val > bounds["max"]:
                    raise ValueError(
                        f"{bounds['label']} must be between {bounds['min']} and {bounds['max']}."
                    )

                feature_values.append(val)

            # 2D array, single row, in the exact training column order.
            input_array = np.array([feature_values])

            # BUG FIX (critical): scale with the SAME fitted scaler used at
            # training time, instead of predicting on raw units.
            input_scaled = scaler.transform(input_array)
            raw_pred = model.predict(input_scaled)[0]

            prediction = round(float(raw_pred), 2)
            rounded_prediction = int(np.clip(round(raw_pred), QUALITY_MIN, QUALITY_MAX))

        except ValueError as ve:
            error_message = str(ve)
        except Exception as e:
            # Don't leak internal exception details/tracebacks to the client;
            # log them server-side instead.
            logging.exception("Unexpected error during prediction")
            error_message = "Something went wrong while generating the prediction. Please try again."

    return render_template(
        "index.html",
        prediction=prediction,
        rounded_prediction=rounded_prediction,
        error_message=error_message,
        form_values=form_values,
        limits=FEATURE_LIMITS,
    )


if __name__ == "__main__":
    # BUG FIX: debug=True should never be used outside local development
    # (it exposes the Werkzeug interactive debugger / arbitrary code
    # execution if reached from outside). Drive it from an environment
    # variable so production runs are safe by default.
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug_mode)
