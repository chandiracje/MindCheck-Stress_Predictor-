# Student Stress Level Prediction API

A FastAPI backend that serves your Random Forest, SVM, and MLP models,
letting the caller pick which model to use for each prediction.

## 1. Get your model files from Drive

Download these 4 files from your Google Drive `AIML_Project` folder:

- `random_forest_model.pkl`
- `svm_model.pkl`
- `mlp_model.keras`
- `preprocessor.pkl`

Place all 4 inside the `models/` folder here, so the structure looks like:

```
stress_predictor_api/
├── app.py
├── requirements.txt
├── README.md
└── models/
    ├── random_forest_model.pkl
    ├── svm_model.pkl
    ├── mlp_model.keras
    └── preprocessor.pkl
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

(Use a virtual environment if you can — `python -m venv venv` then activate it —
so this doesn't clash with other Python projects on your machine.)

## 3. Run the API

```bash
uvicorn app:app --reload
```

Then open **http://127.0.0.1:8000/docs** in your browser — this gives you
an interactive Swagger UI where you can test `/predict` directly, no
front end needed yet.

## 4. Test a prediction

In the Swagger UI, click on `POST /predict`, then "Try it out". Pick a
`model` from the dropdown (`random_forest`, `svm`, or `mlp`), fill in
the example values (already pre-filled), and hit Execute.

Or via curl:

```bash
curl -X POST "http://127.0.0.1:8000/predict?model=random_forest" \
  -H "Content-Type: application/json" \
  -d '{
    "anxiety_level": 14, "self_esteem": 20, "mental_health_history": 0,
    "depression": 15, "headache": 3, "blood_pressure": 2, "sleep_quality": 2,
    "breathing_problem": 2, "noise_level": 3, "living_conditions": 3,
    "safety": 3, "basic_needs": 3, "academic_performance": 2, "study_load": 4,
    "teacher_student_relationship": 2, "future_career_concerns": 4,
    "social_support": 1, "peer_pressure": 3, "extracurricular_activities": 2,
    "bullying": 1
  }'
```

## TensorFlow / Python version note

If `pip install tensorflow` fails with something like *"Could not find a
version that satisfies the requirement tensorflow"*, it almost always
means your Python version is too new — TensorFlow lags behind the
newest Python releases (e.g. it doesn't yet support Python 3.13+).

**Check your version:**
```bash
python --version
```

**If it's too new**, the API is already set up to work fine WITHOUT
TensorFlow — Random Forest and SVM will load and predict normally,
the MLP option just won't be offered. `requirements.txt` already has
`tensorflow` commented out for this reason.

**To also enable the MLP model:**
1. Install Python 3.11 alongside your current version (from
   [python.org](https://www.python.org/downloads/release/python-3119/))
2. Create a separate virtual environment using it:
   ```bash
   py -3.11 -m venv venv311
   venv311\Scripts\activate
   pip install -r requirements.txt
   ```
3. Uncomment the `tensorflow` line in `requirements.txt`, then
   `pip install tensorflow` inside that environment

## scikit-learn version mismatch error

If you see an error like:
```
AttributeError: module 'sklearn.compose._column_transformer' has no
attribute '_RemainderColsList'
```

This means your local scikit-learn version doesn't match the version
that trained and saved your models on Colab — the internal file format
changed between versions, so old saved models can fail to load on a
newer (or older) scikit-learn.

**Fix: reinstall the exact matching version.**

1. Check what version Colab used when you saved the models — run this
   in your Colab notebook:
   ```python
   import sklearn
   print(sklearn.__version__)
   ```
2. Update `requirements.txt` here to match that exact version (it's
   currently pinned to `1.6.1`, edit if Colab reports something
   different)
3. Reinstall:
   ```bash
   pip install -r requirements.txt --force-reinstall
   ```

If you keep hitting version drift issues, the safest long-term fix is
re-saving your models directly in the same environment you're
deploying from — i.e. run your training notebook locally too (or check
Colab's scikit-learn version and match it exactly here) so both sides
always agree.

## Important: if you used engineered/composite features

If your final saved models were trained WITH the composite scores
(mental_health_score, physical_score, etc.) rather than just the 20 raw
features, you need to:

1. Uncomment the `ENGINEERED_COLUMNS` list in `app.py`
2. Add the same composite-score calculation inside the `predict()`
   function (the same `.mean(axis=1)` logic you used in your notebook),
   applied to the incoming `row` DataFrame before scaling

If you're not sure which version your saved models expect, check how
many columns `preprocessor` was fit on:

```python
import joblib
preprocessor = joblib.load("models/preprocessor.pkl")
print(preprocessor.feature_names_in_)
```

This will print the exact column names/order your preprocessor (and
therefore your models) expect.

## Next step: front end

Once this API runs locally and returns predictions correctly, the next
step is a simple Streamlit or HTML form that collects a user's answers
to 20 questions and calls this `/predict` endpoint. Ask your assistant
to build that once this part is confirmed working.
