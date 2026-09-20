"""
Student Stress Level Prediction API
-------------------------------------
Loads the trained SVM, MLP, KNN, Logistic Regression, Random Forest (and,
once trained, Naive Bayes) models plus the shared preprocessing artifacts,
and exposes a /predict endpoint that lets the caller choose which model
to use.

Three different preprocessing pipelines feed the models, depending on
which dataset each model was trained on:

  * PCA pipeline              -> SVM, MLP
      scale (StandardScaler) -> PCA(n_components=2) on the 20 raw features
  * ANOVA-selected pipeline   -> KNN, Logistic Regression
      scale (StandardScaler) -> select the 10 ANOVA-selected raw columns
  * Embedded-selected pipeline -> Random Forest
      scale (StandardScaler) -> compute 5 composite/engineered scores
      -> select the 10 embedded (RF-importance) selected columns

Run locally with:
    uvicorn app:app --reload

Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from enum import Enum
import joblib
import numpy as np
import pandas as pd
import os
import json
import time
import collections
import httpx

# ---------------------------------------------------------------------
# 1. Define the exact feature order your models were trained on.
#    This MUST match the column order used when you built X_train in
#    your notebook (i.e. stressLvlDf.drop(columns=["stress_level"]).columns)
# ---------------------------------------------------------------------
FEATURE_COLUMNS = [
    "anxiety_level",
    "self_esteem",
    "mental_health_history",
    "depression",
    "headache",
    "blood_pressure",
    "sleep_quality",
    "breathing_problem",
    "noise_level",
    "living_conditions",
    "safety",
    "basic_needs",
    "academic_performance",
    "study_load",
    "teacher_student_relationship",
    "future_career_concerns",
    "social_support",
    "peer_pressure",
    "extracurricular_activities",
    "bullying",
]

# The preprocessor's ColumnTransformer scales every column EXCEPT
# mental_health_history (passed through unscaled). Its .transform()
# output comes back in this order: scaled numeric columns first, then
# the passthrough column last — NOT the same order as FEATURE_COLUMNS.
NUMERIC_COLS = [c for c in FEATURE_COLUMNS if c != "mental_health_history"]
PREPROCESSOR_OUTPUT_ORDER = NUMERIC_COLS + ["mental_health_history"]

# Composite/engineered features, computed from the SCALED values (mean of
# already-standardized columns). Only the embedded/RF pipeline needs these
# (its top-10 list includes social_score and physical_score); the ANOVA
# and PCA pipelines are built from the 20 raw scaled columns only.
ENGINEERED_COLUMNS = [
    "mental_health_score",
    "physical_score",
    "environment_score",
    "academic_score",
    "social_score",
]

FEATURE_GROUPS = {
    "mental_health_score": ["anxiety_level", "self_esteem", "mental_health_history", "depression"],
    "physical_score": ["headache", "blood_pressure", "sleep_quality", "breathing_problem"],
    "environment_score": ["noise_level", "living_conditions", "safety", "basic_needs"],
    "academic_score": ["academic_performance", "study_load", "teacher_student_relationship", "future_career_concerns"],
    "social_score": ["social_support", "peer_pressure", "extracurricular_activities", "bullying"],
}

# ---------------------------------------------------------------------
# Which pipeline each model was trained on. Verified against each saved
# model's `feature_names_in_` / `n_features_in_`, NOT just assumed —
# note that KNN turned out to be ANOVA-selected, not PCA.
#   "pca"      -> scale -> PCA(2 components) on the 20 raw features
#   "anova"    -> scale -> select the 10 ANOVA-selected raw columns
#   "embedded" -> scale -> engineer scores -> select the 10 embedded columns
# ---------------------------------------------------------------------
MODEL_PIPELINE = {
    "svm": "pca",
    "mlp": "pca",
    "knn": "anova",
    "logistic_regression": "anova",
    "random_forest": "embedded",
    "naive_bayes": "pca",
}

# Filenames on disk for each model, inside MODEL_DIR.
MODEL_FILENAMES = {
    "random_forest": "random_forest_model.pkl",
    "svm": "svm_model.pkl",
    "knn": "knn_model.pkl",
    "logistic_regression": "logistic_regression_model.pkl",
    "mlp": "mlp_tuned.pkl",
    "naive_bayes": "naive_bayes.pkl",
}

# ---------------------------------------------------------------------
# Human-readable spec for each of the 20 raw features, shared by the
# /chat endpoint's tool schema and system prompt (mirrors the QUESTIONS
# array in static/index.html so the AI and the manual form agree).
# ---------------------------------------------------------------------
FEATURE_SPEC = [
    {"key": "anxiety_level", "chapter": "Mind", "min": 0, "max": 21,
     "desc": "Intensity of the student's anxiety recently (0=calm, 21=overwhelming)."},
    {"key": "self_esteem", "chapter": "Mind", "min": 0, "max": 30,
     "desc": "Student's current sense of self-worth/confidence (0=low, 30=high)."},
    {"key": "mental_health_history", "chapter": "Mind", "min": 0, "max": 1,
     "desc": "Whether they have previously experienced a mental health challenge (0=no, 1=yes)."},
    {"key": "depression", "chapter": "Mind", "min": 0, "max": 27,
     "desc": "How heavy/low their mood has been lately (0=light, 27=heavy)."},
    {"key": "headache", "chapter": "Body", "min": 0, "max": 5,
     "desc": "How often they get headaches (0=never, 5=constantly)."},
    {"key": "blood_pressure", "chapter": "Body", "min": 1, "max": 3,
     "desc": "Blood pressure category (1=low, 2=normal, 3=high)."},
    {"key": "sleep_quality", "chapter": "Body", "min": 0, "max": 5,
     "desc": "Quality of their sleep (0=poor, 5=excellent)."},
    {"key": "breathing_problem", "chapter": "Body", "min": 0, "max": 5,
     "desc": "How often they notice breathing difficulties (0=never, 5=often)."},
    {"key": "noise_level", "chapter": "Surroundings", "min": 0, "max": 5,
     "desc": "How noisy their usual living/study space is (0=quiet, 5=very noisy)."},
    {"key": "living_conditions", "chapter": "Surroundings", "min": 0, "max": 5,
     "desc": "Overall quality of their living conditions (0=difficult, 5=comfortable)."},
    {"key": "safety", "chapter": "Surroundings", "min": 0, "max": 5,
     "desc": "How safe they feel in their daily environment (0=unsafe, 5=very safe)."},
    {"key": "basic_needs", "chapter": "Surroundings", "min": 0, "max": 5,
     "desc": "How well their basic needs are being met (0=struggling, 5=fully met)."},
    {"key": "academic_performance", "chapter": "Academics", "min": 0, "max": 5,
     "desc": "Satisfaction with academic performance (0=unsatisfied, 5=very satisfied)."},
    {"key": "study_load", "chapter": "Academics", "min": 0, "max": 5,
     "desc": "How heavy their study workload feels (0=light, 5=overwhelming)."},
    {"key": "teacher_student_relationship", "chapter": "Academics", "min": 0, "max": 5,
     "desc": "How positive their relationship with teachers is (0=strained, 5=very positive)."},
    {"key": "future_career_concerns", "chapter": "Academics", "min": 0, "max": 5,
     "desc": "How much they worry about their future career (0=rarely, 5=constantly)."},
    {"key": "social_support", "chapter": "People", "min": 0, "max": 3,
     "desc": "How much support they feel from friends/family (0=little, 3=a lot)."},
    {"key": "peer_pressure", "chapter": "People", "min": 0, "max": 5,
     "desc": "How much pressure they feel from peers (0=none, 5=a great deal)."},
    {"key": "extracurricular_activities", "chapter": "People", "min": 0, "max": 5,
     "desc": "How involved they are in extracurricular activities (0=not involved, 5=very involved)."},
    {"key": "bullying", "chapter": "People", "min": 0, "max": 5,
     "desc": "How much bullying they've experienced recently (0=never, 5=frequently)."},
]
FEATURE_SPEC_BY_KEY = {f["key"]: f for f in FEATURE_SPEC}

STRESS_LABELS = {0: "Low Stress", 1: "Medium Stress", 2: "High Stress"}
# NOTE: this maps StressLevelDataset.csv's stress_level (0/1/2) to
# Low/Medium/High — the commonly used interpretation for this dataset,
# but not something explicitly confirmed from official documentation.

# Evaluation results from your notebook — used to power the performance
# comparison chart in the UI (GET /metrics). Update these with your
# actual final numbers once you have them for each retrained model.
MODEL_METRICS = {
    "random_forest": {"label": "Random Forest",        "test_accuracy": None, "cv_accuracy": None},
    "svm":           {"label": "SVM",                  "test_accuracy": None, "cv_accuracy": None},
    "mlp":           {"label": "Neural Net",           "test_accuracy": None, "cv_accuracy": None},
    "knn":           {"label": "KNN",                  "test_accuracy": None, "cv_accuracy": None},
    "logistic_regression": {"label": "Logistic Regression", "test_accuracy": None, "cv_accuracy": None},
    "naive_bayes": {"label": "Naive Bayes", "test_accuracy": None, "cv_accuracy": None},
}

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# ---------------------------------------------------------------------
# 2. Load models + preprocessing artifacts once, at startup (not per-request)
# ---------------------------------------------------------------------
app = FastAPI(
    title="Student Stress Level Prediction API",
    description="Predicts stress level (Low/Medium/High) from 20 psychosocial indicators.",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_models = {}
_preprocessor = None
_pca = None
_anova_features = None      # list[str] — ANOVA-selected 10 raw columns (KNN, Logistic Regression)
_embedded_features = None   # list[str] — embedded/RF-selected 10 columns (Random Forest)


@app.on_event("startup")
def load_artifacts():
    global _preprocessor, _pca, _anova_features, _embedded_features

    try:
        _preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))
    except FileNotFoundError:
        print("WARNING: preprocessor.pkl not found — no models can predict until it's added.")
        return

    try:
        _pca = joblib.load(os.path.join(MODEL_DIR, "pca_model.pkl"))
    except FileNotFoundError:
        print("WARNING: pca_model.pkl not found — SVM/MLP can't predict until it's added.")

    try:
        _anova_features = joblib.load(os.path.join(MODEL_DIR, "anova_top10_features.pkl"))
        print(f"ANOVA-selected features: {_anova_features}")
    except FileNotFoundError:
        print("WARNING: anova_top10_features.pkl not found — KNN/Logistic Regression can't predict until it's added.")

    try:
        _embedded_features = joblib.load(os.path.join(MODEL_DIR, "top10_features_list.pkl"))
        print(f"Embedded (RF-importance) selected features: {_embedded_features}")
    except FileNotFoundError:
        print("WARNING: top10_features_list.pkl not found — Random Forest can't predict until it's added.")

    # Each model loads independently — a missing file just skips that one
    # model instead of breaking the whole app. All models (including MLP,
    # which is a scikit-learn MLPClassifier, not Keras) load via joblib.
    for model_id, filename in MODEL_FILENAMES.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            _models[model_id] = joblib.load(path)
        else:
            print(f"{filename} not found — skipping '{model_id}'.")

    # Sanity-check each loaded model's expected input width against the
    # pipeline it's supposed to use, so a mismatched/stale model file
    # fails loudly at startup instead of producing silently wrong predictions.
    expected_widths = {"pca": 2, "anova": len(_anova_features or []), "embedded": len(_embedded_features or [])}
    for model_id, model in _models.items():
        pipeline = MODEL_PIPELINE.get(model_id)
        if pipeline is None:
            print(f"NOTE: '{model_id}' has no pipeline mapping yet — add one to MODEL_PIPELINE before using it.")
            continue
        n_in = getattr(model, "n_features_in_", None)
        expected = expected_widths.get(pipeline)
        if n_in is not None and expected and n_in != expected:
            print(
                f"WARNING: '{model_id}' expects {n_in} input features but its "
                f"'{pipeline}' pipeline currently produces {expected}. Double-check "
                f"this model file matches the pipeline it's mapped to."
            )

    print(f"Loaded models: {list(_models.keys())}")


# ---------------------------------------------------------------------
# 3. Request/response schemas
# ---------------------------------------------------------------------
class ModelChoice(str, Enum):
    random_forest = "random_forest"
    svm = "svm"
    mlp = "mlp"
    knn = "knn"
    logistic_regression = "logistic_regression"
    naive_bayes = "naive_bayes"


class StressInput(BaseModel):
    anxiety_level: int = Field(..., ge=0, le=21)
    self_esteem: int = Field(..., ge=0, le=30)
    mental_health_history: int = Field(..., ge=0, le=1)
    depression: int = Field(..., ge=0, le=27)
    headache: int = Field(..., ge=0, le=5)
    blood_pressure: int = Field(..., ge=1, le=3)
    sleep_quality: int = Field(..., ge=0, le=5)
    breathing_problem: int = Field(..., ge=0, le=5)
    noise_level: int = Field(..., ge=0, le=5)
    living_conditions: int = Field(..., ge=0, le=5)
    safety: int = Field(..., ge=0, le=5)
    basic_needs: int = Field(..., ge=0, le=5)
    academic_performance: int = Field(..., ge=0, le=5)
    study_load: int = Field(..., ge=0, le=5)
    teacher_student_relationship: int = Field(..., ge=0, le=5)
    future_career_concerns: int = Field(..., ge=0, le=5)
    social_support: int = Field(..., ge=0, le=3)
    peer_pressure: int = Field(..., ge=0, le=5)
    extracurricular_activities: int = Field(..., ge=0, le=5)
    bullying: int = Field(..., ge=0, le=5)

    class Config:
        json_schema_extra = {
            "example": {
                "anxiety_level": 14, "self_esteem": 20, "mental_health_history": 0,
                "depression": 15, "headache": 3, "blood_pressure": 2, "sleep_quality": 2,
                "breathing_problem": 2, "noise_level": 3, "living_conditions": 3,
                "safety": 3, "basic_needs": 3, "academic_performance": 2, "study_load": 4,
                "teacher_student_relationship": 2, "future_career_concerns": 4,
                "social_support": 1, "peer_pressure": 3, "extracurricular_activities": 2,
                "bullying": 1,
            }
        }


class PredictionResponse(BaseModel):
    model_used: str
    predicted_class: int
    predicted_label: str
    confidence: float | None = None
    class_probabilities: dict | None = None


# ---------------------------------------------------------------------
# 4. Shared preprocessing: raw input -> scaled 20 cols -> engineered 25 cols
# ---------------------------------------------------------------------
def _scale_and_engineer(payload: StressInput) -> pd.DataFrame:
    """Returns a single-row DataFrame with all 20 scaled raw columns plus
    the 5 engineered composite columns (25 columns total, FEATURE_COLUMNS
    order then ENGINEERED_COLUMNS order). All three downstream pipelines
    are derived from this."""
    row = pd.DataFrame([payload.dict()])[FEATURE_COLUMNS]

    scaled_array = _preprocessor.transform(row)
    scaled_df = pd.DataFrame(scaled_array, columns=PREPROCESSOR_OUTPUT_ORDER)
    scaled_df = scaled_df[FEATURE_COLUMNS]

    for score_name in ENGINEERED_COLUMNS:
        group_cols = FEATURE_GROUPS[score_name]
        scaled_df[score_name] = scaled_df[group_cols].mean(axis=1)

    return scaled_df[FEATURE_COLUMNS + ENGINEERED_COLUMNS]


def _build_model_input(model_id: str, full_row: pd.DataFrame) -> np.ndarray:
    """Branches into the PCA / ANOVA / embedded pipeline depending on
    which dataset the given model was trained on."""
    pipeline = MODEL_PIPELINE.get(model_id)

    if pipeline == "pca":
        if _pca is None:
            raise HTTPException(status_code=503, detail="pca_model.pkl not loaded.")
        # PCA was fit on the 20 scaled raw features (FEATURE_COLUMNS order).
        return _pca.transform(full_row[FEATURE_COLUMNS].values)

    if pipeline == "anova":
        if not _anova_features:
            raise HTTPException(status_code=503, detail="anova_top10_features.pkl not loaded.")
        return full_row[_anova_features].values

    if pipeline == "embedded":
        if not _embedded_features:
            raise HTTPException(status_code=503, detail="top10_features_list.pkl not loaded.")
        return full_row[_embedded_features].values

    raise HTTPException(
        status_code=503,
        detail=f"No preprocessing pipeline configured for '{model_id}' yet — "
               f"add it to MODEL_PIPELINE once it's trained.",
    )


# ---------------------------------------------------------------------
# 5. AI chat: lets the student fill in the 20 answers by talking instead
#    of clicking through the form. Supports three providers — Anthropic
#    (Claude), OpenAI (ChatGPT), and Google (Gemini). For each, a key can
#    come from either:
#      - a server-side environment variable (ANTHROPIC_API_KEY /
#        OPENAI_API_KEY / GEMINI_API_KEY) — used for every visitor, or
#      - a key the visitor pastes into the front end's settings panel,
#        sent with each /chat request and used only for that request.
#    Either way, once a request reaches this server the key is used only
#    to call the provider — it is never logged or written to disk.
#    NOTE: a visitor-supplied key still passes through this backend in
#    plain JSON over whatever transport you deploy behind (use HTTPS in
#    production), and this endpoint has no per-user auth — see the rate
#    limiter below for the only abuse protection in place.
# ---------------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Update these to models you actually have access to — provider model
# catalogs change often, so check current docs rather than trusting
# these defaults blindly:
#   Anthropic: https://docs.claude.com
#   OpenAI:    https://platform.openai.com/docs/models
#   Gemini:    https://ai.google.dev/gemini-api/docs/models
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GEMINI_URL_TMPL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

SERVER_PROVIDER_KEYS = {
    "anthropic": ANTHROPIC_API_KEY,
    "openai": OPENAI_API_KEY,
    "gemini": GEMINI_API_KEY,
}

# Very lightweight global rate limit so a stray loop (or a stranger who
# finds your URL) can't rack up API usage unattended. Not per-user auth —
# just a blunt safety valve. Tune to taste.
_CHAT_RATE_LIMIT = 30       # max /chat calls...
_CHAT_RATE_WINDOW = 60      # ...per this many seconds
_chat_call_times = collections.deque()


def _check_chat_rate_limit():
    now = time.time()
    while _chat_call_times and now - _chat_call_times[0] > _CHAT_RATE_WINDOW:
        _chat_call_times.popleft()
    if len(_chat_call_times) >= _CHAT_RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="AI chat is getting a lot of use right now — please wait a moment and try again.",
        )
    _chat_call_times.append(now)


def _build_chat_system_prompt(known_answers: dict) -> str:
    lines = [
        "You are MindCheck, a warm, conversational assistant that chats with a "
        "student to gently understand their current stress levels across five "
        "areas of student life: Mind, Body, Surroundings, Academics, and People.",
        "",
        "Have a natural back-and-forth conversation — ask about one or two "
        "things at a time, follow up on what they actually say, and don't "
        "interrogate them with a checklist. You don't need to ask about every "
        "field explicitly if they've already told you enough to infer it.",
        "",
        "CRITICAL — every single turn MUST include a normal written reply to "
        "the student, in plain text, as if you were simply chatting with "
        "them. This is not optional and the tool call below does NOT count "
        "as your reply — the student never sees the tool call, only your "
        "written words. Never respond with a tool call and no text.",
        "",
        "Separately from that written reply, also call the update_stress_answers "
        "tool on every turn with any fields you can now confidently infer or "
        "re-estimate, using exactly these keys and ranges:",
    ]
    for f in FEATURE_SPEC:
        lines.append(f"  - {f['key']} ({f['chapter']}, {f['min']}-{f['max']}): {f['desc']}")
    lines += [
        "",
        f"Known so far: {json.dumps(known_answers)}",
        "",
        "Keep your written replies short (2-4 sentences), warm, and "
        "non-clinical — this is a peer check-in, not an intake form. Once "
        "you have reasonable coverage across all five areas, say so in your "
        "written reply and set ready_for_result to true in your tool call.",
    ]
    return "\n".join(lines)


def _leading_trimmed(messages: list) -> list:
    """Anthropic (and, for consistency, the other providers too) expects
    the conversation to start on a user turn. The front end's static
    opening greeting is UI decoration, not real conversation history —
    drop any leading assistant message(s) before calling any provider."""
    for i, m in enumerate(messages):
        if m.role == "user":
            return messages[i:]
    return []


def _extract_updates(raw_args: dict) -> tuple[dict, bool]:
    """Shared clamping/validation logic for whatever a provider's tool
    call returned, regardless of which provider produced it."""
    raw_args = dict(raw_args or {})
    ready = bool(raw_args.pop("ready_for_result", False))
    updates = {}
    for key, val in raw_args.items():
        spec = FEATURE_SPEC_BY_KEY.get(key)
        if spec is None or not isinstance(val, (int, float)) or isinstance(val, bool):
            continue
        updates[key] = max(spec["min"], min(spec["max"], int(round(val))))
    return updates, ready


# ---- Anthropic (Claude) ----------------------------------------------

def _anthropic_tool() -> dict:
    properties = {
        f["key"]: {"type": "integer", "minimum": f["min"], "maximum": f["max"], "description": f["desc"]}
        for f in FEATURE_SPEC
    }
    properties["ready_for_result"] = {
        "type": "boolean",
        "description": "Set true once you have reasonable coverage across all five areas to move on to the prediction.",
    }
    return {
        "name": "update_stress_answers",
        "description": (
            "Record or update any of the student's stress-assessment feature values "
            "that can be confidently inferred from the conversation so far. Include "
            "only fields you're reasonably confident about."
        ),
        "input_schema": {"type": "object", "properties": properties},
    }


async def _call_anthropic(api_key: str, system_prompt: str, messages: list) -> tuple[str, dict]:
    body = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 700,
        "system": system_prompt,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "tools": [_anthropic_tool()],
        "tool_choice": {"type": "auto"},
    }
    headers = {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION, "content-type": "application/json"}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(ANTHROPIC_URL, headers=headers, json=body)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Anthropic error ({resp.status_code}): {resp.text[:300]}")

    data = resp.json()
    reply_text, tool_args = "", {}
    for block in data.get("content", []):
        if block.get("type") == "text":
            reply_text += block.get("text", "")
        elif block.get("type") == "tool_use" and block.get("name") == "update_stress_answers":
            tool_args = block.get("input", {}) or {}
    return reply_text, tool_args


# ---- OpenAI (ChatGPT) --------------------------------------------------

def _openai_tool() -> dict:
    properties = {
        f["key"]: {"type": "integer", "minimum": f["min"], "maximum": f["max"], "description": f["desc"]}
        for f in FEATURE_SPEC
    }
    properties["ready_for_result"] = {
        "type": "boolean",
        "description": "Set true once you have reasonable coverage across all five areas to move on to the prediction.",
    }
    return {
        "type": "function",
        "function": {
            "name": "update_stress_answers",
            "description": (
                "Record or update any of the student's stress-assessment feature values "
                "that can be confidently inferred from the conversation so far. Include "
                "only fields you're reasonably confident about."
            ),
            "parameters": {"type": "object", "properties": properties},
        },
    }


async def _call_openai(api_key: str, system_prompt: str, messages: list) -> tuple[str, dict]:
    body = {
        "model": OPENAI_MODEL,
        "max_completion_tokens": 700,
        "messages": [{"role": "system", "content": system_prompt}] + [
            {"role": m.role, "content": m.content} for m in messages
        ],
        "tools": [_openai_tool()],
        "tool_choice": "auto",
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(OPENAI_URL, headers=headers, json=body)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OpenAI error ({resp.status_code}): {resp.text[:300]}")

    data = resp.json()
    message = (data.get("choices") or [{}])[0].get("message", {})
    reply_text = message.get("content") or ""
    tool_args = {}
    for call in message.get("tool_calls") or []:
        fn = call.get("function", {})
        if fn.get("name") == "update_stress_answers":
            try:
                tool_args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                tool_args = {}
    return reply_text, tool_args


# ---- Google (Gemini) ----------------------------------------------------

def _gemini_tool() -> dict:
    # Gemini's function-declaration Schema doesn't support minimum/maximum
    # the way JSON Schema does, so the allowed range only appears in the
    # description text here.
    properties = {f["key"]: {"type": "INTEGER", "description": f["desc"]} for f in FEATURE_SPEC}
    properties["ready_for_result"] = {
        "type": "BOOLEAN",
        "description": "Set true once you have reasonable coverage across all five areas to move on to the prediction.",
    }
    return {
        "name": "update_stress_answers",
        "description": (
            "Record or update any of the student's stress-assessment feature values "
            "that can be confidently inferred from the conversation so far. Include "
            "only fields you're reasonably confident about."
        ),
        "parameters": {"type": "OBJECT", "properties": properties},
    }


async def _call_gemini(api_key: str, system_prompt: str, messages: list) -> tuple[str, dict]:
    # Gemini's function-calling "AUTO" mode means the model picks EITHER a
    # text reply OR a function call — unlike Anthropic/OpenAI it doesn't
    # reliably do both in one response. So this is two calls, not one:
    #   1. A plain call with no tools at all — guarantees a real reply.
    #   2. A call with the tool forced on ("ANY") purely to extract fields,
    #      using the reply from step 1 as context. Its own text (if any)
    #      is discarded; only the function call matters here.
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    url = GEMINI_URL_TMPL.format(model=GEMINI_MODEL)
    contents = [
        {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]}
        for m in messages
    ]

    async with httpx.AsyncClient(timeout=30) as client:
        reply_body = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
        }
        resp1 = await client.post(url, headers=headers, json=reply_body)
        if resp1.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Gemini error ({resp1.status_code}): {resp1.text[:300]}")
        parts1 = (((resp1.json().get("candidates") or [{}])[0]).get("content") or {}).get("parts") or []
        reply_text = "".join(p.get("text", "") for p in parts1)

        tool_args = {}
        try:
            extract_body = {
                "systemInstruction": {"parts": [{
                    "text": system_prompt + "\n\nCall update_stress_answers now with whatever you "
                                             "can infer from the conversation so far (an empty object "
                                             "if nothing new). Do not write any other text."
                }]},
                "contents": contents + [{"role": "model", "parts": [{"text": reply_text}]}],
                "tools": [{"functionDeclarations": [_gemini_tool()]}],
                "toolConfig": {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["update_stress_answers"]}},
            }
            resp2 = await client.post(url, headers=headers, json=extract_body)
            if resp2.status_code == 200:
                parts2 = (((resp2.json().get("candidates") or [{}])[0]).get("content") or {}).get("parts") or []
                for part in parts2:
                    if "functionCall" in part and part["functionCall"].get("name") == "update_stress_answers":
                        tool_args = part["functionCall"].get("args", {}) or {}
        except httpx.RequestError:
            pass  # extraction is best-effort — a failed/slow extraction shouldn't block the reply

    return reply_text, tool_args


PROVIDER_CALLERS = {"anthropic": _call_anthropic, "openai": _call_openai, "gemini": _call_gemini}
PROVIDER_LABELS = {"anthropic": "Claude (Anthropic)", "openai": "ChatGPT (OpenAI)", "gemini": "Gemini (Google)"}


class ChatMessage(BaseModel):
    role: str   # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    known_answers: dict = {}
    provider: str = "anthropic"        # "anthropic" | "openai" | "gemini"
    api_key: str | None = None         # bring-your-own-key from the settings panel; overrides the server's key


class ChatResponse(BaseModel):
    reply: str
    updates: dict
    ready: bool


@app.get("/chat/status")
def chat_status():
    """Lets the front end know which providers have a server-side key
    configured, without exposing the keys themselves. A provider showing
    false here can still be used if the visitor supplies their own key
    via the settings panel."""
    return {p: bool(k) for p, k in SERVER_PROVIDER_KEYS.items()}


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    provider = payload.provider if payload.provider in PROVIDER_CALLERS else "anthropic"
    api_key = payload.api_key or SERVER_PROVIDER_KEYS.get(provider)
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                f"AI chat isn't configured for {PROVIDER_LABELS[provider]} — set the server's "
                f"API key, or paste your own key for this provider in the settings panel."
            ),
        )
    _check_chat_rate_limit()

    messages = _leading_trimmed(payload.messages)
    system_prompt = _build_chat_system_prompt(payload.known_answers)

    try:
        reply_text, raw_tool_args = await PROVIDER_CALLERS[provider](api_key, system_prompt, messages)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Couldn't reach {PROVIDER_LABELS[provider]}: {exc}")

    updates, ready = _extract_updates(raw_tool_args)

    if not reply_text.strip():
        reply_text = "Got it — tell me a bit more whenever you're ready."

    return ChatResponse(reply=reply_text.strip(), updates=updates, ready=ready)


# ---------------------------------------------------------------------
# 6. Prediction endpoint
# ---------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
def predict(payload: StressInput, model: ModelChoice = ModelChoice.random_forest):
    if _preprocessor is None or model.value not in _models:
        raise HTTPException(
            status_code=503,
            detail="Model or preprocessor not loaded. Check the models/ folder.",
        )

    full_row = _scale_and_engineer(payload)
    X = _build_model_input(model.value, full_row)

    chosen_model = _models[model.value]

    pred_class = int(chosen_model.predict(X)[0])
    confidence = None
    probabilities = None
    if hasattr(chosen_model, "predict_proba"):
        proba = chosen_model.predict_proba(X)[0]
        probabilities = {STRESS_LABELS[i]: round(float(p), 4) for i, p in enumerate(proba)}
        confidence = round(float(np.max(proba)), 4)

    return PredictionResponse(
        model_used=model.value,
        predicted_class=pred_class,
        predicted_label=STRESS_LABELS[pred_class],
        confidence=confidence,
        class_probabilities=probabilities,
    )


@app.get("/metrics")
def get_metrics():
    """Returns evaluation results for each model, used by the front end's
    performance comparison chart. Only includes models that are actually
    loaded and available to predict with."""
    return {
        model_id: data
        for model_id, data in MODEL_METRICS.items()
        if model_id in _models
    }


@app.get("/")
def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": "Student Stress Level Prediction API is running.",
        "docs": "/docs",
        "available_models": list(_models.keys()) if _models else "none loaded yet",
    }


@app.get("/api")
def api_status():
    return {
        "message": "Student Stress Level Prediction API is running.",
        "docs": "/docs",
        "available_models": list(_models.keys()) if _models else "none loaded yet",
    }