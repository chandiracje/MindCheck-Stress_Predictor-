# Match the Python version you've been developing against locally (3.11),
# since scikit-learn==1.6.1 and your saved model files were built there.
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer separately
# from your code — rebuilds are much faster when you only change app.py
# or the front end.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy everything else: app.py, static/, models/
COPY . .

# Render (and most platforms) inject a PORT env var at runtime and expect
# the app to bind to it — 8000 is just the local fallback.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
