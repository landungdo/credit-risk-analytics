FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so this layer caches unless requirements change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and tests (conftest holds the synthetic generator)
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY api.py .

# The real Lending Club data is licensed and gitignored, so it is NOT in the
# build context. To keep the image self-contained and buildable from a clean
# clone, generate a small synthetic demo dataset with the same schema, then
# train on it. For real results, mount or COPY a real data/sample.csv instead.
RUN python scripts/make_demo_data.py

# Train and persist the model + policy as a build step so the container starts ready
RUN python scripts/train_and_save.py

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
