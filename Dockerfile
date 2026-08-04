FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so this layer caches unless requirements change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY api.py .

# The sample data is needed to train the model at build time.
# In a real deployment this would be a pre-trained artifact pulled from a
# model registry rather than trained in the image.
COPY data/ ./data/

# Train and persist the model as a build step so the container starts ready
RUN python scripts/train_and_save.py

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
