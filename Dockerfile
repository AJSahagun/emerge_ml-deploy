# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables required by Vertex AI
# Note: AIP_STORAGE_URI will be set by Vertex AI pointing to your model files IF
# you upload the model artefacts separately via gcloud ai models upload --artifact-uri=...
# If you copy the model INTO the image (as we do below), AIP_STORAGE_URI might not be used by your app code.
ENV AIP_HEALTH_ROUTE=/health
ENV AIP_PREDICT_ROUTE=/predict
ENV AIP_HTTP_PORT=8080

# Set the working directory in the container
WORKDIR /app

# --- Optional: Install NLTK data if needed by preprocessing.py ---
# RUN pip install nltk && \
#     python -m nltk.downloader -d /usr/local/share/nltk_data stopwords wordnet omw-1.4 punkt # Add 'punkt' if needed for sentence tokenization
# ENV NLTK_DATA=/usr/local/share/nltk_data
# ------------------------------------------------------------------

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir reduces image size, --prefer-binary can speed up builds
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# --- Optional: Install tl_calamancy_md if needed ---
# RUN pip install --no-cache-dir https://huggingface.co/ljvmiranda921/tl_calamancy_md/resolve/main/tl_calamancy_md-any-py3-none-any.whl
# --------------------------------------------------

# Copy the application code into the container at /app/app
COPY ./app /app/app

# Copy the model artefacts into the container at /app/model
# This makes the model part of the image itself.
COPY ./model /app/model

# Copy Tagalog stopwords if needed by preprocessing
COPY ./stopwords-tl.json /app/stopwords-tl.json

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable for model directory within the container
ENV MODEL_DIR=/app/model

# Run the Flask app using gunicorn when the container launches
# workers=1 is often recommended for ML models to avoid memory issues
# threads=8 helps handle concurrent requests
# timeout=0 disables worker timeout (important for long model loading or predictions)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "app.main:app"]