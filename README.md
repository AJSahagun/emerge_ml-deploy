### 1. Directory Structure

```
emerge_deploy/
├── app/
│   ├── __init__.py         # Empty file to make 'app' a Python package
│   ├── main.py             # Flask application code (prediction server)
│   └── preprocessing.py    # Your text cleaning/preprocessing functions
├── model/                  # Directory containing your saved model artefacts
│   ├── config.json         # Saved by model.save_pretrained()
│   ├── pytorch_model.bin   # Saved by model.save_pretrained() (Base model weights)
│   ├── special_tokens_map.json # Saved by tokenizer.save_pretrained()
│   ├── tokenizer_config.json # Saved by tokenizer.save_pretrained()
│   ├── tokenizer.json      # Saved by tokenizer.save_pretrained()
│   └── classifier_head.pth # Your saved classification head state_dict
├── stopwords-tl.json     # Your Tagalog stopwords file (if needed for preprocessing)
├── requirements.txt        # Python dependencies
└── Dockerfile              # Container build instructions
```

**Explanation:**

* **`emerge_deploy/`**: The root directory for your deployment files.
* **`app/`**: Holds the Python code for your prediction server.
    * **`main.py`**: The main Flask application file.
    * **`preprocessing.py`**: Contains the necessary cleaning and preprocessing functions (like `clean_text` from Notebook 1).
* **`model/`**: This directory should contain the output from saving your fine-tuned XLM-R model and tokenizer using `model.bert.save_pretrained(MODEL_SAVE_GCS_DIR)` and `tokenizer.save_pretrained(MODEL_SAVE_GCS_DIR)`, plus the `classifier_head.pth` you saved separately. **You need to copy these files from your GCS bucket (`MODEL_SAVE_GCS_DIR`) into this local `model/` directory before building the container.**
* **`stopwords-tl.json`**: Place the Tagalog stopwords file here if your preprocessing function needs it.
* **`requirements.txt`**: Lists Python libraries needed inside the container.
* **`Dockerfile`**: Instructions for Docker to build your container image.


### 2. Prediction Server (`app/main.py`)

This file contains the Flask application that Vertex AI will interact with.

**Key points in `main.py`:**

* **Model Loading:** Loads the tokenizer, base model, and the *state dict* of the classifier head into the defined `EmergeClassifier` structure. This happens once when the server starts.
* **Error Handling:** Includes basic `try...except` blocks for model loading and prediction. Logs errors to `stderr`.
* **Health Endpoint:** A simple `/health` route returns a 200 status if the model loaded okay, 500 otherwise. Vertex AI uses this.
* **Predict Endpoint:**
    * Expects POST requests with JSON like `{"instances": [{"report_text": "..."}, ...]}`.
    * Iterates through each instance.
    * Calls `clean_text` (and potentially other preprocessing).
    * Tokenizes using the loaded Transformer tokenizer.
    * Runs the input through the loaded `model_full`.
    * Applies `softmax` to get probabilities.
    * Finds the class with the highest probability (`predicted_class`) and its probability (`confidence`).
    * Returns results in the required JSON format `{"predictions": [...]}`.

### 3. Build and Push the Container Image

1.  **Set up Artifact Registry:**
    * If you haven't already, create an Artifact Registry Docker repository in your GCP project (e.g., called `emerge-repo` in region `us-central1`).
    ```bash
    gcloud artifacts repositories create emerge-repo \
        --repository-format=docker \
        --location=us-central1 \
        --description="Docker repository for EMERGE project"
    ```
    * Configure Docker to authenticate with Artifact Registry:
    ```bash
    gcloud auth configure-docker us-central1-docker.pkg.dev # Replace us-central1 if needed
    ```
2.  **Build and Push using Cloud Build:**
    * Navigate to your `emerge_deploy` directory in your terminal (or Cloud Shell).
    * Define your image URI:
    ```bash
    export PROJECT_ID=$(gcloud config get-value project)
    export REPO_NAME=emerge-repo
    export IMAGE_NAME=emerge-classifier
    export IMAGE_TAG=v1
    export REGION=us-central1 # Use the region where you created the repo
    export IMAGE_URI=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}
    ```
    * Submit the build to Cloud Build:
    ```bash
    gcloud builds submit --tag $IMAGE_URI .
    ```
    (The `.` indicates the current directory `emerge_deploy/` is the build context).

    Cloud Build will use the `Dockerfile` in your directory, build the image, and push it to your Artifact Registry repository.

### 4. Upload Model to Vertex AI Model Registry using Custom Container

Now you can register this container image as your model server in Vertex AI.

1.  **Using the GCP Console:**
    * Go to Vertex AI -> Models -> Import.
    * Select "Import new model".
    * Give it a name (e.g., `emerge-xlmr-custom`).
    * Region: Select the appropriate region.
    * **Container settings:** Select "Custom container".
    * **Container image:** Enter the `IMAGE_URI` you pushed to Artifact Registry (e.g., `us-central1-docker.pkg.dev/your-project-id/emerge-repo/emerge-classifier:v1`).
    * **Model framework:** Select "PyTorch" (or leave blank).
    * **Health route:** Should automatically pick up `/health` from the ENV var in the Dockerfile. If not, enter `/health`.
    * **Predict route:** Should automatically pick up `/predict`. If not, enter `/predict`.
    * **Ports:** Enter `8080` for the container port.
    * **Model Artefacts:** Since you copied the model *into* the image, you can leave the "Model directory" field blank here (or point it to the GCS location just for reference, but the container won't use `AIP_STORAGE_URI` to load the model in this setup).
    * Click "Import".

2.  **Using `gcloud` (Command Line):**
    ```bash
    gcloud ai models upload \
      --region=${REGION} \
      --display-name="emerge-xlmr-custom" \
      --container-image-uri=${IMAGE_URI} \
      --container-predict-route="/predict" \
      --container-health-route="/health" \
      --container-ports="8080" \
      --description="Custom container for EMERGE legitimacy classification"
      # Add --artifact-uri=gs://your-bucket/path/to/model/ if you want to keep artefacts separate
      # and load from AIP_STORAGE_URI in main.py instead of copying to image.
    ```

### 5.  Create an Endpoint
      * Go to Vertex AI -\> Endpoints.
      * Click "Create Endpoint".
      * Give it a name (e.g., `emerge-classifier-endpoint`).
      * Configure access settings if needed.

### 6.  Deploy the Model to the Endpoint
      * Select the Endpoint you just created (or create it during model deployment).
      * Click "Deploy model".
      * Choose the model you uploaded from the Model Registry.
      * Configure the machine type for the prediction nodes (start with something like `n1-standard-2` and monitor performance/cost).
      * Set traffic split (usually 100% for the first deployment).
      * Deploy. This can take several minutes as Vertex AI provisions resources.

### 7.  Making HTTP Requests
      * Once deployed, Vertex AI provides a REST endpoint URL.
      * Your system can send `POST` requests to this URL. The request body should be JSON, formatted as expected by your prediction container (usually containing the report text). A common format is:
        ```json
        {
          "instances": [
            { "report_text": "May sunog po dito malapit sa amin..." },
            { "report_text": "Fire alarm activated, smoke detected on 3rd floor." }
          ]
        }
        ```
        *Note: The exact format depends on the container/handler script.*
      * The response from the endpoint will also be JSON, containing the predictions (class and confidence score) for each instance sent in the request, e.g.:
        ```json
        {
          "predictions": [
            { "predicted_class": 1, "confidence": 0.95 },
            { "predicted_class": 1, "confidence": 0.88 }
          ]
        }
        ```
### 8.  Integration 
Your external system will need code (e.g., using Python's `requests` library, or similar libraries in other languages) to:
      * Authenticate to GCP (e.g., using a service account key or Application Default Credentials if running on GCP).
      * Format the report text into the required JSON payload.
      * Send the `POST` request to the Vertex AI Endpoint URL.
      * Parse the JSON response to get the legitimacy class and confidence score.

**Key Considerations for Deployment:**

  * **Custom Container Complexity:** Building a custom container requires knowledge of Docker and setting up a simple web server (Flask/FastAPI). However, it gives you full control over preprocessing and prediction logic. Search the Google Cloud documentation for "Vertex AI Custom Prediction Routines" or "Custom Containers for Prediction".
  * **Preprocessing:** Ensure the *exact* same preprocessing steps used during training are applied during inference within the endpoint container. This is critical for model performance.
  * **Monitoring & Scaling:** Vertex AI provides monitoring for endpoints (latency, error rates, traffic). You can configure autoscaling based on CPU usage or traffic to handle varying loads.