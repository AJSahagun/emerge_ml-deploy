import os
import sys # For error logging
import flask
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# Import preprocessing functions from the local module
from preprocessing import clean_text  # Import specific functions needed

# --- Model Loading ---

# Get model directory from environment variable set in Dockerfile
model_dir = os.environ.get("MODEL_DIR", "/app/model")
classifier_head_path = os.path.join(model_dir, 'classifier_head.pth')

print(f"Loading model artefacts from: {model_dir}", file=sys.stderr) # Log to stderr
print(f"Classifier head path: {classifier_head_path}", file=sys.stderr)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", file=sys.stderr)

tokenizer = None
model_base = None
classifier_head = None
model_full = None

try:
    # Define the model architecture again (must match the trained model)
    class EmergeClassifier(nn.Module):
        def __init__(self, model_base_path, n_classes=2):
            super(EmergeClassifier, self).__init__()
            # Load the base model (e.g., XLM-R)
            self.bert = AutoModel.from_pretrained(model_base_path)
            # Define the classification head structure
            self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

        def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0] # Use [CLS] token
            return self.classifier(pooled_output)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # Initialize the full model structure before loading the head's state_dict
    model_full = EmergeClassifier(model_dir, n_classes=2) # Pass model_dir to load base model config/weights

    # Load the state dict for the classification head
    if os.path.exists(classifier_head_path):
        model_full.classifier.load_state_dict(torch.load(classifier_head_path, map_location=device))
        print("Classifier head state dict loaded successfully.", file=sys.stderr)
    else:
         print(f"Error: Classifier head file not found at {classifier_head_path}", file=sys.stderr)
         # Handle error appropriately - maybe raise an exception or exit

    model_full = model_full.to(device)
    model_full.eval() # Set to evaluation mode
    print("Model loaded successfully.", file=sys.stderr)

except Exception as e:
    print(f"Error loading model: {e}", file=sys.stderr)
    # You might want to exit or prevent the Flask app from starting if model loading fails
    tokenizer = None
    model_full = None

# --- Flask Application ---
app = flask.Flask(__name__)

@app.route(os.environ.get("AIP_HEALTH_ROUTE", "/health"), methods=["GET"])
def health():
    """Health check endpoint required by Vertex AI."""
    # Check if model loaded successfully
    status = 200 if model_full is not None and tokenizer is not None else 500
    return {"status": "healthy" if status==200 else "unhealthy"}, status

@app.route(os.environ.get("AIP_PREDICT_ROUTE", "/predict"), methods=["POST"])
def predict():
    """Prediction endpoint required by Vertex AI."""
    if not model_full or not tokenizer:
         return flask.jsonify({"error": "Model not loaded"}), 500

    try:
        request_json = flask.request.get_json()
        instances = request_json["instances"] # List of inputs
        predictions = []

        for instance in instances:
             # Assuming each instance is a dict, e.g., {"report_text": "..."}
             # Adjust key ('report_text') if your request format is different
             if isinstance(instance, dict) and "report_text" in instance:
                 text = instance["report_text"]
             elif isinstance(instance, str): # Allow plain strings as instances too
                 text = instance
             else:
                 predictions.append({"error": "Invalid instance format", "predicted_class": 0, "confidence": 0.0})
                 continue

             # 1. Preprocess the text (using functions from preprocessing.py)
             processed_text = clean_text(text)
             # Add other preprocessing steps here if needed (e.g., lang detect for routing)

             if not processed_text:
                 predictions.append({"warning": "Empty text after preprocessing", "predicted_class": 0, "confidence": 0.0})
                 continue

             # 2. Tokenize for the specific model
             encoding = tokenizer.encode_plus(
                 processed_text,
                 add_special_tokens=True,
                 max_length=512, # Use the model's max length
                 return_token_type_ids=False,
                 padding='max_length',
                 truncation=True,
                 return_attention_mask=True,
                 return_tensors='pt',
             )

             # 3. Predict using the loaded model
             input_ids = encoding['input_ids'].to(device)
             attention_mask = encoding['attention_mask'].to(device)

             with torch.no_grad():
                 outputs = model_full(input_ids=input_ids, attention_mask=attention_mask)
                 # Apply softmax to get probabilities
                 probabilities = torch.softmax(outputs, dim=1)[0] # Get probabilities for the single input
                 confidence, prediction_class = torch.max(probabilities, dim=0)

             # 4. Format prediction result
             predictions.append({
                 "predicted_class": int(prediction_class.cpu().item()),
                 "confidence": float(confidence.cpu().item())
             })

        return flask.jsonify({"predictions": predictions})

    except Exception as e:
        print(f"Prediction error: {e}", file=sys.stderr)
        return flask.jsonify({"error": str(e)}), 500

# Note: The following is for local testing, Gunicorn runs the app in production
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=int(os.environ.get("AIP_HTTP_PORT", 8080)))