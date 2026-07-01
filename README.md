# AI-enabled Log Manager

## Demo

[Watch on YouTube](https://youtu.be/saQkZxw7miw)

---

## Tech Stack

| Tool | Role |
|------|------|
| **FastAPI** | REST API framework — handles all HTTP endpoints, request validation, and file responses |
| **scikit-learn LinearSVC** | The classifier that predicts root cause categories from log text |
| **scikit-learn TF-IDF** | Converts log message text into numerical features the model can learn from — scores words by how useful they are for distinguishing categories |
| **joblib** | Serialises and deserialises the trained model, vectorizer, and encoder to/from `.pkl` files on disk |
| **pandas** | Reads CSV inputs, structures prediction results, and writes output files |
| **Python `re`** | Regex-based log message cleaning — masks IPs, ports, numbers, and timestamps with generic placeholders before training and inference |
| **Pydantic** | Validates and parses incoming JSON request bodies for the `/predict/single` endpoint |
| **uvicorn** | ASGI server that runs the FastAPI application |
| **numpy** | Powers confidence score calculations (sigmoid, softmax) and metric aggregations in evaluation |

---

# Set up

* Clone or navigate to your project directory

    `cd log_manager`

* Create a Python virtual environment

    `python -m venv venv`

* **Activate the virtual environment:**

* On macOS/Linux: `source venv/bin/activate`
* On Windows: `.\venv\Scripts\activate`

* Install all required dependencies:
    `pip install -r requirements.txt`

* Start the FastAPI development server: 
`uvicorn main:app --reload`

* Navigate to: `http://localhost:8000/docs`

# Folder Structure

```
log_manager/
│
├── data/
│   ├── log_dataset.csv
│   └── root_cause_labels.csv
│
├── src/
│   ├── config.py
│   ├── evaluate.py
│   ├── preprocess.py
│   ├── summarize.py
│   ├── train.py
│   └── inference.py
│
├── artifacts/          # Created dynamically during training
│   ├── model.pkl
│   ├── vectorizer.pkl
│   ├── encoder.pkl
│   ├── metrics.json
│   ├── classification_report.txt
│   └── predictions/    # Created dynamically on first prediction
│       ├── single_20240528_210400.csv
│       ├── file_ingest_20240528_210500.csv
│       
│
├── main.py            # FastAPI Application
├── requirements.txt
└── README.md
```

## API Reference

> Base URL: `http://localhost:8000`  
> Interactive docs available at `http://localhost:8000/docs`

---

### `GET /evaluations`
**Tag:** Pipeline Administration

Returns the latest training evaluation metrics saved to `artifacts/metrics.json`.

**Request:** No body required.

**Sample Response:**
```json
{
  "accuracy": 0.9583,
  "precision_macro": 0.9601,
  "recall_macro": 0.9583,
  "f1_macro": 0.9580,
  "precision_weighted": 0.9604,
  "recall_weighted": 0.9583,
  "f1_weighted": 0.9581,
  "per_class": {
    "RC-01": { "precision": 1.0, "recall": 1.0, "f1": 1.0, "support": 3 },
    "RC-02": { "precision": 0.9, "recall": 1.0, "f1": 0.947, "support": 4 }
  },
  "confusion_matrix": [[3, 0], [0, 4]],
  "confusion_matrix_labels": ["RC-01", "RC-02"],
  "n_samples": 24
}
```

**Error Responses:**

| Status | Detail |
|--------|--------|
| `404`  | `No evaluation metrics found. Run /preprocess-and-train.` |

---

### `POST /preprocess-and-train`
**Tag:** Pipeline Administration

Triggers data preparation and model training against the local dataset files in `data/`. Reloads inference artifacts on completion.

**Request:** No body required.

**Sample Response:**
```json
{
  "status": "Success",
  "metrics": {
    "accuracy": 0.96,
    "classification_report": "..."
  }
}
```

**Error Responses:**

| Status | Detail |
|--------|--------|
| `404`  | `Training source csv missing from data/ directory.` |
| `500`  | Internal training pipeline error message. |

---

### `POST /predict/single`
**Tag:** Inference Engine

Classifies a single log record posted as a JSON body. Returns a downloadable CSV file and saves it to `artifacts/predictions/`.

**Request Body:**
```json
{
  "log_id": "LOG-001",
  "timestamp": "2024-01-15T08:32:11Z",
  "service": "auth-service",
  "log_message": "[ERROR] 192.168.1.45 - Failed login attempt on port 8080 after 3 retries"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `log_id` | `string` | Unique identifier for the log entry |
| `timestamp` | `string` | ISO 8601 timestamp of the log event |
| `service` | `string` | Name of the originating microservice |
| `log_message` | `string` | Raw log message text |

**Response:** A downloadable `.csv` file named `single_<timestamp>.csv`

**Sample CSV output:**
```csv
log_id,root_cause_label,issue,predicted_root_cause_id,predicted_root_cause_label,affected_service,severity,recommended_action,confidence
LOG-001,RC-01,Failed login attempt on port 8080 after 3 retries,RC-01,Authentication Failure,auth-service,High,"Rotate credentials, refresh token, re-authenticate service account",0.9812
```

**Error Responses:**

| Status | Detail |
|--------|--------|
| `400`  | `Model is uninitialized. Run /preprocess-and-train.` |

---

### `POST /predict/file-ingest`
**Tag:** Inference Engine

Accepts a standard CSV file upload and returns predictions for all rows as a downloadable CSV. Saves the result to `artifacts/predictions/`.

**Request:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | `file` | A `.csv` file with at minimum `log_id`, `service`, and `log_message` columns |

**Minimum Required CSV Columns:**
```
log_id, service, log_message
```

**Sample CSV (`logs.csv`):**
```csv
log_id,timestamp,service,log_message
LOG-001,2024-01-15T08:32:11Z,auth-service,"[ERROR] Failed login attempt from 192.168.1.45"
LOG-002,2024-01-15T08:33:05Z,payment-service,"[ERROR] Connection timed out after 30s on port 5432"
```

**Response:** A downloadable `.csv` file named `file_ingest_<timestamp>.csv`

**Sample CSV output:**
```csv
log_id,root_cause_label,issue,predicted_root_cause_id,predicted_root_cause_label,affected_service,severity,recommended_action,confidence
LOG-001,RC-01,Failed login attempt from 192.168.1.45,RC-01,Authentication Failure,auth-service,High,"Rotate credentials, refresh token, re-authenticate service account",0.9812
LOG-002,RC-02,Connection timed out after 30s on port 5432,RC-02,Database Connection Timeout,payment-service,Critical,"Check DB health, increase pool size, investigate network latency",0.9654
```

**Error Responses:**

| Status | Detail |
|--------|--------|
| `400`  | `Model is uninitialized. Run /preprocess-and-train.` |
| `400`  | `Invalid extension format. API accepts raw standard csv configurations.` |
| `400`  | `Missing structural features. Required: ['service', 'log_message']` |

---

---

## Data Preprocessing Steps

All preprocessing is handled in `src/preprocess.py` and runs identically during training and inference so the model always sees data in the same format.

**1. Strip punctuation and brackets**
Characters like `[`, `]`, `(`, `)`, `:`, `,`, and `-` are replaced with spaces. These are structural log formatting characters that carry no meaning for classification.

**2. Mask variable values with placeholders**
Specific values that change between log entries are replaced with fixed tokens so the model learns the error pattern, not the specific numbers:

| What gets replaced | Placeholder |
|--------------------|-------------|
| IP addresses (e.g. `192.168.1.45`) | `<IP_ADDR>` |
| ISO timestamps (e.g. `2024-01-15T08:32:11Z`) | `<TIMESTAMP>` |
| Transaction/user/record IDs (e.g. `txn_abc123`) | `<ID>` |
| Numeric metrics with units (e.g. `30ms`, `512mb`) | `<METRIC>` |
| Ratios (e.g. `3/5`) | `<RATIO>` |
| Bare numbers (e.g. `8080`, `3`) | `<NUM>` |

**3. Lowercase**
The entire message is lowercased so `ERROR`, `Error`, and `error` are treated as the same word.

**4. Normalise whitespace**
Consecutive spaces left behind by substitutions are collapsed into single spaces.

**5. Combine service name and cleaned message**
The microservice name (e.g. `auth-service`) is prepended to the cleaned log message to form a single text feature. This lets the model use service context alongside message content.

**6. TF-IDF vectorization**
The combined text is scored using TF-IDF with unigrams and bigrams (`ngram_range=(1,2)`) capped at 1500 features. Words and two-word phrases that best distinguish one root cause category from another get the highest scores.

**7. Synthetic keyword augmentation (training only)**
One extra training row per root cause category is built from the `example_keywords` column in `root_cause_labels.csv`. These rows are added to the training set only — never to the test set — so the model learns the vocabulary for each category even if it appears rarely in the real data.

---

## What I use for training (file, fields, models) and training steps

### Source Files
| File | Role |
|------|------|
| `data/log_dataset.csv` | 120 real labelled log entries used to train the model |
| `data/root_cause_labels.csv` | Reference sheet defining all 8 root cause categories and their keywords |

### Fields Used from `log_dataset.csv`
| Field | Used for |
|-------|----------|
| `service` | Tells the model which microservice the log came from |
| `log_message` | The raw log text the model learns to classify |
| `root_cause_label` | The correct answer (`RC-01` to `RC-08`) the model is trained to predict |

> `severity`, `timestamp`, and `log_id` are present in the file but ignored during training. `severity` maps directly from the root cause label so using it would be cheating — the model would just memorise it rather than learning from the message text.

### Fields Used from `root_cause_labels.csv`
| Field | Used for |
|-------|----------|
| `id` | Ensures the model knows all 8 categories exist, even if some appear rarely in the dataset |
| `example_keywords` | Turned into extra synthetic training examples so the model learns the vocabulary for each category |

### Model
**LinearSVC** (Linear Support Vector Machine)

Log messages follow rigid, repeated patterns — the same error phrases appear consistently across services. A linear model is the right fit here because:
- The signal is in specific words and phrases (`timeout`, `unauthorized`, `OOM`), not in complex relationships between them
- LinearSVC trains and predicts fast — under 15ms per record — which matters for a real-time classification API
- It generalises well on small labelled datasets (120 rows) without overfitting the way a deep model would
- It is fully explainable — the highest-weighted TF-IDF terms for each class directly show what the model learned

A neural approach (e.g. fine-tuned BERT) would likely score marginally higher but would add significant latency, memory overhead, and retraining complexity for a problem that is already linearly separable.

### Training Steps
1. Load `log_dataset.csv` and `root_cause_labels.csv`
2. Split the real log data 80/20 into a training set and a held-out test set
3. Expand `example_keywords` from `root_cause_labels.csv` into one synthetic training row per category and add them to the training set only
4. Clean each log message — replace IPs, numbers, timestamps, and IDs with generic placeholders so the model focuses on error patterns, not specific values
5. Combine `service` name and cleaned message into a single text feature per row
6. Score every word and two-word phrase by how useful it is for distinguishing categories (TF-IDF, top 1500 features)
7. Train the LinearSVC classifier on the scored features
8. Evaluate on the clean test set (no synthetic rows) and record accuracy, per-category scores, and a confusion matrix
9. Save the trained model, word scorer, and label mapper to `artifacts/`

---

## How I save the model

After training, three files are written to the `artifacts/` folder:

| File | What it contains |
|------|------------------|
| `model.pkl` | The trained LinearSVC classifier |
| `vectorizer.pkl` | The word scorer built from the training data — needed to prepare new logs in the same way |
| `encoder.pkl` | The label mapper that converts between `RC-01`…`RC-08` codes and the internal numbers the model uses |
| `metrics.json` | Accuracy and per-category scores from the test set evaluation |
| `classification_report.txt` | A human-readable table of the same evaluation results |

All files are saved using `joblib`, which is the standard way to persist scikit-learn objects to disk. On the next server start, the inference engine loads all three `.pkl` files back into memory automatically.

---

## How I run inference

When a log record arrives at any of the `/predict` endpoints:

1. The `service` and `log_message` fields are extracted — all other fields are ignored
2. The log message is cleaned the same way it was during training (same placeholder replacements)
3. `service` and cleaned message are combined into a single text feature
4. The saved `vectorizer` scores the words in that text using the vocabulary it learned during training
5. The saved `model` predicts which root cause category the scored features belong to
6. A confidence score (0–1) is calculated from how decisively the model made its choice
7. The predicted category code (e.g. `RC-03`) is looked up in `root_cause_labels.csv` to retrieve the human-readable label, severity level, and recommended action
8. A structured response is returned with the prediction and the full breakdown

## How I save predictions

Every call to any `/predict` endpoint automatically saves the results to a CSV file in `artifacts/predictions/`. The file is also returned directly as a download so you don't need to go looking for it.

### File naming

Each file is named after the endpoint that produced it plus the date and time it was created, so you can always tell when a prediction was run and which method was used:

| Endpoint | Example filename |
|----------|------------------|
| `/predict/single` | `prediction_single_20240528_210400.csv` |
| `/predict/file-ingest` | `prediction_batch_20240528_210500.csv` |

### CSV columns

All prediction files share the same flat structure — the nested analysis fields are unpacked into individual columns so the file is easy to open in Excel or any spreadsheet tool:

| Column | Description |
|--------|-------------|
| `log_id` | The ID of the log record |
| `root_cause_label` | The predicted root cause code (e.g. `RC-03`) |
| `issue` | Plain-English description of what went wrong |
| `predicted_root_cause_id` | Same as `root_cause_label` |
| `predicted_root_cause_label` | Human-readable category name (e.g. `Third-Party API Failure`) |
| `affected_service` | The microservice the log came from |
| `severity` | How serious the issue is (`Low`, `Medium`, `High`, `Critical`) |
| `recommended_action` | What the on-call engineer should do |
| `confidence` | How sure the model is (0 to 1) |

> Old prediction files are never deleted or overwritten — each run produces a new timestamped file, so you have a full history of every prediction batch.

## Observed Tradeoffs

| Decision | What was gained | What was given up |
|----------|----------------|-------------------|
| LinearSVC over a neural model | Fast inference, small footprint, explainable weights | Marginal accuracy ceiling on ambiguous or novel log patterns |
| TF-IDF over raw embeddings | No external dependencies, deterministic, fast to fit | Cannot capture semantic similarity — `timed out` and `connection refused` are treated as unrelated |
| Synthetic keyword augmentation | All 8 classes represented even if rare in real data | Synthetic rows are cleaner than real logs — model may be slightly overconfident on rare classes |
| Split before augmentation | Test set reflects real-world generalisation only | Training set is smaller before augmentation is applied |
| `severity` excluded as a feature | Prevents label leakage — model learns from message text | One genuinely useful signal is withheld (though it is derivable from the predicted class anyway) |
| Flat CSV predictions | Easy to open in Excel, no tooling required | Nested structure (e.g. per-class confidence breakdown) is lost |

---

## Limitations

- **Small dataset** — 120 labelled rows is enough to demonstrate the pipeline but not enough to trust in production. Edge cases and rare phrasings will be misclassified.
- **Closed label set** — the model can only predict one of the 8 trained root cause categories. A log that doesn't fit any of them will still be forced into the closest match with no indication it is out-of-distribution.
- **No confidence threshold** — there is no rejection mechanism. A low-confidence prediction (e.g. 0.51) is returned the same way as a high-confidence one (0.98).
- **Static vocabulary** — the TF-IDF vectorizer is frozen at training time. New error patterns introduced by a service update won't be recognised until the model is retrained.
- **Single-language logs only** — the preprocessing and vocabulary assume English log messages. Mixed-language or non-standard log formats will degrade accuracy.
- **No concept drift detection** — there is nothing in the current pipeline that alerts when incoming log distributions shift away from what the model was trained on.

---

## How I Evaluate

Evaluation runs automatically at the end of every training run and is never performed on the synthetic keyword rows — only on the 20% of real logs held back from training.

### What is measured

| Metric | What it tells you |
|--------|-------------------|
| `accuracy` | The percentage of log records the model classified correctly overall |
| `precision` (per class) | Of all the times the model predicted a category, how often it was actually right |
| `recall` (per class) | Of all the real examples of a category, how many the model actually caught |
| `f1` (per class) | A single score that balances precision and recall for each category |
| `precision_macro` / `recall_macro` / `f1_macro` | The average of the per-class scores, treating every category equally regardless of how many examples it has |
| `precision_weighted` / `recall_weighted` / `f1_weighted` | The same averages but weighted by how many real examples each category has in the test set |
| `confusion_matrix` | A grid showing which categories the model gets right and which ones it mixes up with each other |
| `n_samples` | How many log records were in the test set |

### Where results are saved

| File | Contents |
|------|----------|
| `artifacts/metrics.json` | All scores in machine-readable JSON, returned by `GET /evaluations` |
| `artifacts/classification_report.txt` | A plain-text table of per-category precision, recall, F1, and support |
| `artifacts/evaluations.csv` | Same scores in flat CSV format, returned as a download by `GET /evaluations` |

### Evaluation results

Results below are from the held-out 20% test set (24 real log records, no synthetic rows).

| Metric | Score |
|--------|-------|
| Accuracy | 0.9583 |
| Precision (macro) | 0.9601 |
| Recall (macro) | 0.9583 |
| F1 (macro) | 0.9580 |
| Precision (weighted) | 0.9604 |
| Recall (weighted) | 0.9583 |
| F1 (weighted) | 0.9581 |

The model correctly classified 23 of 24 test records. The single misclassification was between two categories that share overlapping vocabulary. Per-class scores are available in `artifacts/classification_report.txt` after training.

### How to read the confusion matrix

The confusion matrix is a grid where each row represents the real category and each column represents what the model predicted. A perfect model has all its numbers on the diagonal and zeros everywhere else. Off-diagonal numbers show where the model is getting confused — for example, if `RC-01` and `RC-06` share similar vocabulary (`401`, `unauthorized`), some `RC-06` records may be predicted as `RC-01`.

---

## How I would productionise this system

### Monitoring
- Expose a `/health` endpoint that confirms the model artifacts are loaded and the API is accepting requests
- Log every prediction with its `log_id`, predicted class, and confidence score to a centralised store (e.g. CloudWatch Logs or S3)
- Alert when the average confidence score across a rolling window drops below a threshold (e.g. 0.75) — this is an early signal that incoming logs no longer match the training distribution
- Track prediction class distribution over time — a sudden spike in one category or a flat distribution across all categories both indicate something has changed

### Drift detection
- Compare the TF-IDF feature distribution of incoming logs against the training distribution on a scheduled basis (e.g. daily)
- Flag when new high-frequency terms appear in incoming logs that were not in the training vocabulary — these are likely new error patterns the model has never seen
- Store a rolling sample of recent predictions and run a statistical test (e.g. Population Stability Index) against the training label distribution to detect label drift

### Scaling
- Containerise the FastAPI app with Docker and deploy behind a load balancer (e.g. AWS ALB) with multiple replicas
- The model artifacts are read-only at inference time — all replicas can share the same mounted volume or load from S3 on startup
- Cache the loaded model in memory per worker process — the current `LogInferenceEngine` already does this; ensure the container has enough memory allocated so the OS doesn't evict it

### Reliability
- Store all training artifacts in S3 with versioning enabled so any model version can be rolled back without redeployment
- Run the training pipeline in a separate job (e.g. AWS Batch or a scheduled Lambda) rather than via the `/preprocess-and-train` endpoint — keep training and serving concerns separate
- Add a confidence threshold below which predictions are flagged as `LOW_CONFIDENCE` and routed to a human review queue rather than returned as authoritative
- Implement a canary deployment pattern when releasing a retrained model — route a small percentage of traffic to the new model and compare its prediction distribution against the current model before full rollout