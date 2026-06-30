# AI-enabled Log Manager
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
│   ├── __init__.py
│   ├── config.py
│   ├── preprocess.py
│   ├── train.py
│   └── inference.py
│
├── artifacts/          # Created dynamically during training
│   ├── model.pkl
│   ├── vectorizer.pkl
│   └── encoder.pkl
│
├── main.py            # FastAPI Application
├── requirements.txt
└── README.md
```

## API Reference

> Base URL: `http://localhost:8000`  
> Interactive docs available at `http://localhost:8000/docs`

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

Classifies a single log record posted as a JSON body.

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

**Sample Response:**
```json
{
  "log_id": "LOG-001",
  "root_cause_label": "RC-02",
  "analysis": {
    "severity": "HIGH",
    "summary": "Component 'auth-service' encountered Authentication Failure. Detail: Failed login attempt on port 8080 after 3 retries"
  }
}
```

**Error Responses:**

| Status | Detail |
|--------|--------|
| `400`  | `Model is uninitialized. Run /preprocess-and-train.` |

---

### `POST /predict/file-ingest`
**Tag:** Inference Engine

Accepts a standard CSV file upload and returns predictions for all rows in the file.

**Request:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | `file` | A `.csv` file with at minimum `service` and `log_message` columns |

**Minimum Required CSV Columns:**
```
log_id, timestamp, service, log_message
```

**Sample CSV (`logs.csv`):**
```csv
log_id,timestamp,service,log_message
LOG-001,2024-01-15T08:32:11Z,auth-service,"[ERROR] Failed login attempt from 192.168.1.45"
LOG-002,2024-01-15T08:33:05Z,payment-service,"[ERROR] Connection timed out after 30s on port 5432"
```

**Sample Response:**
```json
[
  {
    "log_id": "LOG-001",
    "root_cause_label": "RC-02",
    "analysis": {
      "severity": "HIGH",
      "summary": "Component 'auth-service' encountered Authentication Failure. Detail: Failed login attempt from 192.168.1.45"
    }
  },
  {
    "log_id": "LOG-002",
    "root_cause_label": "RC-04",
    "analysis": {
      "severity": "CRITICAL",
      "summary": "Component 'payment-service' encountered Network Timeout. Detail: Connection timed out after 30s on port 5432"
    }
  }
]
```

**Error Responses:**

| Status | Detail |
|--------|--------|
| `400`  | `Model is uninitialized. Run /preprocess-and-train.` |
| `400`  | `Invalid extension format. API accepts raw standard csv configurations.` |
| `400`  | `Missing structural features. Required: ['service', 'log_message']` |

---

### `POST /predict/pickle-ingest`
**Tag:** High Volume Processing Optimization

Accepts a serialized Pandas DataFrame (`.pkl`) for high-throughput batch inference. Bypasses CSV string parsing overhead for production-scale ingestion pipelines.

**Request:** `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | `file` | A binary `.pkl` file containing a serialized Pandas DataFrame |

**Generating a valid pickle payload:**
```python
import pickle
import pandas as pd

df = pd.DataFrame([{
    "log_id": "LOG-001",
    "timestamp": "2024-01-15T08:32:11Z",
    "service": "auth-service",
    "log_message": "[ERROR] Failed login attempt from 192.168.1.45"
}])

with open("logs.pkl", "wb") as f:
    pickle.dump(df, f)
```

**Sample Response:**
```json
[
  {
    "log_id": "LOG-001",
    "root_cause_label": "RC-02",
    "analysis": {
      "severity": "HIGH",
      "summary": "Component 'auth-service' encountered Authentication Failure. Detail: Failed login attempt from 192.168.1.45"
    }
  }
]
```

**Error Responses:**

| Status | Detail |
|--------|--------|
| `400`  | `Model runtime uninitialized.` |
| `400`  | `Pickled byte payload must resolve to a valid Pandas DataFrame object.` |
| `400`  | `Failed to securely de-serialize incoming log payload: <error detail>` |

---

## 1. Architectural Strategy & Reasoning
Given the structural constraints of the 120-record synthetic log entry dataset, complex Deep Learning structures (RNNs/Transformers) or local LLMs present high over-fitting thresholds, performance baggage, and unnecessary runtime deployment costs.

This core application leverages a **TF-IDF Vectorizer coupled to a Linear Support Vector Machine (LinearSVC)**.
* **Reasoning:** System logs have fixed, highly repeated error tokens (`401`, `Connection timed out`, `OOM`). Linear models establish strict hyperplane delimiters across text distributions with high structural integrity, while requiring negligible resource envelopes ($<15\text{ms}$ execution latency).

## 2. Preprocessing Framework
* **Token Normalization:** Variable data fields (such as numerical indexes, ports, and unique IP configurations) are systematically extracted and standardized using specific token masks like `[ip_addr]` or `[num]`. This step decouples runtime metrics from the underlying structural error patterns.
* **Composite Feature Merging:** Combines `service`, `severity`, and raw text fields to preserve system-wide diagnostic context across classification tasks.

## 3. Observed System Tradeoffs
* **Execution Efficiency vs. Contextual Flexibility:** The combination of TF-IDF and LinearSVC delivers excellent throughput speeds but cannot map broader semantic variations. If engineers log an issue with highly creative, non-standard text structures, the model's accuracy degrades.

## 4. Current Scale Limitations
* **Cold Starts on New Services:** The system cannot reliably infer root causes for errors originating from newly introduced microservices without retraining the feature matrix.
* **Rigid Summaries:** Summaries rely on programmatic parsing rule sets, which lack the fluid variations of an advanced abstractive generative LLM.

## 5. Production Scaling & MLOps Strategy

### High-Volume Ingestion Management
* **Pickled Payload Endpoints:** Bypasses string parsing overhead by streaming directly serialized binary Pandas payloads straight to memory.
* **Broker Queue Architecture:** For large production deployments, ingest payloads through an Apache Kafka or AWS Kinesis stream buffer, running worker groups via Celery or Redis to avoid blocking API threads.

### Monitoring & Drift Detection
* **Data Drift:** Tracks performance shifts by checking incoming logs for vocabulary updates. If the system flags an unexpected jump in unknown tokens over a rolling 7-day window, it automatically triggers an alert to update the model.
* **Concept Drift:** Monitors variations in historical prediction frequencies (e.g., an unexplained spike in `RC-04` limits).

### Scalable Orchestration
* Containerize via **Docker** configurations running behind **Kubernetes (EKS)** clusters using Horizontal Pod Autoscaling (HPA) configured to scale based on inbound HTTP request rates.