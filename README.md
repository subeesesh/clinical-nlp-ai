# 🩺 Clinical NLP Diagnosis Assistant

A full-stack AI web application that analyzes clinical notes and predicts
the most likely disease based on symptoms and medical context.

Built with **FastAPI · scikit-learn · spaCy · Vanilla JS** — clean,
modular code designed as a portfolio project.

---

## ✨ Features

| Feature | Detail |
|---|---|
| 📝 Clinical note input | Textarea or drag-and-drop .txt file |
| 🔬 Disease prediction | TF-IDF + Logistic Regression classifier |
| 📊 Confidence scoring | Per-class probability distribution |
| 🔍 Entity extraction | Symptoms, medications, conditions |
| 🌐 REST API | FastAPI with auto-generated Swagger docs |
| 💻 Dashboard | Dark-themed, responsive UI |

---

## 🗂 Project Structure

```
clinical-nlp-ai/
├── backend/
│   ├── main.py              # FastAPI app — /diagnose endpoint
│   ├── classifier.py        # TF-IDF + Logistic Regression model
│   ├── entity_extractor.py  # Keyword-based medical NER
│   └── preprocess.py        # Text cleaning utilities
├── frontend/
│   ├── index.html           # Dashboard UI
│   ├── style.css            # Dark medical theme
│   └── script.js            # Fetch API + dynamic rendering
├── dataset/
│   └── medical_notes_dataset.csv   # Labeled training data
├── models/
│   └── trained_classifier.pkl      # Generated after training
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourname/clinical-nlp-ai.git
cd clinical-nlp-ai
```

### 2. Create a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Download spaCy language model

```bash
python -m spacy download en_core_web_sm
```

---

## 🧠 Train the Classifier

```bash
cd backend
python classifier.py
```

Output:
```
[INFO] Loading dataset from: ../dataset/medical_notes_dataset.csv
[INFO] Dataset size: 43 samples, 9 classes
[INFO] Training TF-IDF + Logistic Regression pipeline …
[RESULTS] Test Accuracy: 100.00%
[INFO] Model saved → ../models/trained_classifier.pkl
```

The trained model is saved to `models/trained_classifier.pkl` and
loaded automatically when the API starts.

---

## 🚀 Run the Backend API

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- API root:      http://localhost:8000
- Swagger docs:  http://localhost:8000/docs
- ReDoc:         http://localhost:8000/redoc

### Test the API with curl

```bash
curl -X POST http://localhost:8000/diagnose \
  -H "Content-Type: application/json" \
  -d '{"text": "Patient reports frequent urination, fatigue, and high blood sugar levels."}'
```

Expected response:
```json
{
  "diagnosis": "Diabetes",
  "confidence": 0.91,
  "symptoms": ["frequent urination", "fatigue", "high blood sugar"],
  "medications": [],
  "conditions": ["diabetes"],
  "all_probabilities": {
    "Diabetes": 0.91,
    "Hypertension": 0.03,
    "Heart Disease": 0.02,
    ...
  }
}
```

---

## 🖥 Open the Frontend Dashboard

With the backend running, open `frontend/index.html` directly in your browser:

```bash
# macOS
open frontend/index.html

# Linux
xdg-open frontend/index.html

# Windows
start frontend/index.html
```

Or serve it with Python's built-in server for proper MIME types:

```bash
cd frontend
python -m http.server 3000
# Open http://localhost:3000
```

---

## 🧬 Trained Disease Classes

| Disease | Key Symptoms |
|---|---|
| Diabetes | Frequent urination, fatigue, high blood sugar, blurred vision |
| Hypertension | High blood pressure, headaches, dizziness, nosebleeds |
| Heart Disease | Chest pain, palpitations, sweating, shortness of breath |
| Respiratory Infection | Cough, fever, wheezing, difficulty breathing |
| Arthritis | Joint pain, morning stiffness, swollen joints |
| Gastroenteritis | Diarrhea, abdominal pain, nausea, vomiting |
| Liver Disease | Jaundice, dark urine, elevated liver enzymes |
| Urinary Tract Infection | Burning urination, urinary urgency, pelvic pain |
| Meningitis | Stiff neck, high fever, sensitivity to light, altered consciousness |

---

## 🔧 Extending the Project

### Larger Dataset
Add more rows to `dataset/medical_notes_dataset.csv` and re-run:
```bash
python backend/classifier.py
```

### BERT-based Classifier
Replace `LogisticRegression` in `classifier.py` with a HuggingFace transformer:

```python
from transformers import pipeline
classifier = pipeline("text-classification", model="emilyalsentzer/Bio_ClinicalBERT")
```

### Clinical NER with scispaCy
```bash
pip install scispacy
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_ner_bc5cdr_md-0.5.3.tar.gz
```

Then update `entity_extractor.py` to use `en_ner_bc5cdr_md`.

### Docker Deployment
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt && python -m spacy download en_core_web_sm
COPY backend/ ./backend/
COPY dataset/ ./dataset/
COPY models/ ./models/
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI, Uvicorn |
| ML Model | scikit-learn (TF-IDF + Logistic Regression) |
| NLP / NER | spaCy, keyword matching |
| Deep Learning | PyTorch, HuggingFace Transformers (optional) |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Data | Pandas, NumPy |

---

## ⚠️ Medical Disclaimer

This tool is for **educational and research purposes only**.
It is **not** a substitute for professional medical advice, diagnosis, or treatment.
Always consult a qualified healthcare provider for medical decisions.

---

## 📄 License

MIT License — free to use, modify, and distribute.
