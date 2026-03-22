"""
entity_extractor.py
-------------------
Extracts medical entities (symptoms, medications, conditions) from clinical text.

Strategy
--------
1. Primary  : keyword matching against curated medical dictionaries (fast, no deps)
2. Secondary: spaCy en_core_web_sm NER for general entity recognition (if installed)

Extend by swapping in scispaCy (en_ner_bc5cdr_md) for production-grade clinical NER.
"""

from __future__ import annotations

# ── Symptom dictionary ────────────────────────────────────────────────────────
SYMPTOM_KEYWORDS: list[str] = [
    "fatigue", "fever", "cough", "headache", "nausea", "vomiting",
    "dizziness", "chest pain", "chest tightness", "chest discomfort",
    "shortness of breath", "difficulty breathing", "high blood sugar",
    "frequent urination", "urination at night", "weight loss", "weight gain",
    "blurred vision", "palpitations", "sweating", "joint pain", "stiffness",
    "morning stiffness", "swelling", "swollen joints", "diarrhea",
    "abdominal pain", "abdominal discomfort", "abdominal swelling",
    "loss of appetite", "increased thirst", "dry mouth", "tingling", "numbness",
    "irregular heartbeat", "high blood pressure", "elevated blood pressure",
    "runny nose", "sore throat", "wheezing", "mucus", "chills",
    "dark urine", "yellowing", "jaundice", "burning sensation", "urinary urgency",
    "stiff neck", "neck rigidity", "sensitivity to light", "photophobia",
    "nasal congestion", "blood in urine", "pelvic pain", "polyuria", "polydipsia",
    "slow wound healing", "nosebleed", "lower back pain", "dysuria",
    "loose stools", "stomach cramps", "dehydration", "right upper quadrant pain",
    "altered consciousness", "nuchal rigidity", "cloudy urine",
    "reduced exercise tolerance", "neck pain", "loss of consciousness",
    "troponin", "elevated troponin", "insulin resistance",
]

# ── Medication dictionary ─────────────────────────────────────────────────────
MEDICATION_KEYWORDS: list[str] = [
    "metformin", "insulin", "aspirin", "lisinopril", "atorvastatin",
    "amoxicillin", "ibuprofen", "acetaminophen", "paracetamol",
    "antihypertensives", "beta-blocker", "statin", "antibiotic",
    "bronchodilator", "anticoagulant", "diuretic", "prednisone",
    "warfarin", "heparin", "clopidogrel", "ramipril", "amlodipine",
    "furosemide", "omeprazole", "azithromycin", "ciprofloxacin",
    "doxycycline", "levothyroxine", "metoprolol", "losartan",
]

# ── Medical condition dictionary ──────────────────────────────────────────────
CONDITION_KEYWORDS: list[str] = [
    "diabetes", "hypertension", "heart disease", "respiratory infection",
    "arthritis", "pneumonia", "bronchitis", "gastroenteritis", "liver disease",
    "urinary tract infection", "meningitis", "coronary artery",
    "rheumatoid arthritis", "osteoarthritis", "anemia", "asthma", "copd",
    "myocardial infarction", "atrial fibrillation", "bacterial meningitis",
    "viral gastroenteritis", "insulin resistance", "hba1c",
]


def extract_entities(text: str) -> dict[str, list[str]]:
    """
    Extract medical entities from a clinical note via keyword matching.

    Args:
        text: Raw clinical note string.

    Returns:
        dict with:
            - symptoms   : list[str]
            - medications: list[str]
            - conditions : list[str]
    """
    text_lower = text.lower()

    # Match longest phrases first to avoid partial overlaps
    symptoms   = _match_keywords(text_lower, SYMPTOM_KEYWORDS)
    medications = _match_keywords(text_lower, MEDICATION_KEYWORDS)
    conditions  = _match_keywords(text_lower, CONDITION_KEYWORDS)

    return {
        "symptoms":    symptoms,
        "medications": medications,
        "conditions":  conditions,
    }


def _match_keywords(text: str, keywords: list[str]) -> list[str]:
    """Return deduplicated keywords found in text, longest first."""
    sorted_kws = sorted(keywords, key=len, reverse=True)
    found: list[str] = []
    for kw in sorted_kws:
        if kw in text and kw not in found:
            found.append(kw)
    return found


def try_spacy_ner(text: str) -> dict[str, list[str]]:
    """
    Optional general-purpose NER via spaCy en_core_web_sm.
    Returns entity-label → list of entity texts.
    Gracefully returns an empty dict if spaCy is unavailable.

    Install: pip install spacy && python -m spacy download en_core_web_sm
    """
    try:
        import spacy  # type: ignore
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        entities: dict[str, list[str]] = {}
        for ent in doc.ents:
            entities.setdefault(ent.label_, []).append(ent.text)
        return entities
    except Exception:
        return {}
