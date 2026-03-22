"""
preprocess.py
-------------
Handles text cleaning and normalization for clinical notes.
Removes noise while preserving medically meaningful tokens.
"""

import re
import string


def clean_text(text: str) -> str:
    """
    Clean and normalize raw clinical note text.

    Steps:
    - Lowercase conversion
    - Remove digits (not clinically meaningful in TF-IDF context)
    - Remove punctuation
    - Normalize whitespace
    """
    text = text.lower()
    text = re.sub(r'\d+', '', text)                                        # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))       # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()                              # Normalize whitespace
    return text


def preprocess_batch(texts: list) -> list:
    """Apply cleaning to a list of texts."""
    return [clean_text(t) for t in texts]
