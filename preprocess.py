"""
preprocess.py
-------------
Text preprocessing functions for Social Media Sentiment Analysis.
Preserved from the original Colab notebook pipeline.
"""

import re
import nltk
import emoji

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources on first use
def download_nltk_resources():
    """Download NLTK stopwords and WordNet if not already present."""
    for resource in ['stopwords', 'wordnet', 'omw-1.4']:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)


download_nltk_resources()

# Initialise shared objects once at module load
_stop_words = set(stopwords.words('english'))
_lemmatizer = WordNetLemmatizer()


def convert_emojis(text: str) -> str:
    """
    Convert emojis to their text description.

    E.g. 😊  →  ':smiling_face:'

    Parameters
    ----------
    text : str
        Raw input text that may contain emoji characters.

    Returns
    -------
    str
        Text with emojis replaced by their CLDR short names.
    """
    return emoji.demojize(str(text))


def clean_text(text: str) -> str:
    """
    Full preprocessing pipeline (mirrors the notebook):

    1. Lowercase
    2. Convert emojis to text tokens
    3. Remove URLs
    4. Remove @mentions
    5. Remove hashtag symbols (keep the word)
    6. Remove digits
    7. Remove punctuation / non-word characters
    8. Tokenise, remove stop-words
    9. Lemmatise
    10. Rejoin tokens

    Parameters
    ----------
    text : str
        Raw social-media post.

    Returns
    -------
    str
        Cleaned, lemmatised text ready for TF-IDF vectorisation.
    """
    if not isinstance(text, str):
        text = str(text)

    # Step 1 – lower-case
    text = text.lower()

    # Step 2 – emoji to text
    text = convert_emojis(text)

    # Step 3 – remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Step 4 – remove @mentions
    text = re.sub(r'@\w+', '', text)

    # Step 5 – strip # symbol but keep the word
    text = re.sub(r'#', '', text)

    # Step 6 – remove digits
    text = re.sub(r'\d+', '', text)

    # Step 7 – remove punctuation / non-word characters
    text = re.sub(r'[^\w\s]', '', text)

    # Step 8 – tokenise and remove stop-words
    words = [
        word for word in text.split()
        if word not in _stop_words
    ]

    # Step 9 – lemmatise
    words = [_lemmatizer.lemmatize(word) for word in words]

    # Step 10 – rejoin
    return ' '.join(words)


def preprocess_pipeline(text: str) -> str:
    """
    Convenience wrapper that runs the full preprocessing pipeline.

    Parameters
    ----------
    text : str
        Raw user input.

    Returns
    -------
    str
        Cleaned text suitable for vectorisation.
    """
    return clean_text(text)
