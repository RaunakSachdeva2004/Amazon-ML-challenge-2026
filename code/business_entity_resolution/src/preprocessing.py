"""
Preprocessing module for text normalization, entity cleanups, and field standardization.
"""
import re
import pandas as pd

# Legal Suffix Standardization dictionary
LEGAL_SUFFIXES = {
    r'\bcorp\b': 'corporation',
    r'\binc\b': 'incorporated',
    r'\bltd\b': 'limited',
    r'\bpvt\b': 'private',
    r'\bco\b': 'company',
    r'\bllc\b': 'llc',
    r'\bllp\b': 'llp',
    r'\bgmbh\b': 'gmbh',
    r'\bsa\b': 'sa',
    r'\bsas\b': 'sas',
    r'\bplc\b': 'plc',
}

# Common Address Abbreviation Standardizations
ADDRESS_ABBREVS = {
    r'\brd\b': 'road',
    r'\bst\b': 'street',
    r'\bave\b': 'avenue',
    r'\bblvd\b': 'boulevard',
    r'\bdr\b': 'drive',
    r'\bln\b': 'lane',
    r'\bpkwy\b': 'parkway',
    r'\bapt\b': 'apartment',
    r'\bste\b': 'suite',
    r'\bfl\b': 'floor',
    r'\bctr\b': 'center',
    r'\bplz\b': 'plaza',
    r'\bno\b': 'number',
    r'\bopp\b': 'opposite',
    r'\bnr\b': 'near',
}

def clean_text(text: str) -> str:
    """Basic lowercasing, punctuation stripping, and whitespace normalization."""
    if not isinstance(text, str) or pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[\&\+]', ' and ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_business_name(name: str) -> str:
    """Normalizes business name by expanding legal suffixes and standardizing terms."""
    text = clean_text(name)
    if not text:
        return ""
    for pattern, replacement in LEGAL_SUFFIXES.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_address(address: str) -> str:
    """Normalizes business address by expanding abbreviations and standardizing structure."""
    text = clean_text(address)
    if not text:
        return ""
    for pattern, replacement in ADDRESS_ABBREVS.items():
        text = re.sub(pattern, replacement, text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds preprocessed columns to a source DataFrame:
    - clean_name
    - clean_address
    - clean_country
    - full_text
    """
    df = df.copy()
    df['clean_name'] = df['business_name'].fillna('').apply(normalize_business_name)
    df['clean_address'] = df['business_address'].fillna('').apply(normalize_address)
    df['clean_country'] = df['country'].fillna('').apply(clean_text)
    
    # Combined representation
    df['full_text'] = (
        df['clean_name'] + " " + df['clean_address'] + " " + df['clean_country']
    ).str.strip()
    
    return df
