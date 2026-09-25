"""
Feature extraction module for pairwise business entity resolution.
"""
import re
import pandas as pd
import numpy as np
from rapidfuzz import fuzz, distance

def get_char_ngrams(text: str, n: int = 3) -> set:
    if not text or len(text) < n:
        return set()
    return set(text[i:i+n] for i in range(len(text) - n + 1))

def jaccard_sim(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0

def extract_digits(text: str) -> set:
    return set(re.findall(r'\d+', str(text)))

def compute_pairwise_features(row1: pd.Series, row2: pd.Series) -> dict:
    """Computes a dictionary of similarity features between two business entity records."""
    name1 = str(row1.get('clean_name', ''))
    name2 = str(row2.get('clean_name', ''))
    
    addr1 = str(row1.get('clean_address', ''))
    addr2 = str(row2.get('clean_address', ''))
    
    ctry1 = str(row1.get('clean_country', ''))
    ctry2 = str(row2.get('clean_country', ''))

    feats = {}

    # Name features
    feats['name_fuzz_ratio'] = fuzz.ratio(name1, name2) / 100.0
    feats['name_token_sort_ratio'] = fuzz.token_sort_ratio(name1, name2) / 100.0
    feats['name_token_set_ratio'] = fuzz.token_set_ratio(name1, name2) / 100.0
    feats['name_jaro_winkler'] = distance.JaroWinkler.similarity(name1, name2)

    tokens1 = set(name1.split())
    tokens2 = set(name2.split())
    feats['name_jaccard_tokens'] = jaccard_sim(tokens1, tokens2)
    feats['name_jaccard_char3grams'] = jaccard_sim(get_char_ngrams(name1, 3), get_char_ngrams(name2, 3))

    first_word1 = name1.split()[0] if name1.split() else ""
    first_word2 = name2.split()[0] if name2.split() else ""
    feats['name_first_word_match'] = 1.0 if first_word1 and first_word1 == first_word2 else 0.0

    # Address features
    feats['addr_fuzz_ratio'] = fuzz.ratio(addr1, addr2) / 100.0
    feats['addr_token_sort_ratio'] = fuzz.token_sort_ratio(addr1, addr2) / 100.0
    feats['addr_token_set_ratio'] = fuzz.token_set_ratio(addr1, addr2) / 100.0
    feats['addr_jaro_winkler'] = distance.JaroWinkler.similarity(addr1, addr2)

    addr_tokens1 = set(addr1.split())
    addr_tokens2 = set(addr2.split())
    feats['addr_jaccard_tokens'] = jaccard_sim(addr_tokens1, addr_tokens2)
    feats['addr_jaccard_char3grams'] = jaccard_sim(get_char_ngrams(addr1, 3), get_char_ngrams(addr2, 3))

    # Digit / PIN / ZIP code overlap
    digits1 = extract_digits(addr1)
    digits2 = extract_digits(addr2)
    feats['digits_jaccard'] = jaccard_sim(digits1, digits2)

    # Country feature
    feats['country_match'] = 1.0 if ctry1 == ctry2 and ctry1 != "" else (0.5 if not ctry1 or not ctry2 else 0.0)

    # Length ratios
    feats['name_len_diff'] = abs(len(name1) - len(name2))
    feats['addr_len_diff'] = abs(len(addr1) - len(addr2))

    return feats
