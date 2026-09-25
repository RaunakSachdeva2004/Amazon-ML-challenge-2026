"""
Blocking / Candidate Generation module.
Generates plausible candidate matching pairs (Source 1 -> Source 2 / Source 3)
using hybrid indexing (TF-IDF Cosine Similarity on word and char n-grams).
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Set, Tuple

class CandidateGenerator:
    def __init__(self, top_k_per_source: int = 30, similarity_threshold: float = 0.15):
        self.top_k_per_source = top_k_per_source
        self.similarity_threshold = similarity_threshold

    def generate_candidates(
        self, 
        df_s1: pd.DataFrame, 
        df_s2: pd.DataFrame, 
        df_s3: pd.DataFrame,
        same_country_only: bool = True
    ) -> Dict[str, Set[str]]:
        """
        Returns a dictionary mapping source1_entity_id -> set of candidate_entity_ids (S2 and S3).
        """
        candidates: Dict[str, Set[str]] = {s1_id: set() for s1_id in df_s1['entity_id']}

        # Combine S2 and S2+S3 for indexing
        s2_s3_df = pd.concat([df_s2, df_s3], ignore_index=True)
        s2_s3_ids = s2_s3_df['entity_id'].values
        s2_s3_countries = s2_s3_df['clean_country'].values
        s1_countries = df_s1['clean_country'].values

        # 1. TF-IDF on clean_name (char_wb n-grams)
        vec_name = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=1)
        vec_name.fit(pd.concat([df_s1['clean_name'], s2_s3_df['clean_name']]))

        X_s1_name = vec_name.transform(df_s1['clean_name'])
        X_s2s3_name = vec_name.transform(s2_s3_df['clean_name'])

        # 2. TF-IDF on full_text (word n-grams)
        vec_text = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1)
        vec_text.fit(pd.concat([df_s1['full_text'], s2_s3_df['full_text']]))

        X_s1_text = vec_text.transform(df_s1['full_text'])
        X_s2s3_text = vec_text.transform(s2_s3_df['full_text'])

        # Build Nearest Neighbors index
        nn_name = NearestNeighbors(n_neighbors=min(self.top_k_per_source, len(s2_s3_df)), metric='cosine', algorithm='brute')
        nn_name.fit(X_s2s3_name)

        nn_text = NearestNeighbors(n_neighbors=min(self.top_k_per_source, len(s2_s3_df)), metric='cosine', algorithm='brute')
        nn_text.fit(X_s2s3_text)

        # Query for name similarity
        distances_name, indices_name = nn_name.kneighbors(X_s1_name)
        # Query for text similarity
        distances_text, indices_text = nn_text.kneighbors(X_s1_text)

        s1_ids = df_s1['entity_id'].values

        for idx, s1_id in enumerate(s1_ids):
            s1_country = s1_countries[idx]

            # Collect name candidates
            for dist, cand_idx in zip(distances_name[idx], indices_name[idx]):
                sim = 1.0 - dist
                if sim >= self.similarity_threshold:
                    cand_id = s2_s3_ids[cand_idx]
                    cand_country = s2_s3_countries[cand_idx]
                    if not same_country_only or s1_country == cand_country or not s1_country or not cand_country:
                        candidates[s1_id].add(cand_id)

            # Collect full_text candidates
            for dist, cand_idx in zip(distances_text[idx], indices_text[idx]):
                sim = 1.0 - dist
                if sim >= self.similarity_threshold:
                    cand_id = s2_s3_ids[cand_idx]
                    cand_country = s2_s3_countries[cand_idx]
                    if not same_country_only or s1_country == cand_country or not s1_country or not cand_country:
                        candidates[s1_id].add(cand_id)

        return candidates

def candidate_dict_to_dataframe(candidate_map: Dict[str, Set[str]]) -> pd.DataFrame:
    """Converts candidate dictionary into standard candidate_pairs.tsv DataFrame format."""
    rows = []
    for s1_id, cand_ids in candidate_map.items():
        cand_str = ",".join(sorted(cand_ids)) if cand_ids else ""
        rows.append({'source1_entity_id': s1_id, 'candidate_entity_ids': cand_str})
    return pd.DataFrame(rows)
