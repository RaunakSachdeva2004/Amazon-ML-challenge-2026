import pandas as pd
import numpy as np
from rapidfuzz import fuzz, distance

def safe_str(s):
    if pd.isna(s): return ""
    return str(s)

def extract_features(df_s1, df_s23, df_cands):
    print("Extracting features...")
    # join candidates with s1 and s23 to get features
    s1_dict = df_s1.set_index('entity_id').to_dict('index')
    s23_dict = df_s23.set_index('entity_id').to_dict('index')
    
    features = []
    
    for idx, row in df_cands.iterrows():
        s1_id = row['source1_entity_id']
        t_id = row['candidate_entity_id']
        
        s1 = s1_dict.get(s1_id)
        t = s23_dict.get(t_id)
        
        if not s1 or not t:
            continue
            
        f = {}
        # Keep keys
        f['source1_entity_id'] = s1_id
        f['candidate_entity_id'] = t_id
        
        # Blocking features
        f['block_rank'] = row['block_rank']
        f['score_token'] = row['score_token']
        f['score_tfidf_name'] = row['score_tfidf_name']
        f['score_tfidf_addr'] = row['score_tfidf_addr']
        f['score_sn'] = row['score_sn']
        f['score_phon'] = row['score_phon']
        
        # Meta features
        f['country_match'] = 1 if safe_str(s1['country']).lower() == safe_str(t['country']).lower() else 0
        f['source_pair'] = 1 if 'S2' in t_id else 0  # 1 for S2, 0 for S3
        
        # Name features
        n1_raw = safe_str(s1.get('business_name_raw', ''))
        n2_raw = safe_str(t.get('business_name_raw', ''))
        n1_norm = safe_str(s1.get('business_name_norm', ''))
        n2_norm = safe_str(t.get('business_name_norm', ''))
        
        # Normalized levenshtein
        f['name_lev_raw'] = distance.Levenshtein.normalized_similarity(n1_raw, n2_raw)
        f['name_lev_norm'] = distance.Levenshtein.normalized_similarity(n1_norm, n2_norm)
        
        # Jaro-Winkler
        f['name_jw'] = distance.JaroWinkler.similarity(n1_norm, n2_norm)
        
        # Token Jaccard / Dice
        t1 = set(n1_norm.split())
        t2 = set(n2_norm.split())
        inter = len(t1 & t2)
        union = len(t1 | t2)
        f['name_jaccard'] = inter / union if union > 0 else 0
        f['name_dice'] = 2 * inter / (len(t1) + len(t2)) if len(t1) + len(t2) > 0 else 0
        
        # Token sort / set
        f['name_token_sort'] = fuzz.token_sort_ratio(n1_norm, n2_norm) / 100.0
        f['name_token_set'] = fuzz.token_set_ratio(n1_norm, n2_norm) / 100.0
        
        # Length diff
        l1 = len(n1_norm)
        l2 = len(n2_norm)
        f['name_len_diff_abs'] = abs(l1 - l2)
        f['name_len_diff_rel'] = abs(l1 - l2) / max(l1, l2, 1)
        
        # Acronym
        acr1 = "".join([x[0] for x in n1_norm.split() if x])
        acr2 = "".join([x[0] for x in n2_norm.split() if x])
        f['name_acronym_match'] = 1 if (acr1 == n2_norm.replace(" ", "") or acr2 == n1_norm.replace(" ", "")) and len(acr1)>1 else 0
        
        # Suffix stripped exact
        n1_strip = safe_str(s1.get('business_name_stripped', ''))
        n2_strip = safe_str(t.get('business_name_stripped', ''))
        f['name_exact_strip'] = 1 if n1_strip and n1_strip == n2_strip else 0
        
        # Address features
        a1_norm = safe_str(s1.get('business_address_norm', ''))
        a2_norm = safe_str(t.get('business_address_norm', ''))
        
        f['addr_lev_norm'] = distance.Levenshtein.normalized_similarity(a1_norm, a2_norm)
        f['addr_jw'] = distance.JaroWinkler.similarity(a1_norm, a2_norm)
        
        at1 = set(a1_norm.split())
        at2 = set(a2_norm.split())
        ainter = len(at1 & at2)
        aunion = len(at1 | at2)
        f['addr_jaccard'] = ainter / aunion if aunion > 0 else 0
        
        # Postal code match
        p1 = safe_str(s1.get('postal_code_guess', ''))
        p2 = safe_str(t.get('postal_code_guess', ''))
        if p1 and p2:
            f['addr_pc_match'] = 1 if p1 == p2 else -1
        else:
            f['addr_pc_match'] = 0
            
        # Street number match
        sn1 = safe_str(s1.get('street_number_guess', ''))
        sn2 = safe_str(t.get('street_number_guess', ''))
        if sn1 and sn2:
            f['addr_sn_match'] = 1 if sn1 == sn2 else -1
        else:
            f['addr_sn_match'] = 0
            
        # LCS length norm
        lcs = distance.LCSseq.similarity(a1_norm, a2_norm)
        f['addr_lcs_norm'] = lcs / max(len(a1_norm), len(a2_norm), 1)
        
        # Address length diff
        al1 = len(a1_norm)
        al2 = len(a2_norm)
        f['addr_len_diff_rel'] = abs(al1 - al2) / max(al1, al2, 1)
        
        features.append(f)
        
    df_features = pd.DataFrame(features)
    return df_features
