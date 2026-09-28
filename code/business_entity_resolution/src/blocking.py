import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import jellyfish
from collections import defaultdict
import math
import time

def build_inverted_index(df_target, col='business_name_stripped', min_idf=0.0):
    token_freq = defaultdict(int)
    doc_tokens = []
    
    for text in df_target[col]:
        tokens = set(str(text).split())
        doc_tokens.append(tokens)
        for t in tokens:
            token_freq[t] += 1
            
    n_docs = len(df_target)
    idf = {t: math.log(n_docs / f) for t, f in token_freq.items()}
    
    inv_index = defaultdict(list)
    max_docs = n_docs * 0.01  
    for i, tokens in enumerate(doc_tokens):
        for t in tokens:
            if token_freq[t] < max_docs and idf[t] >= min_idf:
                inv_index[t].append(i)
                
    return inv_index, idf, doc_tokens

def block_token_inverted(df_source, df_target, col='business_name_stripped', inv_index=None, idf=None):
    if inv_index is None or idf is None:
        inv_index, idf, _ = build_inverted_index(df_target, col)
    
    candidates = defaultdict(lambda: defaultdict(float))
    
    target_ids = df_target['entity_id'].values
    source_ids = df_source['entity_id'].values
    
    for i, text in enumerate(df_source[col]):
        s_id = source_ids[i]
        tokens = set(str(text).split())
        for t in tokens:
            if t in inv_index:
                w = idf[t]
                for match_idx in inv_index[t]:
                    t_id = target_ids[match_idx]
                    candidates[s_id][t_id] += w
                    
    return candidates

def block_tfidf_knn(df_source, df_target, col, analyzer='char_wb', ngram_range=(2,4), k=50, prefix='name', vec=None, nn=None):
    if vec is None or nn is None:
        vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=2)
        vec.fit(df_target[col].astype(str))
        target_mat = vec.transform(df_target[col].astype(str))
        nn = NearestNeighbors(n_neighbors=min(k, target_mat.shape[0]), metric='cosine', n_jobs=-1)
        nn.fit(target_mat)
        
    source_mat = vec.transform(df_source[col].astype(str))
    distances, indices = nn.kneighbors(source_mat)
    
    candidates = defaultdict(lambda: defaultdict(float))
    target_ids = df_target['entity_id'].values
    source_ids = df_source['entity_id'].values
    
    for i in range(source_mat.shape[0]):
        s_id = source_ids[i]
        for j in range(indices.shape[1]):
            t_id = target_ids[indices[i, j]]
            sim = 1.0 - distances[i, j]
            candidates[s_id][t_id] = max(candidates[s_id][t_id], sim)
            
    return candidates

def block_sorted_neighborhood(df_source, df_target, col='business_name_stripped', prefix_len=10, window=5):
    def get_key(text):
        return str(text)[:prefix_len].ljust(prefix_len)
        
    s_keys = df_source[col].apply(get_key)
    t_keys = df_target[col].apply(get_key)
    
    s_df = pd.DataFrame({'id': df_source['entity_id'], 'key': s_keys, 'type': 'S'})
    t_df = pd.DataFrame({'id': df_target['entity_id'], 'key': t_keys, 'type': 'T'})
    
    merged = pd.concat([s_df, t_df]).sort_values('key').reset_index(drop=True)
    
    candidates = defaultdict(lambda: defaultdict(float))
    
    n = len(merged)
    for i in range(n):
        if merged.loc[i, 'type'] == 'S':
            s_id = merged.loc[i, 'id']
            for j in range(max(0, i-window), min(n, i+window+1)):
                if merged.loc[j, 'type'] == 'T':
                    t_id = merged.loc[j, 'id']
                    candidates[s_id][t_id] = 1.0
                    
    return candidates

def block_phonetic(df_source, df_target, col='business_name_stripped', phon_to_target=None):
    def get_phonetic(text):
        tokens = str(text).split()
        if not tokens: return ""
        for t in tokens:
            if len(t) > 2:
                return jellyfish.metaphone(t)
        return jellyfish.metaphone(tokens[0])
        
    if phon_to_target is None:
        t_phon = df_target[col].apply(get_phonetic)
        phon_to_target = defaultdict(list)
        target_ids = df_target['entity_id'].values
        for i, p in enumerate(t_phon):
            phon_to_target[p].append(target_ids[i])
            
    s_phon = df_source[col].apply(get_phonetic)
    candidates = defaultdict(lambda: defaultdict(float))
    source_ids = df_source['entity_id'].values
    
    max_phon_matches = 1000
    for i, p in enumerate(s_phon):
        s_id = source_ids[i]
        matches = phon_to_target.get(p, [])
        if len(matches) < max_phon_matches:
            for t_id in matches:
                candidates[s_id][t_id] = 1.0
            
    return candidates

def combine_and_cap_candidates(c_token, c_tfidf_name, c_tfidf_addr, c_sn, c_phon, top_k=60):
    final_candidates = []
    
    s_ids = set(c_token.keys()).union(c_tfidf_name.keys(), c_tfidf_addr.keys(), c_sn.keys(), c_phon.keys())
    
    for s_id in s_ids:
        t_ids = set(c_token[s_id].keys()).union(
            c_tfidf_name[s_id].keys(),
            c_tfidf_addr[s_id].keys(),
            c_sn[s_id].keys(),
            c_phon[s_id].keys()
        )
        
        scores = []
        for t_id in t_ids:
            score_token = c_token[s_id].get(t_id, 0.0)
            score_tfidf_n = c_tfidf_name[s_id].get(t_id, 0.0)
            score_tfidf_a = c_tfidf_addr[s_id].get(t_id, 0.0)
            score_sn = c_sn[s_id].get(t_id, 0.0)
            score_phon = c_phon[s_id].get(t_id, 0.0)
            
            agg_score = score_token/10.0 + score_tfidf_n + score_tfidf_a + score_sn + score_phon
            scores.append((agg_score, t_id, score_token, score_tfidf_n, score_tfidf_a, score_sn, score_phon))
            
        scores.sort(reverse=True, key=lambda x: x[0])
        for rank, (agg, t_id, s_tok, s_tn, s_ta, s_sn, s_ph) in enumerate(scores[:top_k]):
            final_candidates.append({
                'source1_entity_id': s_id,
                'candidate_entity_id': t_id,
                'block_rank': rank + 1,
                'score_token': s_tok,
                'score_tfidf_name': s_tn,
                'score_tfidf_addr': s_ta,
                'score_sn': s_sn,
                'score_phon': s_ph
            })
            
    return pd.DataFrame(final_candidates)

def run_blocking(df_s1, df_s23, top_k=60, indices=None):
    start_time = time.time()
    
    if indices is None:
        print("Building blocking indices (this only happens once)...")
        inv_index, idf, _ = build_inverted_index(df_s23, 'business_name_stripped')
        
        vec_name = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), min_df=2)
        vec_name.fit(df_s23['business_name_norm'].astype(str))
        mat_name = vec_name.transform(df_s23['business_name_norm'].astype(str))
        nn_name = NearestNeighbors(n_neighbors=min(top_k, mat_name.shape[0]), metric='cosine', n_jobs=-1)
        nn_name.fit(mat_name)
        
        vec_addr = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), min_df=2)
        vec_addr.fit(df_s23['business_address_norm'].astype(str))
        mat_addr = vec_addr.transform(df_s23['business_address_norm'].astype(str))
        nn_addr = NearestNeighbors(n_neighbors=min(top_k, mat_addr.shape[0]), metric='cosine', n_jobs=-1)
        nn_addr.fit(mat_addr)
        
        def get_phonetic(text):
            tokens = str(text).split()
            if not tokens: return ""
            for t in tokens:
                if len(t) > 2:
                    return jellyfish.metaphone(t)
            return jellyfish.metaphone(tokens[0])
            
        t_phon = df_s23['business_name_stripped'].apply(get_phonetic)
        phon_to_target = defaultdict(list)
        target_ids = df_s23['entity_id'].values
        for i, p in enumerate(t_phon):
            phon_to_target[p].append(target_ids[i])
            
        indices = {
            'inv_index': inv_index, 'idf': idf,
            'vec_name': vec_name, 'nn_name': nn_name,
            'vec_addr': vec_addr, 'nn_addr': nn_addr,
            'phon_to_target': phon_to_target
        }
    
    c_token = block_token_inverted(df_s1, df_s23, inv_index=indices['inv_index'], idf=indices['idf'])
    c_tfidf_name = block_tfidf_knn(df_s1, df_s23, col='business_name_norm', k=top_k, vec=indices['vec_name'], nn=indices['nn_name'])
    c_tfidf_addr = block_tfidf_knn(df_s1, df_s23, col='business_address_norm', k=top_k, prefix='addr', vec=indices['vec_addr'], nn=indices['nn_addr'])
    c_sn = block_sorted_neighborhood(df_s1, df_s23)
    c_phon = block_phonetic(df_s1, df_s23, phon_to_target=indices['phon_to_target'])
    
    df_cand = combine_and_cap_candidates(c_token, c_tfidf_name, c_tfidf_addr, c_sn, c_phon, top_k=top_k)
    return df_cand, indices
