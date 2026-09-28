import pandas as pd
import unicodedata
import re
import string

SUFFIX_DICT = {
    'corporation': 'corp', 'co': 'corp', 'corp': 'corp',
    'limited': 'ltd', 'ltd': 'ltd', 'ltd.': 'ltd',
    'private': 'pvt', 'pvt': 'pvt',
    'incorporated': 'inc', 'inc': 'inc', 'inc.': 'inc',
    'llc': 'llc', 'l.l.c.': 'llc', 'l.l.c': 'llc',
    'llp': 'llp', 'pllc': 'pllc',
    'pc': 'pc', 'p.c.': 'pc',
    'lp': 'lp',
    'company': 'co'
}

ADDR_ABBR_DICT = {
    'st': 'street', 'st.': 'street',
    'rd': 'road', 'rd.': 'road',
    'ave': 'avenue', 'ave.': 'avenue',
    'apt': 'apartment', 'apt.': 'apartment',
    'blvd': 'boulevard', 'blvd.': 'boulevard',
    'dr': 'drive', 'dr.': 'drive',
    'ln': 'lane', 'ln.': 'lane',
    'ct': 'court', 'ct.': 'court',
    'pl': 'place', 'pl.': 'place',
    'sq': 'square', 'sq.': 'square',
    'hwy': 'highway', 'pkwy': 'parkway'
}

def clean_text(series):
    # Unicode NFKC
    s = series.astype(str).apply(lambda x: unicodedata.normalize('NFKC', x) if x != 'nan' else '')
    # Lowercase
    s = s.str.lower()
    # & -> and
    s = s.str.replace(r'\b&\b', ' and ', regex=True).str.replace('&', ' and ')
    return s

def strip_punctuation(series):
    # Strip punctuation
    s = series.str.replace(f"[{re.escape(string.punctuation)}]", " ", regex=True)
    # Whitespace collapse
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    return s

def normalize_names(df, col='business_name'):
    # Keep raw
    df[f'{col}_raw'] = df[col].astype(str).replace('nan', '')
    
    cleaned = clean_text(df[col])
    
    def process_suffix(text):
        tokens = text.split()
        if not tokens:
            return text, text
        
        # Check last token against suffix dict
        last_token = tokens[-1]
        if last_token in SUFFIX_DICT:
            norm_suffix = SUFFIX_DICT[last_token]
            norm_name = " ".join(tokens[:-1] + [norm_suffix])
            stripped_name = " ".join(tokens[:-1])
        else:
            # Check last token without punctuation if it had any
            last_token_clean = last_token.strip(string.punctuation)
            if last_token_clean in SUFFIX_DICT:
                norm_suffix = SUFFIX_DICT[last_token_clean]
                norm_name = " ".join(tokens[:-1] + [norm_suffix])
                stripped_name = " ".join(tokens[:-1])
            else:
                norm_name = text
                stripped_name = text
        return norm_name, stripped_name
        
    res = cleaned.apply(process_suffix)
    df[f'{col}_norm_suffix'] = res.apply(lambda x: x[0])
    df[f'{col}_stripped_suffix'] = res.apply(lambda x: x[1])
    
    # Finally apply punctuation stripping and whitespace collapse on these
    df[f'{col}_norm'] = strip_punctuation(df[f'{col}_norm_suffix'])
    df[f'{col}_stripped'] = strip_punctuation(df[f'{col}_stripped_suffix'])
    
    return df

def normalize_addresses(df, col='business_address'):
    df[f'{col}_raw'] = df[col].astype(str).replace('nan', '')
    
    cleaned = clean_text(df[col])
    
    def process_addr(text):
        tokens = text.split()
        out = []
        for t in tokens:
            t_clean = t.strip(string.punctuation)
            if t_clean in ADDR_ABBR_DICT:
                out.append(ADDR_ABBR_DICT[t_clean])
            elif t in ADDR_ABBR_DICT:
                out.append(ADDR_ABBR_DICT[t])
            else:
                out.append(t)
        return " ".join(out)
        
    addr_expanded = cleaned.apply(process_addr)
    df[f'{col}_norm'] = strip_punctuation(addr_expanded)
    
    # Regex extraction from df[f'{col}_norm']
    # PIN/ZIP (5-6 digits, maybe some spaces but let's just look for digit clusters)
    def extract_postal(text):
        m = re.findall(r'\b\d{5,6}\b', text)
        return m[-1] if m else None
    
    # Leading street number (digits at start)
    def extract_street_number(text):
        m = re.search(r'^\d+', text)
        return m.group(0) if m else None
    
    def extract_remainder(row):
        text = row[f'{col}_norm']
        pc = row['postal_code_guess']
        sn = row['street_number_guess']
        if pc:
            text = text.replace(pc, '')
        if sn:
            text = text.replace(sn, '', 1)
        return re.sub(r'\s+', ' ', text).strip()
        
    df['postal_code_guess'] = df[f'{col}_norm'].apply(extract_postal)
    df['street_number_guess'] = df[f'{col}_norm'].apply(extract_street_number)
    df['address_remainder'] = df.apply(extract_remainder, axis=1)
    
    return df

def process_dataframe(df):
    df = normalize_names(df)
    df = normalize_addresses(df)
    return df

if __name__ == "__main__":
    print("Testing normalize.py on sample data...")
    df = pd.read_csv('student_resource/dataset/train/train_source1.tsv', sep='\t', nrows=5)
    df = process_dataframe(df)
    for col in df.columns:
        print(f"--- {col} ---")
        print(df[col].head(3))
