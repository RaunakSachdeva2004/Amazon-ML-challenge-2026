import pandas as pd
import json

def inspect_file(path):
    print(f"--- Inspecting {path} ---")
    # Read a sample to save memory
    df = pd.read_csv(path, sep='\t', nrows=50000, dtype=str)
    print("Shape (sample):", df.shape)
    print("Columns:", df.columns.tolist())
    print("Null rates:")
    print(df.isnull().mean())
    print("Head:")
    print(df.head(3))
    if 'business_name' in df.columns:
        names = df['business_name'].dropna().str.lower()
        suffixes = names.str.split().str[-1].value_counts().head(20)
        print("Top last tokens in business_name (potential suffixes):")
        print(suffixes)
    if 'business_address' in df.columns:
        print("Sample addresses:")
        print(df['business_address'].dropna().head(10))

inspect_file('student_resource/dataset/train/train_source1.tsv')
inspect_file('student_resource/dataset/train/train_ground_truth.tsv')
