# src/dataset_builder.py
import pandas as pd
import os
from parser import parse_file
from template_extractor import normalize_msg

def build_dataset(raw_log_path='data/sample_logs.log', output_path='data/processed/parsed.pkl'):
    """
    Reads a raw log file, parses it, normalizes messages,
    and saves the result as a pandas DataFrame pickle file.
    """
    print("--- Starting Dataset Build ---")

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading and parsing raw log file: {raw_log_path}")
    rows = []
    for p in parse_file(raw_log_path):
        # Add a normalized message field for TF-IDF
        p['norm_msg'] = normalize_msg(p.get('msg', ''))
        rows.append(p)

    if not rows:
        print("No logs were parsed. Aborting.")
        return

    df = pd.DataFrame(rows)

    # Convert timestamp column to datetime objects for time-series analysis
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    print(f"Successfully parsed {len(df)} log lines.")

    # Save the processed DataFrame
    df.to_pickle(output_path)
    print(f"Processed dataset saved to: {output_path}")
    print("--- Dataset Build Finished ---")

if __name__ == '__main__':
    build_dataset()