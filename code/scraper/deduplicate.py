import pandas as pd
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
RAW_FILE = os.path.join(OUTPUT_DIR, "raw_jobs.csv")

def remove_duplicates():
    if not os.path.exists(RAW_FILE):
        print(f"Error: Base file {RAW_FILE} does not exist.")
        return
        
    df = pd.read_csv(RAW_FILE)
    initial_count = len(df)
    
    # Drop duplicates based on company, title, location, ignoring job_id and salary jitter
    df_dedup = df.drop_duplicates(subset=['company', 'title', 'location'])
    
    final_count = len(df_dedup)
    removed = initial_count - final_count
    
    df_dedup.to_csv(RAW_FILE, index=False)
    print(f"Removed {removed} duplicate rows.")
    print(f"Dataset now contains {final_count} unique jobs.")

if __name__ == "__main__":
    remove_duplicates()
