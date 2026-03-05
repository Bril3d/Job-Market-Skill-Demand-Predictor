import pandas as pd
import numpy as np
import random
import os

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
RAW_FILE = os.path.join(OUTPUT_DIR, "raw_jobs.csv")

def augment_data(target_count=8000):
    if not os.path.exists(RAW_FILE):
        print(f"Error: Base file {RAW_FILE} does not exist.")
        return
        
    df = pd.read_csv(RAW_FILE)
    if df.empty:
        print("Error: Base file is empty.")
        return
        
    print(f"Base data loaded: {len(df)} jobs.")
    
    current_count = len(df)
    needed = target_count - current_count
    
    if needed <= 0:
        print("Target already reached.")
        return
        
    print(f"Augmenting data to reach {target_count} jobs ({needed} synthetic rows)...")
    
    synthetic_samples = df.sample(n=needed, replace=True).copy()
    
    def modify_salary(sal):
        if pd.isna(sal) or not str(sal).strip():
            return sal
        try:
            parts = str(sal).replace('$', '').replace('k', '').split('-')
            if len(parts) == 2:
                s1 = int(parts[0].strip())
                s2 = int(parts[1].strip())
                shift = random.choice([-10, -5, 0, 5, 10])
                return f"${s1+shift}k - ${s2+shift}k"
        except:
            pass
        return sal

    synthetic_samples['salary'] = synthetic_samples['salary'].apply(modify_salary)
    
    final_df = pd.concat([df, synthetic_samples], ignore_index=True)
    final_df['job_id'] = [f"{row['job_id']}_{i}" if i >= current_count else row['job_id'] for i, row in final_df.iterrows()]
    
    final_df.to_csv(RAW_FILE, index=False)
    print(f"Successfully saved {len(final_df)} jobs to {RAW_FILE}")

if __name__ == "__main__":
    augment_data(8500)
