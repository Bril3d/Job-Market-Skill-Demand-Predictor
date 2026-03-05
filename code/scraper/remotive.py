import requests
import pandas as pd
import os
import time

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "raw_jobs.csv")

# GitHub Jobs API was shut down in 2021.
# Remotive provides an excellent, free API with thousands of tech jobs.
REMOTIVE_API = "https://remotive.com/api/remote-jobs"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

def fetch_remotive_jobs():
    """Fetches job listings from Remotive API."""
    all_jobs_data = []
    seen_job_ids = set()

    # Remotive doesn't require pagination by default, it usually returns ~1000-2000 jobs per category
    categories = ["software-dev", "data", "devops"]
    
    for category in categories:
        print(f"Fetching '{category}' jobs from Remotive...")
        try:
            time.sleep(1)
            response = requests.get(f"{REMOTIVE_API}?category={category}", headers=HEADERS, timeout=15)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"Error fetching data from Remotive for {category}: {e}")
            continue

        jobs = data.get('jobs', [])
        for item in jobs:
            job_id = f"remo_{item.get('id', '')}"
            
            if not job_id or job_id in seen_job_ids:
                continue
                
            seen_job_ids.add(job_id)

            company_name = item.get('company_name', '')
            job_title = item.get('title', '')
            
            salary = item.get('salary', '')
            location_str = item.get('candidate_required_location', 'Remote')
            
            item_tags = item.get('tags', [])
            tags_str = ", ".join(item_tags) if isinstance(item_tags, list) else str(item_tags)
            
            posted_date = item.get('publication_date', '')
            job_link = item.get('url', '')

            if job_title and company_name:
                all_jobs_data.append({
                    "job_id": job_id,
                    "company": company_name,
                    "title": job_title,
                    "salary": salary,
                    "location": location_str,
                    "tags": tags_str,
                    "posted_date": posted_date,
                    "link": job_link
                })
                
    print(f"Successfully extracted {len(all_jobs_data)} unique jobs from Remotive.")
    return pd.DataFrame(all_jobs_data)

def append_to_raw(df_new):
    """Appends to raw_jobs.csv and deduplicates."""
    if df_new.empty:
        print("No new jobs to append.")
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if os.path.exists(OUTPUT_FILE):
        df_existing = pd.read_csv(OUTPUT_FILE)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
        
    initial_count = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=['company', 'title', 'location'])
    final_count = len(df_combined)
    
    df_combined.to_csv(OUTPUT_FILE, index=False)
    print(f"Appended Remotive jobs. Removed {initial_count - final_count} duplicates.")
    print(f"Total dataset now contains {final_count} unique jobs.")

if __name__ == "__main__":
    print("Starting Remotive API Scraper...")
    df = fetch_remotive_jobs()
    if not df.empty:
        append_to_raw(df)
