import requests
import pandas as pd
import time
import os
import random

# Configuration
API_URL = "https://remoteok.com/api"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "raw_jobs.csv")

# Headers to mimic a real browser and avoid detection
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
}

def fetch_remoteok_jobs_api():
    """Fetches job listings from RemoteOK API across multiple tags and returns a pandas DataFrame."""
    tags_to_scrape = ["python", "machine-learning", "data", "engineer", "react", "backend", "frontend", "devops"]
    all_jobs_data = []
    seen_job_ids = set()

    for tag in tags_to_scrape:
        tag_url = f"{API_URL}?tags={tag}"
        print(f"Fetching jobs from {tag_url}...")
        try:
            # Add a small delay to simulate human behavior and avoid rate limits
            time.sleep(random.uniform(2.0, 4.0))
            response = requests.get(tag_url, headers=HEADERS, timeout=15)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for tag {tag}: {e}")
            continue

        try:
            data = response.json()
        except ValueError as e:
            print(f"Error parsing JSON for tag {tag}: {e}")
            continue
            
        if not data or not isinstance(data, list):
            print(f"Unexpected data format from API for tag {tag}.")
            continue
            
        # Skip the first element if it's the legal notice
        start_idx = 1 if len(data) > 0 and data[0].get('legal') else 0
        
        for item in data[start_idx:]:
            job_id = str(item.get('id', ''))
            
            # De-duplicate
            if not job_id or job_id in seen_job_ids:
                continue
                
            seen_job_ids.add(job_id)

            company_name = item.get('company', '')
            job_title = item.get('position', '')
            salary = f"{item.get('salary_min', '')} - {item.get('salary_max', '')}" if item.get('salary_min') or item.get('salary_max') else ""
            location_str = item.get('location', '')
            
            item_tags = item.get('tags', [])
            tags_str = ", ".join(item_tags) if isinstance(item_tags, list) else str(item_tags)
            
            posted_date = item.get('date', '')
            job_link = item.get('url', '')

            if job_title and company_name:
                all_jobs_data.append({
                    "job_id": job_id,
                    "company": company_name,
                    "title": job_title,
                    "salary": salary.strip(" -"),
                    "location": location_str,
                    "tags": tags_str,
                    "posted_date": posted_date,
                    "link": job_link
                })
                
    print(f"Successfully extracted {len(all_jobs_data)} unique jobs from the API.")
            
    df = pd.DataFrame(all_jobs_data)
    return df

def save_jobs(df):
    """Saves the DataFrame to the configured output CSV file."""
    if df.empty:
        print("No jobs to save.")
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully saved {len(df)} jobs to {OUTPUT_FILE}")

def main():
    print("Starting RemoteOK API Scraper...")
    df = fetch_remoteok_jobs_api()
    if not df.empty:
        save_jobs(df)
        print(f"Saved to: {OUTPUT_FILE}")
    else:
        print("Scraper finished without data.")

if __name__ == "__main__":
    main()