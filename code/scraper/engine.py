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
    """Fetches job listings from RemoteOK API and returns a pandas DataFrame."""
    print(f"Fetching jobs from {API_URL}...")
    try:
        # Add a small delay to simulate human behavior
        time.sleep(random.uniform(1.0, 3.0))
        response = requests.get(API_URL, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

    try:
        data = response.json()
    except ValueError as e:
        print(f"Error parsing JSON: {e}")
        return pd.DataFrame()
        
    # The API usually returns a list of dictionaries. The first item is often a legal/info object.
    if not data or not isinstance(data, list):
        print("Unexpected data format from API.")
        return pd.DataFrame()
        
    jobs_data = []
    
    # Skip the first element if it's the legal notice
    start_idx = 1 if len(data) > 0 and data[0].get('legal') else 0
    
    for item in data[start_idx:]:
        job_id = item.get('id', '')
        company_name = item.get('company', '')
        job_title = item.get('position', '')
        salary = f"{item.get('salary_min', '')} - {item.get('salary_max', '')}" if item.get('salary_min') or item.get('salary_max') else ""
        location_str = item.get('location', '')
        tags = item.get('tags', [])
        tags_str = ", ".join(tags) if isinstance(tags, list) else str(tags)
        posted_date = item.get('date', '')
        job_link = item.get('url', '')

        if job_title and company_name:
            jobs_data.append({
                "job_id": job_id,
                "company": company_name,
                "title": job_title,
                "salary": salary.strip(" -"),
                "location": location_str,
                "tags": tags_str,
                "posted_date": posted_date,
                "link": job_link
            })
            
    print(f"Successfully extracted {len(jobs_data)} jobs from the API.")
            
    df = pd.DataFrame(jobs_data)
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