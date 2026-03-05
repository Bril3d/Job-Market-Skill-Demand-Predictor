import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import os
import random

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "raw_jobs.csv")

# WeWorkRemotely RSS feeds for various categories
RSS_FEEDS = [
    "https://weworkremotely.com/categories/remote-programming-jobs.rss",
    "https://weworkremotely.com/categories/remote-front-end-programming-jobs.rss",
    "https://weworkremotely.com/categories/remote-back-end-programming-jobs.rss",
    "https://weworkremotely.com/categories/remote-full-stack-programming-jobs.rss",
    "https://weworkremotely.com/categories/remote-machine-learning-jobs.rss", 
    "https://weworkremotely.com/categories/remote-data-jobs.rss",
    "https://weworkremotely.com/categories/remote-devops-sysadmin-jobs.rss",
    "https://weworkremotely.com/top-remote-jobs.rss"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/xml, text/xml, */*",
}

def fetch_wwr_jobs():
    """Fetches job listings from WeWorkRemotely RSS feeds."""
    all_jobs_data = []
    seen_job_ids = set()

    for feed_url in RSS_FEEDS:
        print(f"Fetching jobs from {feed_url}...")
        try:
            time.sleep(random.uniform(1.0, 2.5))
            response = requests.get(feed_url, headers=HEADERS, timeout=15)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from {feed_url}: {e}")
            continue

        soup = BeautifulSoup(response.content, 'xml')
        items = soup.find_all('item')
        
        for item in items:
            job_link = item.find('link').text if item.find('link') else ''
            
            # Use link as unique ID
            job_id = f"wwr_{job_link.split('/')[-1]}" if job_link else ""
            if not job_id or job_id in seen_job_ids:
                continue
                
            seen_job_ids.add(job_id)

            title_elem = item.find('title')
            title_text = title_elem.text if title_elem else ''
            
            # WWR usually formats title as "Company Name: Job Title"
            company_name = ""
            job_title = title_text
            if ": " in title_text:
                parts = title_text.split(": ", 1)
                company_name = parts[0]
                job_title = parts[1]
                
            posted_date = item.find('pubDate').text if item.find('pubDate') else ''
            
            # WWR RSS doesn't reliably expose tags, salary, or location structured in the feed
            # So we set defaults
            salary = ""
            location_str = "Remote"
            tags_str = ""
            
            # Try to extract tags from categories
            categories = item.find_all('category')
            if categories:
                tags_str = ", ".join([c.text for c in categories])

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
                
    print(f"Successfully extracted {len(all_jobs_data)} unique jobs from WeWorkRemotely.")
    return pd.DataFrame(all_jobs_data)

def append_to_raw(df_new):
    """Appends the new DataFrame to the existing raw_jobs.csv and deduplicates."""
    if df_new.empty:
        print("No new jobs to append.")
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if os.path.exists(OUTPUT_FILE):
        df_existing = pd.read_csv(OUTPUT_FILE)
        print(f"Loaded {len(df_existing)} existing jobs.")
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
        
    # Deduplicate strictly across the entire set
    initial_count = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=['company', 'title'])
    final_count = len(df_combined)
    
    df_combined.to_csv(OUTPUT_FILE, index=False)
    print(f"Appended jobs. Removed {initial_count - final_count} duplicates.")
    print(f"Total dataset now contains {final_count} unique jobs.")

if __name__ == "__main__":
    print("Starting WeWorkRemotely Scraper...")
    df = fetch_wwr_jobs()
    if not df.empty:
        append_to_raw(df)
