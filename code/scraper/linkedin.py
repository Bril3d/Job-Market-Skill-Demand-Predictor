import requests
import pandas as pd
from bs4 import BeautifulSoup
import time
import os
import random
import urllib.parse

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "raw_jobs.csv")

SEARCH_TERMS = [
    "Machine Learning Engineer",
    "Data Scientist",
    "Python Developer",
    "Data Engineer",
    "AI Engineer",
    "Software Engineer",
    "Backend Developer",
    "Frontend Developer",
    "DevOps"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.5",
}

def fetch_linkedin_jobs(target_count=1000):
    """Fetches job listings from LinkedIn public search pages."""
    all_jobs_data = []
    seen_job_ids = set()

    for term in SEARCH_TERMS:
        print(f"Fetching '{term}' jobs from LinkedIn...")
        query = urllib.parse.quote_plus(term)
        start = 0
        
        # Scrape up to 20 pages per term
        while start < 500:
            url = f"https://www.linkedin.com/jobs/search/?keywords={query}&location=Worldwide&start={start}"
            try:
                time.sleep(random.uniform(2.0, 5.0))
                response = requests.get(url, headers=HEADERS, timeout=15)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"Error fetching LinkedIn data at start {start}: {e}")
                break

            soup = BeautifulSoup(response.content, 'html.parser')
            job_cards = soup.find_all('div', class_='base-card')
            
            if not job_cards:
                print(f"No more jobs found for '{term}' at offset {start}.")
                break
                
            new_jobs = False
            for card in job_cards:
                job_id = card.get('data-entity-urn', '')
                if not job_id:
                    continue
                job_id = f"li_{job_id.split(':')[-1]}"
                
                if job_id in seen_job_ids:
                    continue
                    
                seen_job_ids.add(job_id)
                new_jobs = True

                title_elem = card.find('h3', class_='base-search-card__title')
                job_title = title_elem.text.strip() if title_elem else ''
                
                company_elem = card.find('h4', class_='base-search-card__subtitle')
                company_name = company_elem.text.strip() if company_elem else 'Unknown Company'

                location_elem = card.find('span', class_='job-search-card__location')
                location_str = location_elem.text.strip() if location_elem else 'Unknown Location'
                
                date_elem = card.find('time')
                posted_date = date_elem.get('datetime', '') if date_elem else ''
                
                link_elem = card.find('a', class_='base-card__full-link')
                job_link = link_elem.get('href', '') if link_elem else ''

                if job_title and company_name:
                    all_jobs_data.append({
                        "job_id": job_id,
                        "company": company_name,
                        "title": job_title,
                        "salary": "", # LinkedIn public search rarely shows salary
                        "location": location_str,
                        "tags": term,
                        "posted_date": posted_date,
                        "link": job_link
                    })
            
            if not new_jobs:
                break
                
            start += 25
            
    print(f"Successfully extracted {len(all_jobs_data)} unique jobs from LinkedIn.")
    return pd.DataFrame(all_jobs_data)

def append_to_raw(df_new):
    """Appends the new DataFrame to existing raw_jobs.csv and deduplicates."""
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
    print(f"Appended LinkedIn jobs. Removed {initial_count - final_count} duplicates.")
    print(f"Total dataset now contains {final_count} unique jobs.")

if __name__ == "__main__":
    print("Starting LinkedIn Scraper...")
    df = fetch_linkedin_jobs()
    if not df.empty:
        append_to_raw(df)
