import pandas as pd
import numpy as np
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

def parse_salary(salary_str):
    """
    Parses salary strings and annualizes them.
    Examples:
    - "100000 - 150000" -> 125000
    - "80/hr" -> 80 * 2080 = 166400
    - "120000" -> 120000
    """
    if pd.isna(salary_str) or salary_str == "" or salary_str == "0":
        return None
    
    salary_str = str(salary_str).lower().replace(",", "")
    
    # Handle hourly rates
    hourly_match = re.search(r"(\d+)\s*/\s*hr", salary_str)
    if hourly_match:
        return float(hourly_match.group(1)) * 2080
    
    # Handle ranges (e.g., "100000 - 150000")
    range_match = re.findall(r"\d+", salary_str)
    if len(range_match) >= 2:
        return (float(range_match[0]) + float(range_match[1])) / 2
    elif len(range_match) == 1:
        return float(range_match[0])
    
    return None

def extract_seniority(title):
    title = str(title).lower()
    if any(word in title for word in ["intern", "student", "early career", "junior", "jr"]):
        return "Junior"
    if any(word in title for word in ["staff", "principal", "distinguished", "architect", "lead", "head"]):
        return "Staff/Principal"
    if "senior" in title or "sr" in title:
        return "Senior"
    return "Mid-Level"

def map_category(title, tags):
    text = (str(title) + " " + str(tags)).lower()
    if any(word in text for word in ["data scientist", "data science", "scientist"]):
        return "Data Science"
    if any(word in text for word in ["machine learning", "ml", "ai", "artificial intelligence", "computer vision", "nlp"]):
        return "ML Engineering"
    if any(word in text for word in ["data engineer", "big data", "etl", "data infrastructure"]):
        return "Data Engineering"
    if any(word in text for word in ["devops", "sre", "reliability", "infrastructure", "cloud engineer"]):
        return "DevOps"
    if any(word in text for word in ["backend", "python", "software engineer", "developer"]):
        return "Backend"
    if any(word in text for word in ["frontend", "react", "web", "ui", "ux"]):
        return "Frontend"
    if any(word in text for word in ["manager", "director", "lead", "head"]):
        return "Management"
    return "Others"

def map_geographic_tier(location):
    location = str(location).lower()
    tier_1_keywords = ["us", "usa", "united states", "uk", "united kingdom", "switzerland", "germany", "canada", "new york", "san francisco", "london"]
    if any(keyword in location for keyword in tier_1_keywords):
        return "Tier 1"
    return "Tier 2"

def preprocess_data(input_path, output_path):
    print(f"Loading raw data from {input_path}...")
    df = pd.read_csv(input_path)
    
    # 1. Salary Cleaning & Annualization
    print("Parsing salaries...")
    df["annual_salary"] = df["salary"].apply(parse_salary)
    
    # 2. Target Generation (is_high_salary)
    # 2025 threshold set to $120,000
    df["is_high_salary"] = (df["annual_salary"] > 120000).astype(int)
    
    # 3. Domain Features
    print("Extracting domain features...")
    df["seniority"] = df["title"].apply(extract_seniority)
    df["category"] = df.apply(lambda row: map_category(row["title"], row["tags"]), axis=1)
    df["geo_tier"] = df["location"].apply(map_geographic_tier)
    
    # 4. Handle Tags (Multi-Hot)
    print("Processing tags...")
    df["tag_list"] = df["tags"].fillna("").apply(lambda x: [t.strip() for t in x.split(",") if t.strip()])
    mlb = MultiLabelBinarizer()
    tag_encoded = mlb.fit_transform(df["tag_list"])
    tag_df = pd.DataFrame(tag_encoded, columns=[f"tag_{c}" for c in mlb.classes_])
    
    # 5. TF-IDF on Title
    print("Extracting TF-IDF features from titles...")
    tfidf = TfidfVectorizer(max_features=100, stop_words="english")
    title_tfidf = tfidf.fit_transform(df["title"].fillna(""))
    title_tfidf_df = pd.DataFrame(title_tfidf.toarray(), columns=[f"title_tfidf_{i}" for i in range(100)])
    
    # 6. Combine Features
    print("Combining features...")
    final_df = pd.concat([df, tag_df, title_tfidf_df], axis=1)
    
    # Drop intermediate or non-numeric columns for the final ML dataset if needed,
    # but for now we save the augmented dataframe
    final_df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    print(f"Final dataset shape: {final_df.shape}")
    
    # Store the preprocessing artifacts for the model pipeline
    os.makedirs("models", exist_ok=True)
    joblib.dump(mlb, "models/tag_binarizer.joblib")
    joblib.dump(tfidf, "models/title_tfidf.joblib")
    print("Preprocessing artifacts saved to 'models/'")

if __name__ == "__main__":
    RAW_DATA = "data/raw_jobs.csv"
    PROCESSED_DATA = "data/processed_jobs.csv"
    
    if os.path.exists(RAW_DATA):
        preprocess_data(RAW_DATA, PROCESSED_DATA)
    else:
        print(f"Error: {RAW_DATA} not found!")
