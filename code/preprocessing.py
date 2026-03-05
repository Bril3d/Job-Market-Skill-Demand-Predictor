import pandas as pd
import numpy as np
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

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

def extract_experience(title, tags):
    """
    Extracts years of experience if mentioned in title or tags.
    Returns 0 if not found.
    """
    text = (str(title) + " " + str(tags)).lower()
    matches = re.findall(r"(\d+)\s*\+?\s*years?", text)
    if matches:
        return max([int(m) for m in matches])
    return 0

def compute_demand_features(df):
    """
    Computes demand_score and demand_label based on tag frequency.
    """
    print("\nComputing Demand Features...")
    
    # 1. Parse tags list
    df["tag_list"] = df["tags"].fillna("").apply(lambda x: [t.strip().lower() for t in str(x).split(",") if t.strip()])
    
    # 2. Count global skill frequency
    all_tags = [tag for sublist in df["tag_list"] for tag in sublist]
    tag_counts = pd.Series(all_tags).value_counts().to_dict()
    print(f"  Unique tags found: {len(tag_counts)}")
    
    # 3. Compute demand_score per job (sum of tag frequencies)
    df["demand_score"] = df["tag_list"].apply(lambda tags: sum([tag_counts.get(tag, 0) for tag in tags]))
    
    # 4. Normalize demand_score to [0, 1]
    max_score = df["demand_score"].max()
    if max_score > 0:
        df["demand_score_norm"] = df["demand_score"] / max_score
    else:
        df["demand_score_norm"] = 0
        
    # 5. Create demand_label (1 if > median demand_score)
    median_score = df["demand_score"].median()
    df["demand_label"] = (df["demand_score"] > median_score).astype(int)
    
    print(f"  Demand Score Range: {df['demand_score'].min()} - {df['demand_score'].max()}")
    print(f"  Median Demand Score: {median_score}")
    print(f"  Label distribution: {df['demand_label'].value_counts(normalize=True).to_dict()}")
    
    return df

def add_engineered_features(df):
    """
    Adds binary features based on keyword groups in tags.
    """
    print("Adding engineered features...")
    
    # Helper to check keywords in tag list
    def check_keywords(tag_list, keywords):
        return 1 if any(kw in " ".join(tag_list) for kw in keywords) else 0

    df["num_skills"] = df["tag_list"].apply(len)
    df["has_ai"] = df["tag_list"].apply(lambda tags: check_keywords(tags, ["ai", "machine learning", "pytorch", "tensorflow", "nlp", "vision"]))
    df["has_cloud"] = df["tag_list"].apply(lambda tags: check_keywords(tags, ["aws", "azure", "gcp", "cloud", "docker", "kubernetes"]))
    df["has_backend"] = df["tag_list"].apply(lambda tags: check_keywords(tags, ["backend", "python", "java", "django", "fastapi", "node", "api"]))
    df["has_frontend"] = df["tag_list"].apply(lambda tags: check_keywords(tags, ["frontend", "react", "vue", "angular", "javascript", "typescript", "ui", "ux"]))
    
    return df

def preprocess_data(input_path, cleaned_path, features_path):
    print(f"Loading raw data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"  Total jobs loaded: {len(df)}")
    
    # 0. Remove duplicate job postings
    dedup_cols = ["title", "tags"]
    if "company" in df.columns:
        dedup_cols = ["title", "company", "tags"]
    before = len(df)
    df = df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
    print(f"  Duplicates removed: {before - len(df)} (remaining: {len(df)})")
    
    # 1. Domain Features (Basic Cleaning)
    print("\nExtracting domain features...")
    df["seniority"] = df["title"].apply(extract_seniority)
    df["category"] = df.apply(lambda row: map_category(row["title"], row["tags"]), axis=1)
    df["geo_tier"] = df["location"].apply(map_geographic_tier)
    df["years_exp"] = df.apply(lambda row: extract_experience(row["title"], row["tags"]), axis=1)
    
    # Save intermediate cleaned data
    df.to_csv(cleaned_path, index=False)
    print(f"Cleaned intermediate data saved to {cleaned_path}")
    
    # 2. Demand Target Generation
    df = compute_demand_features(df)
    
    # 3. Feature Engineering
    df = add_engineered_features(df)
    
    # 4. Handle Tags (Multi-Hot)
    print("Processing tags for ML...")
    mlb = MultiLabelBinarizer()
    tag_encoded = mlb.fit_transform(df["tag_list"])
    tag_df = pd.DataFrame(tag_encoded, columns=[f"tag_{c}" for c in mlb.classes_])
    
    # 5. TF-IDF on Title
    print("Extracting TF-IDF features from titles...")
    n_features = min(200, len(df))
    tfidf = TfidfVectorizer(max_features=n_features, stop_words="english")
    title_tfidf = tfidf.fit_transform(df["title"].fillna(""))
    actual_features = title_tfidf.shape[1]
    title_tfidf_df = pd.DataFrame(title_tfidf.toarray(), columns=[f"title_tfidf_{i}" for i in range(actual_features)])
    
    # 6. Combine Features
    print("Combining all features...")
    # Select original cols + engineered + encoded
    # NOTE: demand_score, demand_score_norm, and tag_frequency_score are EXCLUDED
    # to prevent target leakage (they are used to compute demand_label)
    core_cols = ["title", "seniority", "category", "geo_tier", "years_exp", "num_skills", 
                 "has_ai", "has_cloud", "has_backend", "has_frontend", 
                 "demand_label"]
    
    final_df = pd.concat([df[core_cols].reset_index(drop=True), tag_df, title_tfidf_df], axis=1)
    
    final_df.to_csv(features_path, index=False)
    print(f"\nFinal features dataset saved to {features_path}")
    print(f"Final dataset shape: {final_df.shape}")
    
    # Store the preprocessing artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(mlb, "models/tag_binarizer.joblib")
    joblib.dump(tfidf, "models/title_tfidf.joblib")
    print("Preprocessing artifacts saved to 'models/'")

if __name__ == "__main__":
    RAW_DATA = "data/raw_jobs.csv"
    CLEANED_DATA = "data/cleaned_jobs.csv"
    FEATURES_DATA = "data/features_jobs.csv"
    
    if os.path.exists(RAW_DATA):
        preprocess_data(RAW_DATA, CLEANED_DATA, FEATURES_DATA)
    else:
        print(f"Error: {RAW_DATA} not found!")
