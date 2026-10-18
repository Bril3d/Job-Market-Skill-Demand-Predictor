import pandas as pd
import re
import os

def extract_seniority(title):
    title = str(title).lower()
    if any(word in title for word in ["intern", "student", "early career", "junior", "jr"]):
        return "Junior"
    if any(word in title for word in ["staff", "principal", "distinguished", "architect", "lead", "head"]):
        return "Staff/Principal"
    if "senior" in title or "sr" in title:
        return "Senior"
    return "Mid-Level"

def map_category(title, tags=""):
    # Primary classification driven by title, supplemented by tags if needed
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

def extract_experience(title, tags=""):
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
    
    LIMITATION NOTE (Target Definition Validity):
    demand_score is computed as the sum of tag frequencies in the scraped corpus.
    This creates an inherent volume bias (jobs with more tags receive higher scores,
    with correlation ~0.915). To prevent circular target reconstruction, input features
    must NOT include tag counts (num_skills) or tag-based keyword flags (e.g. has_cloud).
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

def preprocess_data(input_path, cleaned_path, features_path):
    print(f"Loading raw data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"  Total jobs loaded: {len(df)}")
    
    # 0. Initial deduplication
    dedup_cols = ["title", "company", "tags"]
    dedup_cols = [c for c in dedup_cols if c in df.columns]
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
    
    # 3. Deduplication by Title
    # To prevent identical job postings from appearing across train and test splits,
    # deduplicate by title.
    before_title_dedup = len(df)
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)
    print(f"  Title deduplication: removed {before_title_dedup - len(df)} duplicate titles (final: {len(df)})")
    
    # 4. Feature Selection WITHOUT Target-Derived Leakage
    # NOTE: TF-IDF vectorization is deliberately NOT performed here.
    # Fitting TF-IDF across the entire dataset before train/test split causes data leakage.
    # TfidfVectorizer is moved inside the scikit-learn ColumnTransformer in modeling.py.
    # Tag-derived shortcut flags (has_cloud, has_backend, etc.) are excluded because
    # they trivially reconstruct the tag-frequency definition of demand_score.
    core_cols = ["title", "seniority", "category", "geo_tier", "years_exp", "demand_label"]
    final_df = df[core_cols].copy()
    
    final_df.to_csv(features_path, index=False)
    print(f"\nFinal features dataset saved to {features_path}")
    print(f"Final dataset shape: {final_df.shape}")
    print(f"Final class distribution:\n{final_df['demand_label'].value_counts().to_dict()}")

if __name__ == "__main__":
    RAW_DATA = "data/raw_jobs.csv"
    CLEANED_DATA = "data/cleaned_jobs.csv"
    FEATURES_DATA = "data/features_jobs.csv"
    
    if os.path.exists(RAW_DATA):
        preprocess_data(RAW_DATA, CLEANED_DATA, FEATURES_DATA)
    else:
        print(f"Error: {RAW_DATA} not found!")
