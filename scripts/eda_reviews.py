import json
import pandas as pd

# ---------- LOAD ----------
file_path = "../data/dataset_airbnb-reviews-scraper.json"

with open(file_path, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

print("Columns:", df.columns.tolist())

# ---------- BASIC COUNTS ----------
total_reviews = len(df)
print(f"\nTotal reviews: {total_reviews}")

# ---------- DUPLICATES (FIXED) ----------

# use safe columns (no dicts)
subset_cols = []

if "text" in df.columns:
    subset_cols.append("text")

if "createdAt" in df.columns:
    subset_cols.append("createdAt")

# optional: reviewer id if exists
if "reviewer" in df.columns:
    df["reviewer_id"] = df["reviewer"].apply(lambda x: x.get("id") if isinstance(x, dict) else None)
    subset_cols.append("reviewer_id")

dup_count = df.duplicated(subset=subset_cols).sum()

print(f"Duplicate reviews (based on text + date + reviewer): {dup_count}")

# ---------- DATE CLEAN ----------
# try common date fields
date_col = None
for col in ["date", "createdAt", "reviewDate"]:
    if col in df.columns:
        date_col = col
        break

if not date_col:
    raise ValueError("No date column found")

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# drop bad dates
df = df.dropna(subset=[date_col])

# ---------- WINTER FILTER (Nov–Mar) ----------
df["month"] = df[date_col].dt.month

winter_df = df[df["month"].isin([11, 12, 1, 2, 3])]

print(f"\nWinter reviews (Nov–Mar): {len(winter_df)}")

# ---------- OPTIONAL: breakdown ----------
print("\nReviews by month:")
print(df["month"].value_counts().sort_index())

print("\nWinter breakdown:")
print(winter_df["month"].value_counts().sort_index())
