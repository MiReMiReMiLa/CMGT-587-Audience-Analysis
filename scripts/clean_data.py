import json
import pandas as pd

INPUT = "../data/dataset_airbnb-reviews-scraper.json"
OUTPUT = "../data/reviews_clean.csv"

with open(INPUT, "r") as f:
    data = json.load(f)

rows = []

for item in data:
    rows.append({
        "review_id": item.get("id"),
        "listing_id": item.get("reviewee", {}).get("id"),
        "text": item.get("text"),
        "createdAt": item.get("createdAt")
    })

df = pd.DataFrame(rows)

df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce")
df["month"] = df["createdAt"].dt.month

df = df.dropna(subset=["text"])

df.to_csv(OUTPUT, index=False)

print("Saved clean data:", OUTPUT)
