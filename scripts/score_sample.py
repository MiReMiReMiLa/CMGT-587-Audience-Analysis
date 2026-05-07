import pandas as pd
import json
from openai import OpenAI
from tqdm import tqdm
import time

# ---------- INIT ----------
client = OpenAI()

INPUT = "../data/reviews_sample_50.csv"
OUTPUT = "../outputs/scored_sample_50.csv"

df = pd.read_csv(INPUT)

# ---------- PROMPT ----------
PROMPT = """
You are evaluating Airbnb reviews.

Score the following dimensions (0–5):

1. Cleanliness: hygiene and maintenance
2. Winter amenities: heating, fireplace, hot tub, etc.
3. View: scenery, environment
4. Coziness: emotional warmth and comfort
5. Activities: things to do nearby
6. Luxury: premium quality
7. Negative: complaints or issues

Scoring:
0 = not mentioned
1–2 = weak
3 = moderate
4 = strong
5 = very strong

Important:
- Do NOT rely only on keywords
- Be consistent across reviews

Return JSON only with this format:
{{
  "cleanliness": 0,
  "winter_amenities": 0,
  "view": 0,
  "coziness": 0,
  "activities": 0,
  "luxury": 0,
  "negative": 0
}}

Review:
\"\"\"
{review}
\"\"\"
"""

# ---------- SCORING FUNCTION ----------
def score_one(text, retries=2):
    for _ in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # you can switch to gpt-5.4 if needed
                temperature=0,
                response_format={"type": "json_object"},  # ⭐ guarantees valid JSON
                messages=[
                    {"role": "user", "content": PROMPT.format(review=str(text)[:800])}
                ],
            )

            content = response.choices[0].message.content
            return json.loads(content)

        except Exception as e:
            print("Retrying due to error:", e)
            time.sleep(1)

    return None


# ---------- MAIN LOOP ----------
results = []

for text in tqdm(df["text"]):
    result = score_one(text)
    results.append(result)

# ---------- MERGE ----------
score_df = pd.DataFrame(results)

final_df = pd.concat([df.reset_index(drop=True), score_df], axis=1)

# ---------- SAVE ----------
final_df.to_csv(OUTPUT, index=False)

print("✅ Saved:", OUTPUT)


# ---------- QUICK CHECK ----------
print("\nMissing values per column:")
print(final_df.isna().sum())

print("\nScore distribution preview:")
print(final_df.describe())