import pandas as pd
import json
from openai import OpenAI
from tqdm import tqdm
import time

# ---------- INIT ----------
client = OpenAI()

INPUT = "../data/reviews_sample_50.csv"
OUTPUT = "../outputs/scored_sample_50_v3.csv"

df = pd.read_csv(INPUT)

# ---------- PROMPT (FINAL CALIBRATED VERSION) ----------
PROMPT = """
You are evaluating Airbnb reviews.

Score the following dimensions (0–5):

1. Cleanliness:
Cleanliness, hygiene, and maintenance of the property.

2. Winter amenities:
Amenities that improve winter comfort (fireplace, heating, hot tub, sauna, heated floors).

3. View:
Scenic or environmental appeal (lake, mountains, waterfront, snow, windows with views).
IMPORTANT:
- If scenery or views are mentioned → assign at least 3
- Do NOT confuse with "beautiful home"

4. Coziness:
Emotional warmth, comfort, relaxing indoor experience.
IMPORTANT:
- Must reflect relaxation or warmth
- Do NOT assign for general positive reviews only

5. Activities:
Things to do nearby (skiing, wineries, hiking, attractions, exploration).
- Include implied activities if clearly suggested

6. Luxury:
Premium or high-end experience (design, finishes, upscale amenities).
IMPORTANT:
- Requires clear premium signal
- Do NOT assign for "nice", "great", or "beautiful" alone

7. Negative:
Any complaints, discomfort, inconvenience, or issues.

Scoring rules:
0 = not present  
1–2 = weak / slightly implied  
3 = moderate  
4 = strong  
5 = very strong  

IMPORTANT RULES:
- Do NOT rely only on keywords
- Capture IMPLIED meaning
- If clearly implied → assign at least 2–3
- Keep dimensions distinct

SHORT REVIEW RULE:
- If a review is short but clearly positive → assign moderate scores (2–3) where appropriate

Examples:

Text: "The house was clean and organized"
→ cleanliness: 4

Text: "Beautiful home"
→ view: 0, luxury: 2

Text: "We loved staying inside and relaxing"
→ coziness: 4

Text: "Fireplace and hot tub were amazing"
→ winter_amenities: 5

Text: "Close to skiing and wineries"
→ activities: 5

Text: "Lake views were incredible"
→ view: 5

Return JSON only:
{
  "cleanliness": 0,
  "winter_amenities": 0,
  "view": 0,
  "coziness": 0,
  "activities": 0,
  "luxury": 0,
  "negative": 0
}

Review:
<<<REVIEW>>>
"""

# ---------- SCORING FUNCTION ----------
def score_one(text, retries=2):
    for _ in range(retries):
        try:
            prompt_filled = PROMPT.replace("<<<REVIEW>>>", str(text)[:800])

            response = client.chat.completions.create(
                model="gpt-4o-mini",  # upgrade to gpt-5.4 if needed
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "user", "content": prompt_filled}
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
    results.append(score_one(text))

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