import pandas as pd
import json
from openai import OpenAI
from tqdm import tqdm
import time

# ---------- INIT ----------
client = OpenAI()

INPUT = "../data/reviews_clean.csv"
OUTPUT = "../outputs/scored_reviews.csv"

df = pd.read_csv(INPUT)

# ---------- FINAL PROMPT ----------
PROMPT = """
You are evaluating Airbnb reviews.

Score the following dimensions (0–5):

1. Cleanliness:
Cleanliness, hygiene, and maintenance.

2. General amenities:
Practical, non-seasonal amenities.

Includes:
- kitchen supplies, toiletries, towels, games, appliances
- "well stocked", "everything we needed"

EXCLUDES:
- fireplace, hot tub, sauna, heating → winter_amenities
- high-end / premium features → luxury

3. Winter amenities:
Amenities that improve winter comfort:
- fireplace, heating, hot tub, sauna, heated floors

4. View:
Scenic/environmental appeal:
- lake, mountains, waterfront, scenery, windows

IMPORTANT:
- If view/scenery is mentioned → assign ≥3
- Do NOT confuse with "beautiful home"

5. Coziness:
Emotional warmth, comfort, relaxing indoor experience

IMPORTANT:
- Must reflect relaxation or warmth
- NOT just general positivity
- NOT functional (e.g., well stocked)

6. Activities:
Things to do nearby:
- skiing, wineries, hiking, exploration

7. Luxury:
Premium or high-end experience

IMPORTANT:
- requires clear premium signal
- NOT "nice", "great", "beautiful" alone

8. Negative:
Complaints or issues

----------------------

SCORING RULES:

0 = not present at all  

1 = very weak / vague mention  
    (generic praise, unclear signal)

2 = implied but meaningful  
    (clear signal, not emphasized)

3 = moderate / explicitly mentioned  

4 = strong  

5 = very strong / standout  

----------------------

IMPORTANT RULES:

- Capture IMPLIED meaning
- If clearly implied → assign ≥2
- Keep dimensions distinct (no overlap)

- Amenities must NOT include winter amenities
- Cozy must NOT include functionality
- Luxury must NOT include general praise

SHORT REVIEW RULE:
- If short but clearly positive → assign 1–2 instead of 0

----------------------

Examples:

Text: "The house was clean and organized"
→ cleanliness: 4

Text: "Well stocked kitchen and games"
→ amenities: 4

Text: "Fireplace and hot tub were amazing"
→ winter_amenities: 5

Text: "Lake views were incredible"
→ view: 5

Text: "We loved relaxing inside all weekend"
→ coziness: 4

Text: "Close to skiing and wineries"
→ activities: 5

Text: "High-end finishes and beautiful design"
→ luxury: 4

Text: "Beautiful place"
→ view: 0, luxury: 1

----------------------

Return JSON only:
{
  "cleanliness": 0,
  "amenities": 0,
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
                model="gpt-4o-mini",  # can upgrade to gpt-5.4 if needed
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
print("\nMissing values:")
print(final_df.isna().sum())

print("\nDistribution:")
print(final_df.describe())