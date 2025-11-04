import os
import re
import time
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from ddgs import DDGS


# -------------------------------
# CONFIGURATION
# -------------------------------
TOPICS = {
    "training_science": [
        "endurance training physiology",
        "polarized training distance running",
        "lactate threshold running performance"
    ],
    "coaching_guides": [
        "marathon training plan beginner",
        "half marathon pacing strategy",
        "science of the long run"
    ],
    "nutrition": [
        "nutrition for distance runners",
        "carbohydrate loading endurance",
        "hydration strategy marathon"
    ],
    "injury_prevention": [
        "common running injuries prevention",
        "strength training for runners"
    ],
    "mindset": [
        "mental strategies for endurance athletes",
        "motivation techniques for marathon runners"
    ]
}

OUT_DIR = "data"
MAX_RESULTS_PER_QUERY = 3
SLEEP_BETWEEN_REQUESTS = 2  # polite delay in seconds

# -------------------------------
# UTILITIES
# -------------------------------
def clean_text(text):
    """Clean extracted HTML text."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def fetch_page(url):
    """Fetch and clean page text."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        if "text/html" not in r.headers.get("Content-Type", ""):
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "header", "footer", "nav", "form"]):
            tag.extract()
        return clean_text(soup.get_text())
    except Exception as e:
        print(f"[!] Failed to fetch {url}: {e}")
        return None

# -------------------------------
# MAIN
# -------------------------------
os.makedirs(OUT_DIR, exist_ok=True)

with DDGS() as ddgs:
    for category, queries in TOPICS.items():
        cat_dir = os.path.join(OUT_DIR, category)
        os.makedirs(cat_dir, exist_ok=True)
        print(f"\n📚 Collecting docs for: {category}")

        for q in queries:
            print(f"🔍 Searching for: {q}")
            results = list(ddgs.text(q, max_results=MAX_RESULTS_PER_QUERY))

            for i, res in enumerate(tqdm(results, desc=q)):
                url = res.get("href")
                if not url:
                    continue

                text = fetch_page(url)
                print(f"Fetched from {url}: {len(text) if text else 0} chars")
                if text and len(text) > 300:
                    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', q)[:40]
                    out_path = os.path.join(cat_dir, f"{safe_name}_{i}.txt")
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(f"URL: {url}\n\n{text}")
                    print(f"✅ Saved: {out_path}")

                time.sleep(SLEEP_BETWEEN_REQUESTS)
