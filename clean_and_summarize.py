import os
from bs4 import BeautifulSoup
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

LANGUAGE = "english"
SUMMARY_SENTENCES = 5

input_dir = "data"
output_dir = "data_clean"
os.makedirs(output_dir, exist_ok=True)

summarizer = LsaSummarizer(Stemmer(LANGUAGE))
summarizer.stop_words = get_stop_words(LANGUAGE)

def clean_text(text):
    soup = BeautifulSoup(text, "html.parser")
    cleaned = soup.get_text(separator=" ")
    cleaned = ' '.join(cleaned.split())
    return cleaned

def summarize_text(text):
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    summary = summarizer(parser.document, SUMMARY_SENTENCES)
    return " ".join(str(sentence) for sentence in summary)

for topic_folder in os.listdir(input_dir):
    topic_path = os.path.join(input_dir, topic_folder)
    if not os.path.isdir(topic_path):
        continue

    out_topic_path = os.path.join(output_dir, topic_folder)
    os.makedirs(out_topic_path, exist_ok=True)

    for file in os.listdir(topic_path):
        if not file.endswith(".txt"):
            continue

        file_path = os.path.join(topic_path, file)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        cleaned = clean_text(text)
        summary = summarize_text(cleaned)

        out_file_path = os.path.join(out_topic_path, file)
        with open(out_file_path, "w", encoding="utf-8") as f:
            f.write(summary)

        print(f"✅ Cleaned & summarized: {out_file_path}")
