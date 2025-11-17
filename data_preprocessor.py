import os
import re
from typing import List, Dict, Union


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s!?.,']", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_and_clean_reviews(base_dir: str) -> List[Dict[str, Union[str, int]]]:
    """
    Loads text reviews from specified 'pos' and 'neg' subfolders,
    cleans them, and returns a list of dictionaries.
    """
    reviews_data: List[Dict[str, Union[str, int]]] = []

    positive_dir = os.path.join(base_dir, "txt_sentoken", "pos")
    negative_dir = os.path.join(base_dir, "txt_sentoken", "neg")

    # Sort filenames for reproducibility
    for filename in sorted(os.listdir(positive_dir)):
        with open(os.path.join(positive_dir, filename), "r", encoding="utf-8") as f:
            raw_review = f.read()
            cleaned_review = clean_text(raw_review)
            reviews_data.append({"review": cleaned_review, "sentiment": 1})

    # Sort filenames here too
    for filename in sorted(os.listdir(negative_dir)):
        with open(os.path.join(negative_dir, filename), "r", encoding="utf-8") as f:
            raw_review = f.read()
            cleaned_review = clean_text(raw_review)
            reviews_data.append({"review": cleaned_review, "sentiment": 0})

    return reviews_data


if __name__ == "__main__":
    DATA_PATH = "./data/review_polarity"

    print("Loading and cleaning data...")
    all_processed_data: List[Dict[str, Union[str, int]]] = load_and_clean_reviews(
        DATA_PATH
    )

    print(f"Total reviews processed: {len(all_processed_data)}")
    print(
        f"First cleaned review (first 200 chars): {all_processed_data[0]['review'][:200]}..."
    )
    print(f"First label: {all_processed_data[0]['sentiment']}")

    # How you'd extract for models:
    # X = [item['review'] for item in all_processed_data]
    # y = [item['sentiment'] for item in all_processed_data]
    # print(f"\nExample of extracted X[0]: {X[0][:200]}...")
    # print(f"Example of extracted y[0]: {y[0]}")
