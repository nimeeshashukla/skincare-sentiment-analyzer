import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split

print("1/3 Cleaning products and ingredients...")
# Handles both lowercase or uppercase folder name
data_dir = "Data" if os.path.exists("Data") else "data"
prod_path = os.path.join(data_dir, "product_info.csv")
rev_path = os.path.join(data_dir, "reviews_0-250.csv")

products = pd.read_csv(prod_path)

def clean_ingredients(raw):
    if pd.isna(raw):
        return ""
    # Strip variation tags, markdown brackets, and special characters
    text = re.sub(r"\[.*?\]|Variation \d+:?", "", str(raw))
    text = re.sub(r"[\r\n\t]+", " ", text)
    tokens = [item.strip().lower() for item in text.split(",") if len(item.strip()) > 1]
    return ", ".join(tokens)

products["cleaned_ingredients"] = products["ingredients"].apply(clean_ingredients)
products = products[["product_id", "product_name", "brand_name", "cleaned_ingredients"]].dropna()
products.to_csv(os.path.join(data_dir, "cleaned_products.csv"), index=False)
print(f"-> Saved {len(products)} cleaned products.")

print("2/3 Cleaning reviews and creating sentiment labels...")
reviews = pd.read_csv(rev_path, usecols=["product_id", "rating", "review_text"])
reviews = reviews.dropna(subset=["review_text", "rating"])

# Label sentiment: 0 = Negative (1-2 stars), 1 = Neutral (3 stars), 2 = Positive (4-5 stars)
def label_sentiment(r):
    if r <= 2:
        return 0
    elif r == 3:
        return 1
    return 2

reviews["sentiment"] = reviews["rating"].apply(label_sentiment)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

reviews["cleaned_text"] = reviews["review_text"].apply(clean_text)
reviews = reviews[reviews["cleaned_text"].str.len() > 10]

# Sample 10,000 per class (30,000 total) for fast, balanced training without memory bottlenecks
sampled_groups = [
    group.sample(n=min(len(group), 10000), random_state=42)
    for _, group in reviews.groupby("sentiment")
]
sample_df = pd.concat(sampled_groups, ignore_index=True)

print("3/3 Splitting train, validation, and test sets (70/15/15)...")
train_df, temp_df = train_test_split(
    sample_df, test_size=0.30, random_state=42, stratify=sample_df["sentiment"]
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df["sentiment"]
)

train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
val_df.to_csv(os.path.join(data_dir, "val.csv"), index=False)
test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)

print(f"Done! Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")