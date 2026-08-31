from datasets import load_dataset
from transformers import AutoTokenizer


DATASET_NAME = "sh0416/ag_news"
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 100


def load_data():
    return load_dataset(DATASET_NAME)


def combine_text(row):
    return {
        "text": row["title"] + " " + row["description"]
    }

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_token_type_ids=False
    )