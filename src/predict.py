from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MODEL_PATH = Path("models/distilbert-ag-news")

LABEL_NAMES = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    return tokenizer, model

def predict(text):
    tokenizer, model = load_model()

    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=100,
        return_tensors="pt",
        return_token_type_ids=False
    )

    model.eval()

    with torch.inference_mode():
        output = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"]
        )

    probabilities = torch.softmax(
        output.logits,
        dim=1
    )

    predicted_class = probabilities.argmax(dim=1).item()
    confidence = probabilities[0, predicted_class].item()

    return {
        "class": LABEL_NAMES[predicted_class],
        "confidence": confidence
    }