from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score


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

def normalize_label(row):
    return {
        "label": row["label"] - 1
    }


def prepare_datasets():
    dataset = load_data()

    dataset = dataset.map(combine_text)

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True
    )

    train_val_dataset = tokenized_dataset["train"].train_test_split(
        test_size=0.1,
        seed=42
    )

    train_dataset = train_val_dataset["train"]
    val_dataset = train_val_dataset["test"]
    test_dataset = tokenized_dataset["test"]

    train_dataset = train_dataset.map(normalize_label)
    val_dataset = val_dataset.map(normalize_label)
    test_dataset = test_dataset.map(normalize_label)

    columns = [
        "input_ids",
        "attention_mask",
        "label"
    ]

    train_dataset.set_format(type="torch", columns=columns)
    val_dataset.set_format(type="torch", columns=columns)
    test_dataset.set_format(type="torch", columns=columns)

    return train_dataset, val_dataset, test_dataset

def build_model():
    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=4
    )


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predicted_classes = predictions.argmax(axis=1)

    return {
        "accuracy": accuracy_score(
            labels,
            predicted_classes
        )
    }

def train():
    train_dataset, val_dataset, _ = prepare_datasets()

    train_dataset = train_dataset.shuffle(seed=42).select(
        range(20000)
    )

    val_dataset = val_dataset.shuffle(seed=42).select(
        range(2000)
    )

    model = build_model()

    training_args = TrainingArguments(
        output_dir="distilbert-ag-news",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        dataloader_pin_memory=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    results = trainer.evaluate()

    print(results)

    model.save_pretrained(
        "models/distilbert-ag-news"
    )

    tokenizer.save_pretrained(
        "models/distilbert-ag-news"
    )


if __name__ == "__main__":
    train()