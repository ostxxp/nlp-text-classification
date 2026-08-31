# NLP Text Classification

Классификация новостей AG News на 4 категории:

- World
- Sports
- Business
- Sci/Tech

В проекте сравниваются три подхода к текстовой классификации: классический TF-IDF baseline, собственная PyTorch-модель с обучаемыми embeddings и fine-tuned DistilBERT.

## Результаты

| Модель | Test Accuracy |
|---|---:|
| TF-IDF + Logistic Regression | 0.915 |
| PyTorch Embedding Classifier | 0.913 |
| Fine-tuned DistilBERT | 0.920 |

Лучший результат показал DistilBERT — **92.0% accuracy** на официальной test-выборке AG News.

При этом классический TF-IDF baseline показал сопоставимое качество при значительно меньшей вычислительной стоимости.

## Подходы

### TF-IDF + Logistic Regression

Тексты преобразуются в разреженные TF-IDF-векторы размерностью до 20 000 признаков.

На полученных признаках обучается Logistic Regression.

Этот подход используется как классический NLP baseline.

### PyTorch Embedding Classifier

Реализована собственная нейросетевая модель на PyTorch:

```text
text → token IDs → Embedding → mean pooling → Linear → class logits
```

Модель использует:

- vocabulary из 20 000 токенов;
- padding и unknown tokens;
- обучаемые embeddings;
- mini-batch training;
- train / validation split;
- CrossEntropyLoss;
- Adam optimizer.

### DistilBERT

Используется предобученная модель `distilbert-base-uncased` из Hugging Face Transformers.

Pipeline:

```text
text → tokenizer → input_ids + attention_mask → DistilBERT → 4 class logits
```

Для fine-tuning использовалась подвыборка из 20 000 train-примеров.

## Датасет

Используется AG News — датасет новостей с четырьмя классами.

- Train: 120 000 примеров
- Test: 7 600 примеров
- 1 900 test-примеров на каждый класс

Для обучения нейросетевых моделей исходный train дополнительно разделяется на train и validation.

## Структура проекта

```text
nlp-text-classification/
├── notebooks/
│   ├── 01_nlp_baseline.ipynb
│   └── 02_transformer.ipynb
├── src/
├── tests/
├── README.md
└── .gitignore
```

`01_nlp_baseline.ipynb` содержит TF-IDF baseline и собственную PyTorch-модель.

`02_transformer.ipynb` содержит токенизацию и fine-tuning DistilBERT.

## Запуск

Установить зависимости:

```
pip install -r requirements.txt
```

Обучить DistilBERT:

```
python src/train.py
```

После обучения модель и tokenizer сохраняются локально в `models/distilbert-ag-news/`. Веса модели не хранятся в Git-репозитории.

Получить предсказание для нового текста:

```
python -c "from src.predict import predict; print(predict('Apple reports record quarterly revenue as iPhone sales grow worldwide.'))"
```

Пример результата:

```
{'class': 'Sci/Tech', 'confidence': 0.7717501521110535}
```

## Технологии

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- scikit-learn
- NumPy

## Основные выводы

Более сложная модель не гарантирует значительно более высокое качество.

TF-IDF + Logistic Regression достиг accuracy 91.5%, практически сравнявшись с нейросетевыми подходами.

Fine-tuned DistilBERT показал лучший результат — 92.0%, несмотря на обучение только на части исходного train-датасета.