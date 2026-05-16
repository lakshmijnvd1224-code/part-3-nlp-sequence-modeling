# Part 3 — NLP and Sequence Modeling Mini Project

## Overview
This project builds an NLP pipeline to classify customer support
messages into 3 sentiment categories: positive, negative, and neutral.
It compares traditional text vectorization with a sequence-based
deep learning approach using LSTM.

## Dataset
- **File:** customer_support_text_classification.csv
- **Source:** https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing
- **Records:** 1500
- **Target:** sentiment_label (positive, negative, neutral)
- **Note:** Dataset file is not uploaded to this repository. Please download from the source link above.

## Approach
1. Loaded and explored dataset — 1500 balanced customer support messages
2. Preprocessed text — lowercasing, removing special characters, tokenization, stopword removal
3. Vectorized text using TF-IDF (146 features)
4. Built baseline Logistic Regression model — achieved 100% accuracy
5. Built LSTM sequence model — achieved 65% accuracy
6. Compared both models and explained the performance difference

## Results
| Model | Accuracy |
|---|---|
| Logistic Regression + TF-IDF | 100% |
| LSTM Sequence Model | 65% |

## Observations
- Logistic Regression with TF-IDF achieved perfect 100% accuracy
- The dataset contains very distinct vocabulary per sentiment class
- LSTM achieved only 65% as it needs more data to learn sequence patterns
- Short text messages (average 12 words) favour traditional ML over LSTM
- Neutral class was hardest for LSTM to classify correctly
- TF-IDF is more suitable for short, consistent text classification tasks

## Task 6 : Attention and Transformer Reflection

**Why RNNs struggle with long-term dependencies:**
RNNs process text word by word and pass information forward through
a hidden state. For long sequences, the information from early words
gets diluted or lost by the time the model reaches the end. This is
called the vanishing gradient problem — gradients become too small
to update weights effectively for distant words.

**How LSTMs help with memory:**
LSTMs solve this by introducing three gates — input, forget, and output.
These gates control what information to keep, what to discard, and what
to pass forward. This allows LSTMs to selectively remember important
information from much earlier in the sequence, making them far better
than basic RNNs for longer text.

**What attention solves in sequence-to-sequence tasks:**
Even LSTMs struggle when sequences are very long. Attention mechanisms
allow the model to look back at all previous words at every step and
decide which ones are most relevant for the current prediction. Instead
of relying on a single compressed hidden state, attention gives the
model direct access to all parts of the input — like being able to
re-read any part of a sentence at any time.

**Why transformers are important in modern NLP and Generative AI:**
Transformers use attention entirely, without any recurrence. This means
they can process all words in parallel instead of one by one, making
them much faster to train on large datasets. Models like BERT, GPT,
and ChatGPT are all transformer-based. Transformers have revolutionised
NLP by enabling models to understand context across very long documents
and generate human-like text, making them the foundation of all modern
Generative AI systems.

## Repository Structure
part-3-nlp-sequence-modeling/
│
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── model_evaluation.png or .csv
    └── sample_predictions.txt

## Libraries Used
- TensorFlow — building and training the LSTM model
- scikit-learn — TF-IDF vectorization, Logistic Regression, evaluation metrics
- pandas — loading and handling the dataset
- numpy — numerical operations
- matplotlib — plotting accuracy and loss curves
- seaborn — confusion matrix heatmap
- nltk — text preprocessing, stopword removal, tokenization
