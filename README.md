🧠 Abstractive Text Summarization using T5/BART

This project fine-tunes a Transformer-based encoder-decoder model (T5 or BART) to perform abstractive text summarization — generating concise summaries of longer articles or documents.

📂 Project Overview

Goal: Train a summarization model that can understand long pieces of text and generate short, fluent summaries in natural language.

Architecture:
Encoder–Decoder models like T5 and BART are used because they can read a passage (encoder) and generate a new one (decoder).

🧰 Features

Preprocessing of custom CSV dataset (article → summary pairs)

Fine-tuning on a small sample or full dataset

Evaluation using ROUGE-1, ROUGE-2, and ROUGE-L

Interactive Gradio demo for testing summaries

Option to deploy via Kaggle, Colab, or Hugging Face Spaces

🗂️ Dataset

Dataset format:
Each CSV file (train, validation, test) should contain:

article, summary


Example:

"The economy is facing inflation and policy changes.", "The economy struggles with inflation."


Update the paths in the notebook:

TRAIN_CSV = "/kaggle/input/cnndata/train.csv"
VAL_CSV   = "/kaggle/input/cnndata/validation.csv"
TEST_CSV  = "/kaggle/input/cnndata/test.csv"

⚙️ Training Setup

Key parameters (editable in notebook):

MODEL_NAME = "t5-base" or "facebook/bart-base"

MAX_INPUT_LENGTH = 512

MAX_TARGET_LENGTH = 128

NUM_EPOCHS = 3

LR = 5e-5

The model is trained using Hugging Face Transformers and Seq2SeqTrainer.

📊 Evaluation Metrics

After training, the model is evaluated using:

ROUGE-1 (unigram overlap)

ROUGE-2 (bigram overlap)

ROUGE-L (longest common subsequence)

Qualitative results are also displayed for manual inspection.

💬 Gradio App

Run the Gradio app cell in the notebook to launch an interactive demo:

python app.py


or simply execute the Gradio cell.
It allows you to enter any paragraph and see the summarized output instantly.
or go to this link :https://6de5f7f3040f766edd.gradio.live/
🚀 Deliverables

Complete Jupyter Notebook with:

Preprocessing

Fine-tuning

Evaluation

Gradio demo

Model saved in /models/t5_summarization/

Evaluation report including ROUGE scores and examples

(Optional) Deployed demo via Gradio or Hugging Face Spaces

(Optional) Blog post describing dataset, model, metrics, and results
