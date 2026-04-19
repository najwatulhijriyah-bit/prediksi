# app/model.py

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.config import MODEL_NAME, DEVICE, id2label
import random

# ========================
# LOAD INDOBERT
# ========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

indobert = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=4,
    ignore_mismatched_sizes=True
).to(DEVICE)

indobert.eval()

# ========================
# PREDICT INDOBERT
# ========================
def predict_indobert(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():
        outputs = indobert(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    conf, pred = torch.max(probs, dim=1)

    return id2label[pred.item()], round(conf.item(), 4)

# ========================
# TEMP (GANTI MODEL ASLI)
# ========================
def predict_lstm(text):
    return random.choice(list(id2label.values())), round(random.uniform(0.6, 0.9), 4)

def predict_gru(text):
    return random.choice(list(id2label.values())), round(random.uniform(0.6, 0.9), 4)