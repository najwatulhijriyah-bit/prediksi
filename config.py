# app/config.py

import torch

MODEL_NAME = "indobenchmark/indobert-base-p1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

id2label = {
    0: "non-toxic",
    1: "hate speech",
    2: "insult",
    3: "threat"
}