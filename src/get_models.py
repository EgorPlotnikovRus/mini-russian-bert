import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoModelForMaskedLM, BertConfig, BertForMaskedLM


def get_model(params, tokenizer):
    student_config = BertConfig(vocab_size=tokenizer.vocab_size, **params)
    student = BertForMaskedLM(student_config)

    return student