# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 20:27:50 2024

@author: Yiqian Zhang

Generator (trying to use the Trainer from the baseline model)
"""
import os
from transformers import BertTokenizer, BertLMHeadModel, BertForMaskedLM, TrainingArguments, Trainer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from torch import cuda, nn, optim
import numpy as np
import pandas as pd
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from collections import Counter
from tqdm import tqdm # process bar
import math
from datasets import Dataset
import random

## 1. create a dataset for 'dev-small.csv'
def get_data(file_path):
  df = pd.read_csv(file_path)
  df["text"] = df["text"].astype(str)
  return Dataset.from_pandas(df[["text"]])
dataSet = get_data('./dev-small.csv') 
print("dataSet: ", dataSet)

## 2.set up the device for GPU usage
torch.cuda.empty_cache()
device = torch.device('cuda' if cuda.is_available() else 'cpu')
print("Using GPU or CPU? ", device)

## 3.define the model and create some settings
# 1) GPT2
##tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
##model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
##tokenizer.pad_token = tokenizer.eos_token # by default, pad_token is not defined in the GPT2 model
# 2) BERT
tokenizer = BertTokenizer.from_pretrained("./bert-base-cased")
model = BertLMHeadModel.from_pretrained("./bert-base-cased")
tokenizer.eos_token = '[SEP]' # specify the eos_token
tokenizer.pad_token = '[SEP]' # specify the pad_token

## 4.tokenized the input
def tokenize_function(data, tokenizer):
  tokenized_output = tokenizer(data["text"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
  tokenized_output["labels"] = tokenized_output["input_ids"] # take input_ids as the labels
  return tokenized_output
tokenized_datasets = dataSet.map(lambda x: tokenize_function(x, tokenizer), batched=True) # traverse each data in the dataset to get tokenized datasets

## 5.defien the arguments for training
training_args = TrainingArguments(
##  output_dir="./GPT2_Model_Trainer", # store the outputs of training
  output_dir="./BERT_Model_Trainer", # store the outputs of training
  evaluation_strategy="epoch",
  learning_rate=0.005, #0.005 for BERT and 5e-5 for GPT2
  per_device_train_batch_size=4,
  per_device_eval_batch_size=4,
  num_train_epochs=3, # overall 3 epochs
  weight_decay=0.01, # the weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in AdamW optimizer
  logging_dir="./logs",
  save_steps=100, # the checkpoint save strategy to adopt during training
  logging_steps=10
)
##print("training_args: ", training_args)

## 7.define the trainer
trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=tokenized_datasets,
  eval_dataset=tokenized_datasets
)
##print("trainer: ", trainer)

## 8.begin to train the model
trainer.train()

## 9.save the model
# 1) GPT2
##model.save_pretrained("./GPT2_Model_Trainer")
##tokenizer.save_pretrained("./GPT2_Model_Trainer")
# 2) BERT
model.save_pretrained("./Bert_Model_Trainer")
tokenizer.save_pretrained("./Bert_Model_Trainer")

## 10.randomly choose the start words
start_words = ["What", "How", "Why", "When", "Who", "If", "Where", "Which", "Whose", "Whether"]
input_text = random.choice(start_words)

## 11.test the model
inputs = tokenizer(input_text, return_tensors="pt").to(device)
outputs = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=80, # maximum length of the generated sequences
    num_return_sequences=1, # the number of returned sequences
    no_repeat_ngram_size=2, # avoid repeated words/phrases of length equals no_repeat_ngram_size
    repetition_penalty=1.85, # repetition penalty
    top_p=0.92, # accumulated probability threshold used for Top-p sampling
    top_k=50, # the number of tokens with the highest probability selected at each step
    temperature=0.85, # temperature parameter to adjust randomness
    do_sample=True, # whether to use sampling strategy
    pad_token_id=tokenizer.pad_token_id
#    pad_token_id=tokenizer.encode(tokenizer.pad_token)[0]
)

## 12.decode generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated joke: ", generated_text)
