# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 19:39:47 2024

@author: Yiqian Zhang

*Training the GPT2 model (my version)*
"""
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from torch import cuda, nn, optim
import numpy as np
import pandas as pd
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from collections import Counter
from tqdm import tqdm # process bar
import os
import math
from datasets import Dataset # used for the dataset 'dev-small.csv'

## 1. tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

## 2. define the dataset
class JokeDataset(Dataset):
  def __init__(self):
    self.listOfJokes = self.loadJokes()

  # obtain jokes from the csv
  def loadJokes(self):
#    csvData = pd.read_csv('shortjokes.csv')
    csvData = pd.read_csv('dev-small.csv')
    data=[]
    for index, row in csvData.iterrows():
#      data.append(row['Joke'])
      data.append(row['text'])
    return data

  # obtain the overall number of jokes
  def __len__(self):
    return len(self.listOfJokes) # overall number of jokes
  
  # obtain the joke corresponding to the index
  def __getitem__(self, index):
    return self.listOfJokes[index] # return the joke

## 2. construct datasets
dataSet = JokeDataset()

## 3. get the train_dataset, test_dataset and validation_dataset
# 1)obtain the first 5% of the original data as the dataset to be used (used for the dataset 'shortjokes.csv')
###small_size=int(len(dataSet) * 0.05)
###print("smaller dataset size: ", small_size)
###small_dataSet = dataSet[:small_size]
# 1)obtain the whole dataset (used for the dataset 'dev-small.csv')
small_dataSet=dataSet
# 2)split datasets
train_data_size = int(len(small_dataSet) * 0.8)
test_data_size = int(len(small_dataSet) * 0.1)
validation_data_size = len(small_dataSet) - train_data_size - test_data_size
train_dataSet, test_dataSet, validation_dataSet = random_split(dataset=small_dataSet, 
                                                               lengths=[train_data_size, test_data_size, validation_data_size])

## 4. construct dataLoaders
batch_size = 4
train_loader = DataLoader(train_dataSet, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader = DataLoader(test_dataSet, batch_size=batch_size, shuffle=True, drop_last=False)
validation_loader = DataLoader(validation_dataSet, batch_size=batch_size, shuffle=True, drop_last=False)
print("train_dataset's size: ", len(train_dataSet))
print("test_dataset's size: ", len(test_dataSet))
print("validation_dataset's size: ", len(validation_dataSet))
print("train_dataset's batch size: ",train_loader.batch_size)
print("test_dataset's batch size: ",test_loader.batch_size)
print("validation_dataset's batch size: ",validation_loader.batch_size)
print("the size of each batch in three datasets(train, test and validation): ",len(train_loader), len(test_loader), len(validation_loader))

## 5.define the model
class GPT2Model(torch.nn.Module):
  def __init__(self):
    super(GPT2Model, self).__init__()
    self.gpt2 = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
  def forward(self, **tokenized_input):  
    gpt2_output=self.gpt2(**tokenized_input) # input a mapping into the gpt2()
    logits=gpt2_output.logits # [batch_size, sequence_length, config.vocab_size]    
    prediction = torch.argmax(logits, dim=-1) # convert the output to the word indices
    return gpt2_output.loss, prediction

  # using the trained model to generate sequences
  def generate(self, gpt2_input, attention_mask, max_length, num_return_sequences,
               no_repeat_ngram_size, repetition_penalty, do_sample, top_k, top_p,
               temperature, tokenizer):
    return self.gpt2.generate(input_ids=gpt2_input, # input
                              attention_mask=attention_mask, # attention mask
                              max_length=max_length, # maximum length of the generated sequences
                              num_return_sequences=num_return_sequences, # the number of returned sequences
                              no_repeat_ngram_size=no_repeat_ngram_size, #避免重复出现长度=no_repeat_ngram_size的单词/词组
#                              repetition_penalty=repetition_penalty, # repetition penalty
                              do_sample=do_sample, #是否启用采样策略，选择下一个token
                              top_k=top_k, #每步选择概率最高的token数
                              top_p=top_p, #进行Top-p sampling使用的累积概率阈值
                              temperature=temperature, #调节随机性的温度参数
                              pad_token_id=tokenizer.pad_token_id,
                              )

## 6.set up the device for GPU usage
torch.cuda.empty_cache()
device = torch.device('cuda' if cuda.is_available() else 'cpu')
print("Using GPU or CPU? ", device)

## 7.restore the trained model / create a new model
PATH = './GPT2_model.pth' # path
if os.path.exists(PATH):
    print("There exists such a model.")
    model = torch.load(PATH)
else:
    print("We need to create a model.")
    model = GPT2Model()
    model.to(device)
##print("model: ",model)

## 8.define the optimizer and hyperparameters
learning_rate = 5e-5
sum_loss = 0.0
epochs = 3
optimizer = optim.AdamW(model.parameters(), lr=learning_rate) # optimizer

## 9.train the model 
def train(model, data_loader, optimizer, epochs):
  print('Start training.')
  model.train() # open the training mode
  for epoch in range(epochs): # traverse each epoch
    sum_loss = 0.0
    print("Epoch {} started.".format(epoch+1))
    process = tqdm(data_loader, desc=f"Epoch {epoch+1} / {epochs}", ncols=100) # process bar

    for i, data in enumerate(process, 0):
      # 1)obtain the inputs
      inputs = data

      for j in range(len(inputs)): # traverse each input
        # 2)here, each inputs represents a joke, we should construct an input that conforms to the BERT format
        tokenized_input = tokenizer(inputs[j], padding="max_length", truncation=True, max_length=128, return_tensors='pt').to(device)
        tokenized_input["labels"] = tokenized_input["input_ids"]
        # 3)zero the parameter gradients
        optimizer.zero_grad()
        # 4)obtain the output and loss (here, labels = inputs)
        #  【model(inputs) should be input that conforms to the BERT input format】
        loss, outputs = model(**tokenized_input) # equals to calling model.forward() 
        # 5)compute the gradient
        loss.backward()
        # 6)perform single-step optimization
        optimizer.step()
        # 7)update the total gradient of the current batch
        sum_loss += loss

      # print every 200 mini-batches
      if i % 200 == 199:
        print(f'[{epoch + 1}, {i + 1:5d}]    loss: {sum_loss / 200:.3f}')
        sum_loss = 0.0
      # empty the cache of the cuda
      torch.cuda.empty_cache()
  print('Finished training.')
train(model, train_loader, optimizer, epochs)

## 10.save the model
torch.save(model, PATH)

## 11.evaluate the model
def evaluate(model, data_loader, epochs):
  print('Start evaluation.')
  model.eval() # open the evaluation mode
  for epoch in range(epochs): # traverse each epoch
    losses = [] # store each loss during the evaluation
    for i, data in enumerate(data_loader, 0):
      inputs = data
      for j in range(len(inputs)): # traverse each input
        with torch.no_grad():  
          # 1)here, each inputs represents a joke, we should construct an input that conforms to the BERT format
          tokenized_input = tokenizer(inputs[j], padding="max_length", truncation=True, max_length=128, return_tensors='pt').to(device)
          tokenized_input["labels"] = tokenized_input["input_ids"]
          # 2)get the output and loss
          loss, outputs = model(**tokenized_input) # equals to calling model.forward() 
          losses.append(loss.repeat(batch_size))
    losses = torch.cat(losses) # concat each loss tensor into one tensor
    losses = losses[: len(validation_dataSet)]
    # 3)compute the perplexity
    try:
      perplexity = math.exp(torch.mean(losses))
    except OverflowError:
      perplexity = float("inf")
    print(f"Epoch {epoch+1}: {perplexity}")
  print('Finished evaluation.')
evaluate(model, validation_loader, epochs)  

## 12.define the text procedure
def generateText(input_text):
  # 1.construct an input that conforms to the BERT format
  gpt2_input=tokenizer.encode(input_text, return_tensors="pt").to(device) # return PyTorch tensor
  # 2.construct the attention_mark(using torch.ones to generate a tensor with value 1)
  attention_mask=torch.ones(gpt2_input.shape, dtype=torch.long, device=gpt2_input.device)
  # 3.using Top-k sampling, Top-p sampling, temperature sampling(without beam-search decoding)
  with torch.no_grad():
    outputs=model.generate(gpt2_input, # input
                           attention_mask, # attention mask
                           max_length=80, # maximum length of the generated sequences
                           num_return_sequences=1, # the number of returned sequences
                           no_repeat_ngram_size=2, #避免重复出现长度=no_repeat_ngram_size的单词/词组
                           repetition_penalty=1.83, # repetition penalty (not used)
                           do_sample=True, #是否启用采样策略，选择下一个token
                           top_k=50, #每步选择概率最高的token数
                           top_p=0.92, #进行Top-p sampling使用的累积概率阈值
                           temperature=0.7, #调节随机性的温度参数
                           tokenizer=tokenizer)
  # 4.decode each joke of the output and get the generated sentence (default: 1)
  jokes = []
  for output in outputs:
    jokes.append(tokenizer.decode(output, skip_special_tokens=True))
  return jokes

## 13. test the model
def test(text):
  jokes=generateText(text)
  for i, joke in enumerate(jokes, 0):
    print(f"Generated joke: {joke}")
test("If")
test("When")
test("What")
test("How")
test("Who")
test("Why")
test("Where")
