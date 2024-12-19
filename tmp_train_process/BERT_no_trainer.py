# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 20:27:50 2024

@author: Yiqian Zhang

Training the BertLMHeadModel model (no Trainer version)
"""
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertLMHeadModel, AutoTokenizer
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
from datasets import Dataset

## 1. define the dataset
class JokeDataset(Dataset):
  """
    define the dataset for the Generator
  
    This is a dataset class, including initialization, loading data from csv, 
    obtaining the length of the data and providing a method for returning one 
    item on the index.
  """
  def __init__(self):
    self.listOfJokes = self.loadJokes()

  # obtain jokes from the csv
  def loadJokes(self):
##    csvData = pd.read_csv('shortjokes.csv')
    csvData = pd.read_csv('dev-small.csv')
    data = []
    for index, row in csvData.iterrows():
##      data.append(row['Joke'])
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
# 1-1)obtain the first 1% of the original data as the dataset to be used (used for the dataset 'shortjokes.csv')
##small_size = int(len(dataSet) * 0.01)
##small_dataSet = dataSet[:small_size]
# 1-2)obtain the whole dataset (used for the dataset 'dev-small.csv')
small_dataSet = dataSet
# 2)split datasets
train_data_size = int(len(small_dataSet) * 0.8)
test_data_size = int(len(small_dataSet) * 0.1)
validation_data_size = len(small_dataSet) - train_data_size - test_data_size
train_dataSet, test_dataSet, validation_dataSet = random_split(
    dataset=small_dataSet, 
    lengths=[train_data_size, test_data_size, validation_data_size]
)

## 4. construct dataLoaders
batch_size = 4
train_loader = DataLoader(train_dataSet, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader = DataLoader(test_dataSet, batch_size=batch_size, shuffle=True, drop_last=False)
validation_loader = DataLoader(validation_dataSet, batch_size=batch_size, shuffle=True, drop_last=False)
print("======== Here are the details of the generator datasets. ========")
print("Train_dataset's size: ", len(train_dataSet))
print("Test_dataset's size: ", len(test_dataSet))
print("Validation_dataset's size: ", len(validation_dataSet))
print("Train_dataset's batch size: ", train_loader.batch_size)
print("Test_dataset's batch size: ", test_loader.batch_size)
print("Validation_dataset's batch size: ", validation_loader.batch_size)
print("The size of each data loader (train, test and validation): ", len(train_loader), len(test_loader), len(validation_loader))
print("======== Above are the details of the generator datasets. ========")

## 5. define the model
class BERTModel(torch.nn.Module):
  """
    define the Generator model
  
    This is a BERT model class in the form of Pytorch Neural Network Architecture. 
    1. __init__(): get the BertLMHeadModel from the Hugging Face.
    2. forward(): input a mapping into the model and return the word indices of prediction with loss.
    3. generate(): use the trained model to generate sequences.
  """
  def __init__(self):
    super(BERTModel, self).__init__()
    self.bert = BertLMHeadModel.from_pretrained("./bert-base-cased") # case sensitive
  
  def forward(self, input_ids, attention_mask, token_type_ids, label_ids):  
    bert_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=label_ids) # input a mapping into the bert()
    logits = bert_output.logits
    prediction = torch.argmax(logits, dim=-1) # convert the output to the word indices
    return bert_output.loss, prediction

  # using the trained model to generate sequences
  def generate(self, bert_input, attention_mask, max_length, num_return_sequences,
               no_repeat_ngram_size, repetition_penalty, do_sample, top_k, top_p,
               temperature, tokenizer):
    return self.bert.generate(input_ids=bert_input, # input
                              attention_mask=attention_mask, # attention mask
                              max_length=max_length, # maximum length of the generated sequences
                              num_return_sequences=num_return_sequences, # the number of returned sequences
                              no_repeat_ngram_size=no_repeat_ngram_size, # avoid repeated words/phrases of length equals no_repeat_ngram_size
                              repetition_penalty=repetition_penalty, # repetition penalty
                              do_sample=do_sample, # whether to use sampling strategy
                              top_k=top_k, # the number of tokens with the highest probability selected at each step
                              top_p=top_p, # accumulated probability threshold used for Top-p sampling
                              temperature=temperature, # temperature parameter to adjust randomness
                              bos_token_id=tokenizer.bos_token_id,
                              pad_token_id=tokenizer.pad_token_id,
                              eos_token_id=tokenizer.eos_token_id
                              )

## 6. set up the device for GPU usage
torch.cuda.empty_cache()
device = torch.device('cuda' if cuda.is_available() else 'cpu')
print("Using GPU or CPU? ", device)

## 7. restore the trained model / create a new model
PATH = './BERT_model.pth' # path
if os.path.exists(PATH):
    print("There exists such a model.")
    model = torch.load(PATH)
else:
    print("We need to create a model.")
    model = BERTModel()
    model.to(device)
##print("model: ", model)

## 8. define the optimizer and hyperparameters
learning_rate = 3e-4
sum_loss = 0.0
epochs = 3
optimizer = optim.AdamW(model.parameters(), lr=learning_rate) # optimizer
tokenizer = AutoTokenizer.from_pretrained("./bert-base-cased") # tokenizer

## 9. train the model 
def train(model, data_loader, optimizer, epochs):
  """
    train the Generator model
  
    Args:
        model (BERTModel): Generator model using BertLMHeadModel.
        data_loader (DataLoader): Training data loader for the Generator.
        optimizer (Optimizer): Optimizer using AdamW for the Generator.
        epochs (int): Training epochs for the Generator.
  """
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
        bert_input = tokenizer(inputs[j], # input text(need to get the string in the array)
                               max_length=128, # change from 512 to 128
                               padding=True,
                               truncation=True,
                               return_tensors="pt").to(device) # return PyTorch tensor
        input_ids = bert_input['input_ids']
        attention_mask = bert_input['attention_mask']
        token_type_ids = bert_input['token_type_ids']
        # 3)zero the parameter gradients
        optimizer.zero_grad()
        # 4)obtain the output and loss (here, labels = inputs)
        loss, outputs = model(input_ids, attention_mask, token_type_ids, input_ids) # equals to calling model.forward()
        # 5)compute the gradient
        loss.backward()
        # 6)perform single-step optimization
        optimizer.step()
        # 7)update the total gradient of the current batch
        sum_loss += loss

      # 8)print losses of each 200 batches
      if i % 200 == 199:
        print(f'[{epoch + 1}, {i + 1:5d}]    loss: {sum_loss / 200:.3f}')
        sum_loss = 0.0
      # 9)empty the cache of the cuda
      torch.cuda.empty_cache()
  print('Finished training.')
train(model, train_loader, optimizer, epochs)

## 10. save the model
torch.save(model, PATH)

## 11. evaluate the model
def evaluate(model, data_loader, epochs):
  """
    evaluate the Generator model
  
    Args:
        model (BERTModel): Generator model using BertLMHeadModel.
        data_loader (DataLoader): Validation data loader for the Generator.
        epochs (int): Evaluation epochs for the Generator.
  """
  print('Start evaluation.')
  model.eval() # open the evaluation mode
  for epoch in range(epochs): # traverse each epoch
    losses = [] # store each loss in each epoch
    
    for i, data in enumerate(data_loader, 0):
      # 1)obtain the inputs
      inputs = data
    
      for j in range(len(inputs)): # traverse each input
        with torch.no_grad():  
          # 2)here, each inputs represents a joke, we should construct an input that conforms to the BERT format
          bert_input = tokenizer(inputs[j], # input text(need to get the string in the array)
                                 max_length=128, # change from 512 to 128
                                 padding=True,
                                 truncation=True,
                                 return_tensors="pt").to(device) # return PyTorch tensor
          input_ids = bert_input['input_ids']
          attention_mask = bert_input['attention_mask']
          token_type_ids = bert_input['token_type_ids']
          # 3)get the output and loss
          loss, outputs = model(input_ids, attention_mask, token_type_ids, input_ids) # equals to calling model.forward()
          losses.append(loss.repeat(batch_size)) # repeat batch_size times to enable each input in the current batch to have the loss
    losses = torch.cat(losses) # in each input, concat each loss tensor into one tensor
    losses = losses[: len(validation_dataSet)] # guarantee the number of losses ​​corresponds to the number of samples in the evaluation dataset
    
    # 4)compute the perplexity
    try:
      perplexity = math.exp(torch.mean(losses))
    except OverflowError:
      perplexity = float("inf")
    print(f"Epoch {epoch+1}: {perplexity}")
  print('Finished evaluation.')
evaluate(model, validation_loader, epochs)  

## 12. define the text procedure
def generate_text(input_text):
  """
    generate some sentences after training and evaluating the Generator
  
    Args:
        input_text (str): Start words for generating sentences.
        
    Returns:
        jokes (list): A list storing generated sentences.
  """
  # 1)construct an input that conforms to the BERT format
  bert_input = tokenizer.encode(input_text, return_tensors="pt").to(device) # return PyTorch tensor
  # 2)construct the attention_mark(using torch.ones to generate a tensor with value 1)
  attention_mask = torch.ones(bert_input.shape, dtype=torch.long, device=bert_input.device)
  # 3)using Top-k sampling, Top-p sampling, temperature sampling(without beam-search decoding)
  with torch.no_grad():
    outputs = model.generate(bert_input, # input
                             attention_mask, # attention mask
                             max_length=40, # maximum length of the generated sequences
                             num_return_sequences=1, # the number of returned sequences
                             no_repeat_ngram_size=2, # avoid repeated words/phrases of length equals no_repeat_ngram_size
                             repetition_penalty=1.83, # repetition penalty
                             do_sample=True, # whether to use sampling strategy
                             top_k=50, # the number of tokens with the highest probability selected at each step
                             top_p=0.92, # accumulated probability threshold used for Top-p sampling
                             temperature=0.78, # temperature parameter to adjust randomness
                             tokenizer=tokenizer)
  # 4)decode each joke of the output and get the generated sentence
  jokes = []
  for output in outputs:
    jokes.append(tokenizer.decode(output, skip_special_tokens=True))
  return jokes

## 13. test the model
def test(text):
  """
    test the Generator model
  
    Args:
        text (str): Start words for generating sentences.
  """
  jokes = generate_text(text)
  for i, joke in enumerate(jokes, 0):
    print(f"Generated joke: {joke}")
test("If")
test("When")
test("What")
test("How")
test("Who")
test("Why")
test("Where")
