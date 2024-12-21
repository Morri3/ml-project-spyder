# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 19:39:47 2024

@author: Yiqian Zhang

Generator and Detector
"""
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertForSequenceClassification, BertTokenizer
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
import time
import matplotlib.pyplot as plt # plot graphs

## 0. start time of the whole process
start_time = time.time()

## 1. define the tokenizer for the generator
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

## 2-1. define the dataset for the generator
class GeneratorDataset(Dataset):
  """
    define the dataset for the generator
  
    This is a dataset class, including initialization, loading data from csv, 
    obtaining the length of the data and providing a method for returning one 
    item on the index.
  """
  def __init__(self):
    self.listOfJokes = self.loadJokes()

  # obtain jokes from the csv
  def loadJokes(self):
    csvData = pd.read_csv('shortjokes.csv')
    joke=csvData["Joke"].astype(str).tolist()
    return Dataset.from_dict({"text": joke})

  # obtain the overall number of jokes
  def __len__(self):
    return len(self.listOfJokes) # overall number of jokes
  
  # obtain the joke corresponding to the index
  def __getitem__(self, index):
    return self.listOfJokes[index] # return the joke

## 2-2. define the dataset for the detector
class DetectorDataset(Dataset):
  """
    define the dataset for the detector

    This is a dataset class, including initialization, loading data from csv, 
    obtaining the length of the data and providing a method for returning one 
    item on the index.
  """
  def __init__(self):
    self.listOfJokes = self.loadJokes()

  # obtain jokes from the csv
  def loadJokes(self):
    csvData = pd.read_csv('dev-small.csv')
    text=csvData["text"].astype(str).tolist() # convert to a string list
    humor=csvData["humor"].astype(int).tolist() # convert to a integer list (instead of bool)
    return Dataset.from_dict({"text": text, "label": humor})

  # obtain the overall number of jokes
  def __len__(self):
    return len(self.listOfJokes) # overall number of jokes
  
  # obtain the joke corresponding to the index
  def __getitem__(self, index):
    return self.listOfJokes[index] # return the joke

## 2-3. construct datasets for the generator and detector
dataSet = GeneratorDataset()
print("Generator dataSet's basic info: ", dataSet.listOfJokes)
detector_dataSet = DetectorDataset()
print("Detector dataSet's basic info: ", detector_dataSet.listOfJokes)

## 3-1. get the train_dataset, test_dataset and validation_dataset (generator)
# 1)obtain the first 2% of the original data as the dataset to be used (used for the dataset 'shortjokes.csv')
small_size = int(len(dataSet) * 0.02)
small_dataSet = dataSet[:small_size]
# 2)split datasets randomly (using 'random_split' method)
train_data_size = int(len(small_dataSet['text']) * 0.8)
test_data_size = int(len(small_dataSet['text']) * 0.1)
validation_data_size = len(small_dataSet['text']) - train_data_size - test_data_size
train_dataSet, test_dataSet, validation_dataSet = random_split(
    dataset=small_dataSet['text'], 
    lengths=[train_data_size, test_data_size, validation_data_size]
)

## 3-2. construct dataLoaders
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

## 4-1. get the train_dataset, test_dataset and validation_dataset (detector)
# 1)split datasets randomly (using 'random_split' method)
train_data_detector_size = int(len(detector_dataSet) * 0.8)
test_data_detector_size = int(len(detector_dataSet) * 0.1)
validation_data_detector_size = len(detector_dataSet) - train_data_detector_size - test_data_detector_size
train_dataSet_detector, test_dataSet_detector, validation_dataSet_detector = random_split(
    dataset=detector_dataSet, 
    lengths=[train_data_detector_size, test_data_detector_size, validation_data_detector_size]
)

## 4-2. construct dataLoaders
batch_size_detector = 4
train_loader_detector = DataLoader(train_dataSet_detector, batch_size=batch_size_detector, shuffle=True, drop_last=False)
test_loader_detector = DataLoader(test_dataSet_detector, batch_size=batch_size_detector, shuffle=True, drop_last=False)
validation_loader_detector = DataLoader(validation_dataSet_detector, batch_size=batch_size_detector, shuffle=True, drop_last=False)
print("======== Here are the details of the detector datasets. ========")
print("Train_dataset's size: ", len(train_dataSet_detector))
print("Test_dataset's size: ", len(test_dataSet_detector))
print("Validation_dataset's size: ", len(validation_dataSet_detector))
print("Train_dataset's batch size: ", train_loader_detector.batch_size)
print("Test_dataset's batch size: ", test_loader_detector.batch_size)
print("Validation_dataset's batch size: ", validation_loader_detector.batch_size)
print("The size of each data loader (train, test and validation): ", len(train_loader_detector), len(test_loader_detector), len(validation_loader_detector))
print("======== Above are the details of the detector datasets. ========")

## 5. define the generator model
class GPT2Model(torch.nn.Module):
  """
    define the Generator model
  
    This is a GPT2 model class in the form of Pytorch Neural Network Architecture. 
    1. __init__(): get the GPT2LMHeadModel from the Hugging Face.
    2. forward(): input a mapping into the model and return the word indices of prediction with loss.
    3. generate(): use the trained model to generate sequences.
  """
  def __init__(self):
    super(GPT2Model, self).__init__()
    self.gpt2 = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token # by default, pad_token is not defined in the GPT2 model
    
  def forward(self, **tokenized_input):  
    gpt2_output = self.gpt2(**tokenized_input) # input a mapping into the gpt2()
    logits = gpt2_output.logits 
    prediction = torch.argmax(logits, dim=-1) # convert the output to the word indices
    return gpt2_output.loss, prediction

  def generate(self, gpt2_input, attention_mask, max_length, num_return_sequences,
               no_repeat_ngram_size, repetition_penalty, do_sample, top_k, top_p,
               temperature, tokenizer):
    return self.gpt2.generate(input_ids=gpt2_input, # input
                              attention_mask=attention_mask, # attention mask
                              max_length=max_length, # maximum length of the generated sequences
                              num_return_sequences=num_return_sequences, # the number of returned sequences
                              no_repeat_ngram_size=no_repeat_ngram_size, # avoid repeated words/phrases of length equals no_repeat_ngram_size
#                              repetition_penalty=repetition_penalty, # repetition penalty
                              do_sample=do_sample, # whether to use sampling strategy
                              top_k=top_k, # the number of tokens with the highest probability selected at each step
                              top_p=top_p, # accumulated probability threshold used for Top-p sampling
                              temperature=temperature, # temperature parameter to adjust randomness
                              pad_token_id=tokenizer.pad_token_id,
                              )

## 6. set up the device for GPU usage
torch.cuda.empty_cache()
device = torch.device('cuda' if cuda.is_available() else 'cpu')
print("Using GPU or CPU? ", device)

## 7. restore the trained model / create a new model
PATH = './GENERATOR_model.pth' # path
if os.path.exists(PATH):
    print("There exists the generator model.")
    model = torch.load(PATH)
else:
    print("We need to create a generator model.")
    model = GPT2Model()
    model.to(device)
##print("generator model: ", model)

## 8. define the optimizer and hyperparameters
learning_rate = 5e-5
epochs = 3
optimizer = optim.AdamW(model.parameters(), lr=learning_rate) # optimizer

## 9. train the model 
def train(model, data_loader, optimizer, epochs):
  """
    train the Generator model
  
    Args:
        model (GPT2Model): Generator model using GPT2LMHeadModel.
        data_loader (DataLoader): Training data loader for the Generator.
        optimizer (Optimizer): Optimizer using AdamW for the Generator.
        epochs (int): Training epochs for the Generator.
        
    Returns:
        train_generator_losses (list): Stored losses during training the Generator.
  """
  print('Start training the generator.')
  model.train() # open the training mode
  train_generator_losses = [] # store the losses during training the generator
  for epoch in range(epochs): # traverse each epoch
    sum_loss = 0.0
    print("Epoch {} started.".format(epoch+1))
    process = tqdm(data_loader, desc=f"Epoch {epoch+1} / {epochs}", ncols=100) # process bar

    for i, data in enumerate(process, 0):
      # 1)obtain the inputs
      inputs = data
      
      for j in range(len(inputs)): # traverse each input
        # 2)here, each inputs represaents a joke, we should construct an input that conforms to the GPT2 format
        tokenizer.pad_token = tokenizer.eos_token # by default, pad_token is not defined in the GPT2 model
        tokenized_input = tokenizer(inputs[j], padding="max_length", truncation=True, max_length=128, return_tensors='pt').to(device) # use GPT2 tokenizer
        tokenized_input["labels"] = tokenized_input["input_ids"] # there is no label in the 'shortjokes.csv', so we take the input_ids as the labels
        # 3)zero the parameter gradients
        optimizer.zero_grad()
        # 4)obtain the output and loss (here, labels = inputs)
        loss, outputs = model(**tokenized_input) # equals to calling model.forward() 
        # 5)compute the gradient
        loss.backward()
        # 6)perform single-step optimization
        optimizer.step()
        # 7)update the total gradient of the current batch
        sum_loss += loss
        train_generator_losses.append(loss.cpu().detach()) # move to the CPU and separate gradients

      # 8)print losses of each 200 batches
      if i % 200 == 199:
        print(f'[{epoch + 1}, {i + 1:5d}]    loss: {sum_loss / 200:.3f}')
        sum_loss = 0.0
      # 9)empty the cache of the cuda
      torch.cuda.empty_cache()
  print('Finished training the generator.')
  return train_generator_losses
train_generator_losses = train(model, train_loader, optimizer, epochs)
##print("train_generator_losses: ", len(train_generator_losses))

## 10. save the generator model
torch.save(model, PATH)

## 11. evaluate the model
def evaluate(model, data_loader, epochs):
  """
    evaluate the Generator model
  
    Args:
        model (GPT2Model): Generator model using GPT2LMHeadModel.
        data_loader (DataLoader): Validation data loader for the Generator.
        epochs (int): Evaluation epochs for the Generator.
        
    Returns:
        evaluate_generator_losses (list): Stored losses during evaluating the Generator.
  """
  print('Start evaluating the generator.')
  model.eval() # open the evaluation mode
  evaluate_generator_losses = [] # store the losses during evaluating the generator
  for epoch in range(epochs): # traverse each epoch
    losses = [] # store each loss in each epoch
    
    for i, data in enumerate(data_loader, 0):
      # 1)obtain the inputs
      inputs = data
      
      for j in range(len(inputs)): # traverse each input
        with torch.no_grad():  
          # 2)here, each inputs represents a joke, we should construct an input that conforms to the GPT2 format
          tokenizer.pad_token = tokenizer.eos_token # by default, pad_token is not defined in the GPT2 model
          tokenized_input = tokenizer(inputs[j], padding="max_length", truncation=True, max_length=128, return_tensors='pt').to(device) # use GPT2 tokenizer
          tokenized_input["labels"] = tokenized_input["input_ids"] # there is no label in the 'shortjokes.csv', so we take the input_ids as the labels
          # 3)get the output and loss
          loss, outputs = model(**tokenized_input) # equals to calling model.forward() 
          losses.append(loss.repeat(batch_size)) # repeat batch_size times to enable each input in the current batch to have the loss
          evaluate_generator_losses.append(loss.cpu()) # move to the CPU
    losses = torch.cat(losses) # in each input, concat each loss tensor into one tensor
    losses = losses[: len(validation_dataSet)] # guarantee the number of losses ​​corresponds to the number of samples in the evaluation dataset
    
    # 4)compute the perplexity
    try:
      perplexity = math.exp(torch.mean(losses))
    except OverflowError:
      perplexity = float("inf")
    print(f"Epoch {epoch+1}: {perplexity}")
  print('Finished evaluating the generator.')
  return evaluate_generator_losses
evaluate_generator_losses = evaluate(model, validation_loader, epochs)  
##print("evaluate_generator_losses: ", len(evaluate_generator_losses))

## 12. define the text procedure
def generate_text(input_text):
  """
    generate some sentences after training and evaluating the Generator
  
    Args:
        input_text (str): Start words for generating sentences.
        
    Returns:
        jokes (list): A list storing generated sentences.
  """
  # 1.construct an input that conforms to the GPT2 format
  gpt2_input = tokenizer.encode(input_text, return_tensors="pt").to(device) # return PyTorch tensor
  # 2.construct the attention_mark(using torch.ones to generate a tensor with value 1)
  attention_mask = torch.ones(gpt2_input.shape, dtype=torch.long, device=gpt2_input.device)
  # 3.using Top-k sampling, Top-p sampling, temperature sampling(without beam-search decoding)
  with torch.no_grad():
    outputs = model.generate(gpt2_input, # input
                             attention_mask, # attention mask
                             max_length=80, # maximum length of the generated sequences
                             num_return_sequences=1, # the number of returned sequences
                             no_repeat_ngram_size=2, # avoid repeated words/phrases of length equals no_repeat_ngram_size
                             repetition_penalty=1.83, # repetition penalty (not used)
                             do_sample=True, # whether to use sampling strategy
                             top_k=50, # the number of tokens with the highest probability selected at each step
                             top_p=0.92, # accumulated probability threshold used for Top-p sampling
                             temperature=0.7, # temperature parameter to adjust randomness
                             tokenizer=tokenizer)
  # 4.decode each joke of the output and get the generated sentence
  jokes = []
  for output in outputs:
    jokes.append(tokenizer.decode(output, skip_special_tokens=True))
  return jokes

## 13. test the model
def test_generator(text, generated):
  """
    test the Generator model
  
    Args:
        text (str): Start words for generating sentences.
        generated (list): A list to store generated sentences.
        
    Returns:
        generated (list): A list storing generated sentences.
  """
  jokes = generate_text(text)
  for i, joke in enumerate(jokes, 0):
    print(f"Generated joke: {joke}")
    generated.append(joke)
  return generated
generated = [] # store each generated joke in a list
test_generator("If", generated)
test_generator("When", generated)
test_generator("What", generated)
test_generator("How", generated)
test_generator("Who", generated)
test_generator("Why", generated)
test_generator("Where", generated)

## 14. verify whether the generated sentences are jokes (using BertForSequenceClassification model)
# 1) define the tokenizer for the detector
detector_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 2) define the detector model
class DetectorModel(torch.nn.Module):
  """
    define the Detector model
  
    This is a BERT model class in the form of Pytorch Neural Network Architecture. 
    1. __init__(): get the BertForSequenceClassification from the Hugging Face.
    2. forward(): input a mapping into the model and return the word indices of prediction with loss.
    3. detect(): as same as the forward(), but there is no need to input the labels here.
  """
  def __init__(self):
    super(DetectorModel, self).__init__()
    self.detector = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    
  def forward(self, input_ids, attention_mask, token_type_ids, label_ids): # with inputting the labels
    detector_output = self.detector(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=label_ids) # input a mapping into the detector()
    logits = detector_output.logits
    prediction = torch.argmax(logits, dim=-1) # convert the output to the word indices
    return detector_output.loss, prediction

  def detect(self, **tokenized_input): # here, we don't input the labels
    detector_output = self.detector(**tokenized_input) # input a mapping into the detector()
    logits = detector_output.logits
    prediction = torch.argmax(logits, dim=-1) # convert the output to the word indices
    return prediction

# 3) empty the cache of the cuda
torch.cuda.empty_cache()

# 4) restore the trained model / create a new model
DETECTOR_PATH = './DETECTOR_model.pth' # path
if os.path.exists(DETECTOR_PATH):
    print("There exists the detector model.")
    detector_model = torch.load(DETECTOR_PATH)
else:
    print("We need to create a detector model.")
    detector_model = DetectorModel()
    detector_model.to(device)
##print("detector_model: ", detector_model)

# 5) define hyperparameters
learning_rate_detector = 1e-4
epochs_detector = 5
criterion = nn.CrossEntropyLoss() # loss function
optimizer_detector = optim.AdamW(detector_model.parameters(), lr=learning_rate_detector) # optimizer

# 6) train the detector model
def train_detector(model, tokenizer, data_loader, optimizer, epochs):
  """
    train the Detector model
  
    Args:
        model (DetectorModel): Detector model using BertForSequenceClassification.
        tokenizer (Tokenizer): The tokenizer based on WordPiece.
        data_loader (DataLoader): Training data loader for the Detector.
        optimizer (Optimizer): Optimizer using AdamW for the Detector.
        epochs (int): Training epochs for the Detector.
        
    Returns:
        train_detector_losses (list): Stored losses during training the Detector.
  """
  print('Start training the detector.')
  model.train() # open the training mode
  train_detector_losses = [] # store the losses during training the detector
  for epoch in range(epochs): # traverse each epoch
    sum_loss = 0.0
    print("Epoch {} started.".format(epoch+1))
    process = tqdm(data_loader, desc=f"Epoch {epoch+1} / {epochs}", ncols=100) # process bar

    for i, data in enumerate(process, 0):
      # 1)obtain the inputs
      inputs = data
      
      for j in range(len(inputs['text'])): # traverse each input (here, inputs have two elements 'text' and 'label', which are different from the generator's dataset)
        # 2)here, we should construct an input that conforms to the BERT format
        detector_input = tokenizer(inputs['text'][j], # input text(need to get the string in the array)
                                   max_length=128, # change from 512 to 128
                                   padding=True,
                                   truncation=True,
                                   return_tensors="pt").to(device) # return PyTorch tensor
        input_ids = detector_input['input_ids']
        attention_mask = detector_input['attention_mask']
        token_type_ids = detector_input['token_type_ids']
        label_ids = inputs['label'][j].unsqueeze(0).to(device) # use unsqueeze() to convert a scalar tensor into a tensor with shape (batch_size,)
        # 3)zero the parameter gradients
        optimizer.zero_grad()
        # 4)obtain the output and loss
        loss, outputs = model(input_ids, attention_mask, token_type_ids, label_ids) # equals to calling model.forward()
        # 5)compute the gradient
        loss.backward()
        # 6)perform single-step optimization
        optimizer.step()
        # 7)update the total gradient of the current batch
        sum_loss += loss
        train_detector_losses.append(loss.cpu().detach()) # move to the CPU and separate gradients
      
      # 8)print losses of each 200 batches
      if i % 200 == 199:
        print(f'[{epoch + 1}, {i + 1:5d}]    loss: {sum_loss / 200:.3f}')
        sum_loss = 0.0
      # 9)empty the cache of the cuda
      torch.cuda.empty_cache()
  print('Finished training the detector.')
  return train_detector_losses
train_detector_losses = train_detector(detector_model, detector_tokenizer, train_loader_detector, optimizer_detector, epochs_detector)
##print("train_detector_losses: ", len(train_detector_losses))

# 7) save the detector model
torch.save(detector_model, DETECTOR_PATH)

# 8) evaluate the detector model
def evaluate_detector(model, tokenizer, data_loader, epochs):
  """
    evaluate the Detector model
  
    Args:
        model (DetectorModel): Detector model using BertForSequenceClassification.
        tokenizer (Tokenizer): The tokenizer based on WordPiece.
        data_loader (DataLoader): Validation data loader for the Detector.
        epochs (int): Evaluation epochs for the Detector.
        
    Returns:
        evaluate_detector_losses (list): Stored losses during evaluating the Detector.
  """
  print('Start evaluating the detector.')
  model.eval() # open the evaluation mode
  evaluate_detector_losses = [] # store the losses during evaluating the detector
  for epoch in range(epochs): # traverse each epoch
    losses = [] # store each loss in each epoch
    
    for i, data in enumerate(data_loader, 0):
      # 1)obtain the inputs
      inputs = data
      
      for j in range(len(inputs['text'])): # traverse each input (here, inputs have two elements 'text' and 'label', which are different from the generator's dataset)
        with torch.no_grad():  
          # 2)here, we should construct an input that conforms to the BERT format
          detector_input = tokenizer(inputs['text'][j], # input text(need to get the string in the array)
                                     max_length=128, # change from 512 to 128
                                     padding=True,
                                     truncation=True,
                                     return_tensors="pt").to(device) # return PyTorch tensor
          input_ids = detector_input['input_ids']
          attention_mask = detector_input['attention_mask']
          token_type_ids = detector_input['token_type_ids']
          label_ids = inputs['label'][j].unsqueeze(0).to(device) # use unsqueeze() to convert a scalar tensor into a tensor with shape (batch_size,)
          # 3)get the output and loss
          loss, outputs = model(input_ids, attention_mask, token_type_ids, label_ids) # equals to calling model.forward()
          losses.append(loss.repeat(batch_size_detector)) # repeat batch_size times to enable each input in the current batch to have the loss
          evaluate_detector_losses.append(loss.cpu()) # move to the CPU
    losses = torch.cat(losses) # in each input, concat each loss tensor into one tensor
    losses = losses[: len(validation_dataSet_detector)] # guarantee the number of losses ​​corresponds to the number of samples in the evaluation dataset
    
    # 4)compute the perplexity
    try:
      perplexity = math.exp(torch.mean(losses))
    except OverflowError:
      perplexity = float("inf")
    print(f"Epoch {epoch+1}: {perplexity}")
  print('Finished evaluating the detector.')
  return evaluate_detector_losses
evaluate_detector_losses = evaluate_detector(detector_model, detector_tokenizer, validation_loader_detector, epochs_detector) 
##print("evaluate_detector_losses: ", len(evaluate_detector_losses))

# 9) verify generated jokes
def verify_jokes(model, tokenizer, texts):
  """
    verify generated sentences whether they are jokes
  
    Args:
        model (DetectorModel): Detector model using BertForSequenceClassification.
        tokenizer (Tokenizer): The tokenizer based on WordPiece.
        texts (list): A list storing generated sentences.
  """
  flag = False # check whether there are jokes in the generated sentences
  for text in texts:
    detector_input = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    outputs = model.detect(**detector_input)
    if outputs.item() == 1: # we only output verified jokes
      flag = True
      print("Joke: ", text, "\nIs it a joke? ", outputs.item() == 1)
  if flag == False: # output the tip
    print("There is no joke in the generated sentences.")
verify_jokes(detector_model, detector_tokenizer, generated)

## 15. show images of losses for the generator and detector (pyplot-style)
def show_losses(title, epochs, train_dataset_size, validation_dataset_size, train_losses, evaluate_losses):
  """
    show images of losses for the generator and detector (pyplot-style)
  
    Args:
        title (str): The title of the graph.
        epochs (int): Training or Evaluation epochs for the Generator or Detector.
        train_dataset_size (int): The size of the training dataset.
        validation_dataset_size (int): The size of the validation dataset.
        train_losses (list): A list storing losses during training of the Generator or Detector.
        evaluate_losses (list): A list storing losses during evaluation of the Generator or Detector.
  """
  plt.figure(figsize=(6, 3), layout='constrained') # a figure with a grid of Axes
  plt.plot(range(epochs*train_dataset_size), train_losses, label="Training Loss", color="red") # train
  plt.plot(range(epochs*validation_dataset_size), evaluate_losses, label="Evaluation Loss", color="blue") # validation
  plt.xlabel("Number of data") # xlabel
  plt.ylabel("Losses") # ylabel
  plt.title(title) # title
  plt.legend(['Training', 'Evaluating'], loc='upper right') # legend
  # save graphs in a relative path
  SAVE_DIR_PATH = './Images'
  if not os.path.exists(SAVE_DIR_PATH): # create the path if it doesn't exist
    os.makedirs(SAVE_DIR_PATH)
  file_name = SAVE_DIR_PATH + '/' + title + '.png'
  plt.savefig(file_name, dpi=300) # save graphs
  # show graphs
  plt.show()
show_losses( # generator
    "Generator's training and evaluation results", epochs, 
    len(train_dataSet), len(validation_dataSet),
    train_generator_losses, evaluate_generator_losses
)
show_losses( # detector
    "Detector's training and evaluation results", epochs_detector, 
    len(train_dataSet_detector), len(validation_dataSet_detector),
    train_detector_losses, evaluate_detector_losses
)

## 16. obtain the end time of the whole process and output it
end_time = time.time()
consumed_time = end_time - start_time
print(f"The whole process consumed: {consumed_time: .2f} seconds.")
