# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 20:27:50 2024

@author: Yiqian Zhang
"""
from transformers import BertTokenizer, BertModel, BertLMHeadModel, BertForMaskedLM
import torch
from torch import cuda, nn, optim
import numpy as np
import pandas as pd
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from collections import Counter
from tqdm import tqdm # process bar

## 1. define the dataset
class JokeDataset(Dataset):
  def __init__(self):
    self.listOfJokes = self.loadJokes()
    self.listOfWords = self.loadWords()
    self.listOfUniqueWords = self.obtainUniqueWords()
    self.id2word = {i: w for i, w in enumerate(self.listOfUniqueWords)}
    self.word2id = {w: i for i, w in enumerate(self.listOfUniqueWords)}
    self.listOfIds = [self.word2id[w] for w in self.listOfWords] # id of every word

  # obtain jokes from the csv
  def loadJokes(self):
    #csvData = pd.read_csv('reddit-cleanjokes.csv')
    csvData = pd.read_csv('shortjokes.csv')
    data=[]
    for index, row in csvData.iterrows():
      data.append(row['Joke'])
    return data

  # obtain words from the csv
  def loadWords(self):
    #csvData = pd.read_csv('reddit-cleanjokes.csv')
    csvData = pd.read_csv('shortjokes.csv')
    return csvData['Joke'].str.cat(sep=' ').split(' ')

  # obtain a deduplicated word list
  def obtainUniqueWords(self):
    wordCounts = Counter(self.listOfWords)
    return sorted(wordCounts, key=wordCounts.get, reverse=True)
  
  # obtain the overall number of jokes
  def __len__(self):
    return len(self.listOfJokes) # overall number of jokes
  
  # obtain the joke corresponding to the index
  def __getitem__(self, index):
    return self.listOfJokes[index] # return the joke

## 2. construct datasets
dataSet = JokeDataset()

## 3. split datasets
# 1)obtain the first 1.5% of the original data as the dataset to be used
small_size=int(len(dataSet) * 0.015)
print("smaller dataset size: ", small_size)
partial_dataSet = dataSet[:small_size]
# 2)split datasets
train_data_size = int(len(partial_dataSet) * 0.8)
test_data_size = int(len(partial_dataSet) * 0.1)
validation_data_size = len(partial_dataSet) - train_data_size - test_data_size
 
train_dataSet, test_dataSet, validation_dataSet = random_split(dataset=partial_dataSet, 
                                                               lengths=[train_data_size, test_data_size, validation_data_size])

## 4. construct dataLoaders
batch_size = 4
train_loader = DataLoader(train_dataSet, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader = DataLoader(test_dataSet, batch_size=batch_size, shuffle=True, drop_last=False)
validation_loader = DataLoader(validation_dataSet, batch_size=batch_size, shuffle=True, drop_last=False)
print("the size of words in the original dataset: ", len(dataSet.listOfWords))
print("train_dataset's size: ", len(train_dataSet))
print("test_dataset's size: ", len(test_dataSet))
print("validation_dataset's size: ", len(validation_dataSet))
print("train_dataset's batch size: ",train_loader.batch_size)
print("the size of each batch in the train_dataset: ",len(train_loader))
print("test_dataset's batch size: ",test_loader.batch_size)
print("the size of each batch in the test_dataset: ",len(test_loader))
print("validation_dataset's batch size: ",validation_loader.batch_size)
print("the size of each batch in the validation_dataset: ",len(validation_loader))

## 5.define the model
class BERTModel(torch.nn.Module):
  def __init__(self):
    super(BERTModel, self).__init__()
    self.bert = BertForMaskedLM.from_pretrained('bert-base-cased') # case sensitive
    ##self.dropout = nn.Dropout(0.3) # dropout layer
    ##self.fc1 = nn.Linear(self.bert.config.vocab_size, self.bert.config.hidden_size) # fully connected layer
    ##self.fc2 = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size) # fully connected layer
  
  def forward(self, input_ids, attention_mask, token_type_ids, label_ids):  
  #def forward(self, input_data):
    #_, output= self.bert(**input_data) # input a mapping into the bert()
    #print("6: ",input_data)
    #output= self.bert(**input_data) # input a mapping into the bert()
    
    # ↓⭐
    bert_output=self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=label_ids) # input a mapping into the bert()
    ##print("bert_output: ", bert_output)
    logits=bert_output.logits # [batch_size, sequence_length, config.vocab_size]
    ##print("logits_shape: ", logits.shape)
    ##print("logits: ", logits)
    ##drop_output = self.dropout(logits) # [batch_size, sequence_length, config.vocab_size]
    ##print("drop_shape: ",drop_output.shape)
    ##change_shap = self.fc1(drop_output) # [batch_size, sequence_length, config.hidden_size]
    ##print("change_shap_shape: ",change_shap.shape)
    ##output = self.fc2(change_shap) # [batch_size, sequence_length, config.vocab_size]
    ##print("output_shape: ",output.shape)
    ##print("output: ",output)
    #output = self.fc2(drop_output)
    
    ## convert the output to the word indices
    ##prediction = torch.argmax(output, dim=-1)
    prediction = torch.argmax(logits, dim=-1)
    ##print("prediction: ",prediction)
    
    #return bert_output.loss, output
    return bert_output.loss, prediction

  # using the trained model to generate sequences
  def generate(self, bert_input, attention_mask, max_length, num_return_sequences,
               no_repeat_ngram_size, repetition_penalty, do_sample, top_k, top_p,
               temperature, tokenizer):
    return self.bert.generate(input_ids=bert_input, # input
                           attention_mask=attention_mask, # attention mask
                           max_length=max_length, # maximum length of the generated sequences
                           num_return_sequences=num_return_sequences, # the number of returned sequences
                           no_repeat_ngram_size=no_repeat_ngram_size, #避免重复出现长度=no_repeat_ngram_size的单词/词组
                           repetition_penalty=repetition_penalty, # repetition penalty
                           
                           do_sample=do_sample, #是否启用采样策略，选择下一个token
                           top_k=top_k, #每步选择概率最高的token数
                           top_p=top_p, #进行Top-p sampling使用的累积概率阈值
                           temperature=temperature, #调节随机性的温度参数
                           bos_token_id=tokenizer.bos_token_id,
                           pad_token_id=tokenizer.pad_token_id,
                           eos_token_id=tokenizer.eos_token_id)

## 6.set up the device for GPU usage
device = 'cuda' if cuda.is_available() else 'cpu'
print("Using GPU or CPU? ", device)

## 7.create the model
model = BERTModel()
model.to(device)
#print("model: ",model)

## 8.define the optimizer, loss function and hyperparameters
learning_rate=0.001
sum_loss = 0.0
epochs=2
#criterion = nn.CrossEntropyLoss() # loss function (not used currently)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate) # optimizer

## 7.train the model
def train(model, data_loader, optimizer, epochs):
  tokenizer = BertTokenizer.from_pretrained("./bert-base-cased") #tokenizer
  model.train() # open the training mode
  # traverse each epoch
  for epoch in range(epochs):
    sum_loss = 0.0
    print("Epoch {} started.".format(epoch+1))
    process = tqdm(data_loader, desc=f"Epoch {epoch+1} / {epochs}", ncols=100) # process bar

    for i, data in enumerate(process, 0):
      # 1)obtain the inputs
      inputs = data
      #print("inputs: ",inputs)
      #print("real: ",len(inputs))

      for j in range(len(inputs)): # traverse each input
        # 2)here, each inputs represents a joke, we should construct an input that conforms to the BERT format
        bert_input=tokenizer(inputs[j], # input text(need to get the string in the array)
                           return_tensors="pt") # return PyTorch tensor
        input_ids=bert_input['input_ids']
        attention_mask=bert_input['attention_mask']
        token_type_ids=bert_input['token_type_ids']
        ##print("bert_input: ",bert_input)
      #bert_input=tokenizer.encode(inputs[0], # input text(need to get the string in the array)
      #                            return_tensors="pt") # return PyTorch tensor
      #attention_mask=torch.ones(bert_input.shape, dtype=torch.long, device=bert_input.device)

        # 3)zero the parameter gradients
        optimizer.zero_grad()

        # 4)obtain the output and loss
        #  【model(inputs) should be input that conforms to the BERT input format】
        #   here, labels = inputs
      #outputs = model(**bert_input) # equals to calling model.forward(**inputs)
        loss, outputs = model(input_ids, attention_mask, token_type_ids, input_ids) # equals to calling model.forward() 
      #loss = outputs.loss
      #loss = criterion(outputs, labels)
        ##print("loss: ", loss)
        ##print("outputs: ", outputs)
        ##print("output sentence: ",tokenizer.decode(outputs[0], skip_special_tokens=True))

        # 5)compute the gradient
        loss.backward()
        # 6)perform single-step optimization
        optimizer.step()

        # 7)update the total gradient of the current batch
        sum_loss += loss

      # print every 100 mini-batches
      if i % 100 == 99:
        print(f'[{epoch + 1}, {i + 1:5d}]    loss: {sum_loss / 100:.3f}')
        sum_loss = 0.0
  print('Finished training.')

train(model, train_loader, optimizer, epochs)

## 8.save the model
PATH = './BERT_model.pth'
torch.save(model, PATH)

## 9. restore the trained model
#model = torch.load(PATH)

## 10.define the text procedure
def generateText(input_text):
  tokenizer = BertTokenizer.from_pretrained("./bert-base-cased")
  
  # construct an input that conforms to the BERT format
  bert_input=tokenizer.encode(input_text, # input text
                              return_tensors="pt") # return PyTorch tensor
  print("bert_input: ",bert_input)
  
  # construct the attention_mark(using torch.ones to generate a tensor with value 1)
  attention_mask=torch.ones(bert_input.shape, dtype=torch.long, device=bert_input.device)
  
  # using Top-k sampling, Top-p sampling, temperature sampling(without beam-search decoding)
  outputs=model.generate(bert_input, # input
                         attention_mask, # attention mask
                         max_length=50, # maximum length of the generated sequences
                         #min_length=30, # minimum length of the generated sequences
                         num_return_sequences=1, # the number of returned sequences
                         no_repeat_ngram_size=2, #避免重复出现长度=no_repeat_ngram_size的单词/词组
                         repetition_penalty=1.83, # repetition penalty
                         
                         do_sample=True, #是否启用采样策略，选择下一个token
                         top_k=50, #每步选择概率最高的token数
                         top_p=0.92, #进行Top-p sampling使用的累积概率阈值
                         temperature=0.78, #调节随机性的温度参数
                         #num_beams=5, #beam search使用的beam数
                         tokenizer=tokenizer
  )
  #print("outputs: ",outputs)
  
  # decode each joke of the output and get the generated sentence
  jokes = []
  for output in outputs:
    jokes.append(tokenizer.decode(output, skip_special_tokens=True))
  return jokes

## 11. test the model
text="When life gives you lemons"
jokes=generateText(text)
print("generated joke: ")
for i, joke in enumerate(jokes, 0):
  print(f"{i+1}: {joke}")
