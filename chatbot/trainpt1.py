import json
import numpy as np

with open('intents.json', 'r') as f:
    intents = json.load(f)
   #print(intents)
from main import  tokenize, stem, bag_of_words
#from nltk.stem import WordNetLemmatizer
#lemmatizer=WordNetLemmatizer()
#from main import tokenize ,lemmatizer,pos_tag,bag_of_words

#import torch
#mport torch.nn as nn
from torch.utils.data import Dataset,DataLoader

#empty list   
all_words = []
tags = []
xy = []

# loop through each sentence in our intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        #1. tokenize each pattern
        w = tokenize(pattern)
        all_words.extend(w) #add all_words of a list to next list
        
        #add xy in the corpus#this will know thw pattern and corresponding tag
        xy.append((w, intent['tag']))
        #add documents(both patterns and tag)to end of list
        
        #add to our tags list
        if intent['tag'] not in tags:
            tags.append(intent['tag'])
    
# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

#print(all_words)
#print(tags)
#print(xy)

# create training data
X_train = [] #bow
y_train = [] #associated label for each tag

for(pattern_sentence, tag) in xy:
     # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels
    label = tags.index(tag)
    y_train.append(label)
    
#conversion to numpy array#x y to feature in a label 
X_train = np.array(X_train)
y_train = np.array(y_train)

#pytorch dataset
class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    #acccess dataset with an index
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
# Hyper-parameters 
batch_size = 8

    
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                         )
