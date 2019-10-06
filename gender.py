#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout


# 

# In[ ]:


dataM = pd.read_csv("Female-Names.csv")
dataF = pd.read_csv("Male-Names.csv")


# In[ ]:


df = pd.concat([dataM,dataF])


# In[170]:


#df.head()


# In[ ]:


def should_keep(word):
  if(len(word)) > 19:
    return False
  char_set = [' ', '.', '1', '0', '3', '2', '5', '4', '7', '6', '9', '8', 'END', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i', 'h', 'k', 'j', 'm', 'l', 'o', 'n', 'q', 'p', 's', 'r', 'u', 't', 'w', 'v', 'y', 'x', 'z']  
  for ch in word:
    if ch not in set(char_set):
      return False
  return True
  


# In[ ]:


def clean(word):
  name = str(word)
  name = name.lower()
  if should_keep(name):
    return name
  else:
    return None 
df.name = df.name.apply(lambda word : clean(word))
  


# In[173]:


#df.head(10)


# In[ ]:


df = df.dropna()


# In[ ]:


char_set = [' ', '.', '1', '0', '3', '2', '5', '4', '7', '6', '9', '8', 'END', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i', 'h', 'k', 'j', 'm', 'l', 'o', 'n', 'q', 'p', 's', 'r', 'u', 't', 'w', 'v', 'y', 'x', 'z']
char2idx = {}
index = 0
for ch in char_set:
  char2idx[ch] = index
  index+=1


# In[ ]:


df = df.drop('race',axis=1)
vector_length = 39
max_word_len = 20
words = []
labels= []
for name,gender in df.itertuples(index=False):
  one_hots_word = []
  
  for ch in name:
    vec = np.zeros(vector_length)
    vec[char2idx[ch]] = 1
    one_hots_word.append(vec)
  for _ in range(max_word_len - len(name)):
    vec = np.zeros(vector_length)
    vec[char2idx['END']] = 1
    one_hots_word.append(vec)
  one_hots_word = np.array(one_hots_word)
  words.append(one_hots_word)
  labels.append(gender)


# In[ ]:


words = np.array(words)


# In[178]:


#words.shape


# In[179]:


#len(labels)


# In[ ]:


labels = np.array(labels)
one = OneHotEncoder()
labels_one_hot = one.fit_transform(labels.reshape(-1 , 1)).todense()


# In[181]:


model = Sequential()
model.add(LSTM(128,input_shape=(20,39),return_sequences=True))
model.add(LSTM(120))
model.add(Dropout(.3))
model.add(Dense(2,activation='softmax'))
model.summary()


# In[ ]:


model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(words,labels_one_hot , test_size=0.33, random_state=42)


# In[186]:


model.fit(X_train,y_train,epochs=5)


# In[188]:


model.evaluate(X_test,y_test)

pickle.dump(model, open('model.pkl','wb'))







# In[ ]:


# gur = "gurpreet singh"
# gurf = "gurpreet kaur"


# In[ ]:


# cleaned = clean(gur)


# # In[206]:


# cleaned


# In[ ]:


# def word2vec(urname):
#   char_set = [' ', '.', '1', '0', '3', '2', '5', '4', '7', '6', '9', '8', 'END', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i', 'h', 'k', 'j', 'm', 'l', 'o', 'n', 'q', 'p', 's', 'r', 'u', 't', 'w', 'v', 'y', 'x', 'z']
#   char2idx = {}
#   index = 0
#   for ch in char_set:
#     char2idx[ch] = index
#     index+=1
#   vector_length = 39
#   max_word_len = 20
#   words =[]
#   one_hots_word = []
  
#   for ch in urname:
#     vec = np.zeros(vector_length)
#     vec[char2idx[ch]] = 1
#     one_hots_word.append(vec)
#   for _ in range(max_word_len - len(urname)):
#     vec = np.zeros(vector_length)
#     vec[char2idx['END']] = 1
#     one_hots_word.append(vec)
#   one_hots_word = np.array(one_hots_word)
  
#   return one_hots_word
 
  


# # # In[ ]:


# # cleaned_mat = word2vec("gurpreet kaur")


# # # In[243]:


# # cleaned_mat.shape


# # # In[ ]:


# # cleaned_mat = cleaned_mat.reshape(1,20,39)


# # # In[245]:


# # model.predict(cleaned_mat)


# # # In[246]:


# np.argmax(model.predict(cleaned_mat))


# # In[247]:


# one.categories_


# # In[ ]:




