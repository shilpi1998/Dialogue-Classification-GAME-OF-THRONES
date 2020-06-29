import numpy as np       
import pandas as pd 
import os,pickle
from keras.callbacks import ModelCheckpoint
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, Input, Dropout,Conv1D,MaxPooling1D,Flatten
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
from keras.models import load_model
from keras.layers import Embedding
from keras.utils import to_categorical 

dic = {0:'HATE' ,
       1:'ROMANTIC',
       2:'neutral',
       3:'mixed' }

# In[2]:

train_csv = './test.csv'
data = pd.read_csv(train_csv)
pd.set_option('display.max_colwidth',-1)

# In[8]:
    
num_words = 5000
tokenizer = Tokenizer(num_words=num_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   lower=True,split=' ')
tokenizer.fit_on_texts(data['Sentence'].values)

# loading
with open('./tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

X = tokenizer.texts_to_sequences(data['Sentence'].values)
      
word_index = tokenizer.word_index


# In[16]:

max_length_of_text = 150
X = pad_sequences(X, maxlen=max_length_of_text)

# In[30]:

embed_dim = 50
batch_size = 16

inputs = Input((max_length_of_text, ))
x = Embedding(num_words, 50)(inputs)
x = LSTM(64, dropout=0.4, recurrent_dropout=0.4)(x)
x = Dense(32,activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(4,activation='softmax')(x)
model = Model(inputs, x)


model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

filepath="./weights_save.hdf5"
model = load_model(filepath)
y = np.argmax(model.predict(X),axis=1)


y = [dic.get(y[l]) for l in range(len(y))]
y = pd.DataFrame({"Index"     : [i+1 for i in range(len(y))] ,
                  "Sentiment" : y})
y.to_csv("./solution.csv",index=False)

