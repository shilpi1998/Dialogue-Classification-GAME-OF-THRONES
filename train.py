import numpy as np       
import pandas as pd 
import os
from keras.callbacks import ModelCheckpoint
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Dense, Embedding, LSTM, Input, Dropout,Conv1D,MaxPooling1D,Flatten
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import re
from keras.utils import to_categorical 
import pickle

dic1 = {0:'HATE' ,
        1:'ROMANTIC',
        2:'ROMANTIC',
        3:'mixed' }

dic2 = {'HATE' :0,
        'ROMANTIC' :1,
        'neutral'  :2,
        'mixed'    :3 }


# In[2]:

train_csv = './train.csv'
data = pd.read_csv(train_csv)
pd.set_option('display.max_colwidth',-1)


pos = data.loc[data['Sentiment']=='POSTIVE']
pos = pos.iloc[0: 115]
mix = data.loc[data['Sentiment']=='MIXED'] 
mix = mix.append(mix,ignore_index=True)
mix = mix.append(mix,ignore_index=True)
data = data.append(mix, ignore_index=True)
data = data.append(pos, ignore_index=True)

print('length of data:' + str(len(data))  )

data = data.sample(frac=1).reset_index(drop=True)

num_words = 5000
tokenizer = Tokenizer(num_words=num_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   lower=True,split=' ')
tokenizer.fit_on_texts(data['Utterance'].values)
X = tokenizer.texts_to_sequences(data['Utterance'].values)
print(X)
word_index = tokenizer.word_index


# In[16]:

max_length_of_text = 150
X = pad_sequences(X, maxlen=max_length_of_text)


# In[24]:

y = data['Sentiment']
y = [dic2.get(l) for l in y]
print(len(y))
print(y)
y = to_categorical(y, num_classes=4)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

# In[30]:

inputs = Input((max_length_of_text, ))
x = Embedding(num_words, 50)(inputs)
x = LSTM(64, dropout=0.4, recurrent_dropout=0.4)(x)
x = Dense(32,activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(4,activation='softmax')(x)
model = Model(inputs, x)
print(model.summary())

model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

filepath="./weights_chkpt.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit(X_train, y_train, batch_size = 16, epochs = 200,validation_data=(X_test, y_test),callbacks=callbacks_list)


# saving token
with open('./tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

model.load_weights('./weights_chkpt.hdf5')
model.save('./weights_save.hdf5')


