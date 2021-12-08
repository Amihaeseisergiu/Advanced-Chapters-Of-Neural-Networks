#!/usr/bin/env python
# coding: utf-8

# In[300]:


import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, LSTM, Embedding, Input
from tensorflow.keras.layers import Dropout, Conv1D
from tensorflow.keras.layers import Activation, BatchNormalization, Add
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Model


# In[301]:


max_words = 5000
max_review_length = 500
embedding_vecor_length = 32

epochs = 3
batch_size = 64
learning_rate = 0.1


# In[302]:


(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)

x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)


# In[303]:


inputs = Input(shape=(None,), dtype="int32")
x = Embedding(max_words, embedding_vecor_length, input_length=max_review_length)(inputs)
x = Conv1D(filters=32, kernel_size=3, padding='same')(x)
x = BatchNormalization()(x)
shortcut = Activation('elu')(x)
x = Conv1D(filters=32, kernel_size=3, padding='same')(shortcut)
x = Activation('elu')(x)
x = Conv1D(filters=32, kernel_size=3, padding='same')(x)
x = BatchNormalization()(x)
outputs = Add()([shortcut, x])
outputs = LSTM(125)(outputs)
outputs = Dropout(0.2)(outputs)
outputs = Dense(1, activation='sigmoid')(outputs)
model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()


# In[304]:


model.fit(x_train, y_train, validation_data=(x_test, y_test),
          workers=4, epochs=epochs, batch_size=batch_size)

model.save('model.h5')


# In[ ]:




