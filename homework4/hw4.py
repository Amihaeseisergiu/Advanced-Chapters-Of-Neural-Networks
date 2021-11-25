#!/usr/bin/env python
# coding: utf-8

# In[123]:


import keras
import numpy as np
from keras.layers import Add, Dense, Conv2D, BatchNormalization
from keras.layers import Activation, AveragePooling2D, Input, Flatten
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.models import Model
from keras.layers import add


# In[124]:


batch_size = 128
epochs = 182
n_classes = 10
learning_rate = 1e-1


# In[125]:


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train, n_classes)
y_test = to_categorical(y_test, n_classes)


# In[126]:


class Resnet:
    def __init__(self, size=44, stacks=3, starting_filter=16):
        self.size = size
        self.stacks = stacks
        self.starting_filter = starting_filter
        self.residual_blocks = (size - 2) // 6
        
    def get_model(self, input_shape=(32, 32, 3), n_classes=10):
        n_filters = self.starting_filter

        inputs = Input(shape=input_shape)
        network = self.layer(inputs, n_filters)
        network = self.stack(network, n_filters, True)

        for _ in range(self.stacks - 1):
            n_filters *= 2
            network = self.stack(network, n_filters)

        network = Activation('elu')(network)
        network = AveragePooling2D(pool_size=network.shape[1])(network)
        network = Flatten()(network)
        outputs = Dense(n_classes, activation='softmax', 
                        kernel_initializer='he_normal')(network)

        model = Model(inputs=inputs, outputs=outputs)

        return model
    
    def stack(self, inputs, n_filters, first_stack=False):
        stack = inputs

        if first_stack:
            stack = self.identity_block(stack, n_filters)
        else:
            stack = self.convolution_block(stack, n_filters)

        for _ in range(self.residual_blocks - 1):
            stack = self.identity_block(stack, n_filters)

        return stack
    
    def identity_block(self, inputs, n_filters):
        shortcut = inputs

        block = self.layer(inputs, n_filters, normalize_batch=False)
        block = self.layer(block, n_filters, activation=None)

        block = Add()([shortcut, block])

        return block

    def convolution_block(self, inputs, n_filters, strides=2):
        shortcut = inputs

        block = self.layer(inputs, n_filters, strides=strides,
                           normalize_batch=False)
        block = self.layer(block, n_filters, activation=None)

        shortcut = self.layer(shortcut, n_filters,
                              kernel_size=1, strides=strides,
                              activation=None)

        block = Add()([shortcut, block])

        return block
    
    def layer(self, inputs, n_filters, kernel_size=3,
              strides=1, activation='elu', normalize_batch=True):
    
        convolution = Conv2D(n_filters, kernel_size=kernel_size,
                             strides=strides, padding='same',
                             kernel_initializer="he_normal",
                             kernel_regularizer=l2(1e-4))

        x = convolution(inputs)

        if normalize_batch:
            x = BatchNormalization()(x)

        if activation is not None:
            x = Activation(activation)(x)

        return x
    
def learning_rate_schedule(epoch):
    new_learning_rate = learning_rate

    if epoch <= 91:
        pass
    elif epoch > 91 and epoch <= 137:
        new_learning_rate = learning_rate * 0.1
    else:
        new_learning_rate = learning_rate * 0.01
        
    print('Learning rate:', new_learning_rate)
    
    return new_learning_rate


# In[127]:


resnet = Resnet()

model = resnet.get_model()

optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer, metrics=['accuracy'])

model.summary()


# In[128]:


lr_scheduler = LearningRateScheduler(learning_rate_schedule)
lr_reducer = ReduceLROnPlateau(factor=0.001, patience=3, min_lr=1e-5)

callbacks = [lr_reducer, lr_scheduler]

datagen = ImageDataGenerator(width_shift_range=4,
                             height_shift_range=4,
                             rotation_range=20,
                             zoom_range=0.1,
                             horizontal_flip=True)
datagen.fit(x_train)

model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
          validation_data=(x_test, y_test),
          epochs=epochs, workers=4,
          callbacks=callbacks)

model.save('model.h5')


# In[ ]:




