#!/usr/bin/env python
# coding: utf-8

# In[515]:


"""
References:
https://arxiv.org/pdf/1512.03385.pdf
https://arxiv.org/pdf/1604.04112v4.pdf
https://github.com/yu4u/cutout-random-erasing
"""
import keras
import numpy as np
from keras.layers import Add, Dense, Conv2D, BatchNormalization
from keras.layers import Activation, AveragePooling2D, Input, Flatten
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.models import Model


# In[516]:


batch_size = 128
epochs = 180
n_classes = 10
learning_rate = 0.1


# In[517]:


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train, n_classes)
y_test = to_categorical(y_test, n_classes)


# In[518]:


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

    if epoch <= 90:
        pass
    elif epoch > 90 and epoch <= 140:
        new_learning_rate = learning_rate * 0.1
    else:
        new_learning_rate = learning_rate * 0.01
        
    print('Learning rate:', new_learning_rate)
    
    return new_learning_rate

def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        if input_img.ndim == 3:
            img_h, img_w, img_c = input_img.shape
        elif input_img.ndim == 2:
            img_h, img_w = input_img.shape

        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            if input_img.ndim == 3:
                c = np.random.uniform(v_l, v_h, (h, w, img_c))
            if input_img.ndim == 2:
                c = np.random.uniform(v_l, v_h, (h, w))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w] = c

        return input_img

    return eraser


# In[519]:


resnet = Resnet()

model = resnet.get_model()

optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer, metrics=['accuracy'])

model.summary()


# In[520]:


lr_scheduler = LearningRateScheduler(learning_rate_schedule)
callbacks = [lr_scheduler]

datagen = ImageDataGenerator(width_shift_range=4,
                             height_shift_range=4,
                             horizontal_flip=True,
                             preprocessing_function=get_random_eraser(pixel_level=True))
datagen.fit(x_train)

model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
          validation_data=(x_test, y_test),
          epochs=epochs, workers=4,
          callbacks=callbacks)

model.save('model.h5')


# In[ ]:




