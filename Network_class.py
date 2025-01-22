'''
1. ResNet152V2
2. InceptionNet v4, 
3. InceptionResNetV2
4. DenseNet201, 2017 
5. EfficientNetB7, 2019

'''
# Modules


from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import tensorflow.keras as K
import sys
from tensorflow.keras.optimizers import Adam,SGD
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.python.keras.layers import InputSpec, Layer
from tensorflow.keras.layers import UpSampling2D,Conv2D, Activation,Flatten
from tensorflow.keras.layers import Dense,Input,add,PReLU

def res_block(model, kernal_size, filters, strides):
    gen = model
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model)
    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)
    model = Conv2D(filters = filters, kernel_size = kernal_size, strides = strides, padding = "same")(model) 
    model = add([gen, model])
    return model

def dense_block(inp_):

    model = res_block(inp_, 3, 512, 1)
    model = res_block(model, 3, 512, 1)
    model = add([inp_, model])
    model = res_block(model, 3, 512, 1)
    model = add([inp_, model])
    model = res_block(model, 3, 512, 1)
    model = add([inp_, model])
    model = res_block(model, 3, 512, 1)

    return model


def res_res_res_block(inp_):

    model = dense_block(inp_)
    model = dense_block(model)
    model = add([inp_, model])
    model = dense_block(model)
    model = add([inp_, model])
    model = dense_block(model)
    model = add([inp_, model])
    model = dense_block(model)

    return model

def discriminator():
    inp = Input(shape = (1000,1000,3))

    model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "valid")(inp)
    model = Conv2D(filters = 64, kernel_size = 4, strides = 1, padding = "valid")(model)
    model = Conv2D(filters = 64, kernel_size = 3, strides = 2, padding = "valid")(model)
    model = Conv2D(filters = 128, kernel_size = 3, strides = 1, padding = "valid")(model)
    model = Conv2D(filters = 128, kernel_size = 4, strides = 2, padding = "valid")(model)
    model = Conv2D(filters = 128, kernel_size = 3, strides = 2, padding = "valid")(model)

    for index in range(3):
        model = res_block(model, 3, 128, 1)

    model = Conv2D(filters = 128, kernel_size = 3, strides = 2, padding = "valid")(model)
    model = Conv2D(filters = 64, kernel_size = 3, strides = 2, padding = "valid",name='last_conv')(model)
           
    model = Flatten()(model)
    model = Dense(8, activation='softmax')(model)

    model = Model(inputs=inp, outputs=model, name='Discriminator')
    d_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=d_opt, loss="categorical_crossentropy",metrics=['accuracy'])

    print(model.summary())

    return model

#generator()
discriminator()