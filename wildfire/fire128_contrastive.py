# -*- coding: utf-8 -*-


import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import animation as animation
from PIL import Image
import matplotlib as mpl
import time
from tensorflow.keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D,Cropping2D, AveragePooling2D,Dense,Flatten,Reshape,Dropout
from tensorflow.keras.models import Model,Sequential
import tensorflow as tf
from tensorflow import keras
import pickle
#import keras
from tensorflow.keras.layers import LeakyReLU
import time
import tensorflow_addons as tfa

tf.test.gpu_device_name()


field_data = np.load('CA_contrastive/data/all_fire_Bear_field_25fire_tight.npy')

field_data = field_data.reshape(-1,128,128,1)
print(field_data.shape)

input_img = Input(shape=(128,128, 1))

x = Convolution2D(4, (8, 8), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Convolution2D(8, (2, 2), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
flat = Flatten()(x)
encoded = Dense(30)(flat)
encoder = Model(input_img, encoded)

decoder_input= Input(shape=(30,))

decoded = Dense(512)(decoder_input)
x = Reshape((8,8,8))(decoded)
x = Convolution2D(8, (2, 2), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(4, (10, 10), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
#x = Cropping2D(cropping=((1, 0), (8, 0)), data_format=None)(x)
decoded = Convolution2D(1, (8, 8), activation='sigmoid', padding='same')(x)

decoder = Model(decoder_input, decoded)

decoder.summary()

from tensorflow.keras import backend as K

auto_input = Input(batch_shape=(None,128,128, 1))
encoded = encoder(auto_input)
decoded = decoder(encoded)

autoencoder = Model(auto_input, decoded)
autoencoder.compile(optimizer='adam', loss="binary_crossentropy")
K.set_value(autoencoder.optimizer.learning_rate, 0.001)
print(autoencoder.summary())

history = autoencoder.fit(field_data, field_data, epochs=500, batch_size=64,shuffle=True,verbose = 2)


encoder.save('CA_contrastive/model/CAE_encoder_Bear_30_25fire_tight_brut_new.h5')
decoder.save('CA_contrastive/model/CAE_decoder_Bear_30_25fire_tight_brut_new.h5')

latent_code_CAE = encoder.predict(field_data)

np.save('CA_contrastive/data/CAE128_latent_Bear_30_25fire_tight_brut_new.npy',latent_code_CAE)

"""# Contrastive loss"""


class SupervisedContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super().__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors

        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)


def add_projection_head(encoder):
    inputs = tf.keras.Input(shape=(128,128, 1))
    features = encoder(inputs)
    outputs = Dense(100, activation="relu")(features)
    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name="cifar-encoder_with_projection-head"
    )
    return model

inputs = tf.keras.Input(shape=(128,128, 1))
features = encoder(inputs)
last = Dense(100, activation="relu")(features)
decoded = decoder(features)
model = Model(inputs=inputs, outputs=[decoded,last])


label_data = []

for i in range(25):
  label_data += [i]*(5*50)

label_data = np.array(label_data).astype(np.uint8)

alpha = 0.99

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    #loss=SupervisedContrastiveLoss(0.01),
    loss=["binary_crossentropy",SupervisedContrastiveLoss(0.01)],loss_weights=[alpha,1-alpha]
)
history = model.fit(
    x=field_data, y=[field_data,label_data], batch_size=64, epochs=800,verbose=2
)

np.save('CA_contrastive/data/CA_MSE_loss_final.npy',np.array(history.history['model_1_loss']))
np.save('CA_contrastive/data/CA_CL_loss_final.npy',np.array(history.history['dense_2_loss']))

plt.plot(np.array(history.history['model_1_loss'])[50:]*0.99,'b',linewidth = 2)

plt.plot(np.array(history.history['dense_2_loss'])[50:]*0.01,'r',linewidth = 2)

plt.legend()



encoder.save('CA_contrastive/model/CAE_encoder_contrastive_Bear_30_25fire_tight.h5')
decoder.save('CA_contrastive/model/CAE_decoder_contrastive_Bear_30_25fire_tight.h5')

latent_code_contrastive = encoder.predict(field_data)

