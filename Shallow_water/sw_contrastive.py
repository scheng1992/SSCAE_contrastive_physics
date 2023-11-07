# -*- coding: utf-8 -*-

from tensorflow import keras
#from tensorflow.python.keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D,Cropping2D, AveragePooling2D,Dense,Flatten,Reshape,Dropout,TimeDistributed,LSTM,LeakyReLU,RepeatVector

from tensorflow.keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D,Cropping2D,TimeDistributed, AveragePooling2D,Dense,Flatten,Reshape,Dropout,LSTM,LeakyReLU,RepeatVector,Conv2DTranspose
from tensorflow.keras.models import Model,Sequential


import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
#import seaborn_image as isns
import numpy as np
#from matplotlib import cm
#from matplotlib import animation as animation
from PIL import Image

import tensorflow as tf
from tensorflow import keras
import time

from google.colab import drive
drive.mount('/content/drive')

tf.test.gpu_device_name()

field_data = np.load('drive/MyDrive/SW_contrastive/data/frame_data_200.npy')
field_data = field_data/np.max(field_data)

print(field_data.shape)

#TRAIN AUTOENCODER

input_img = Input(shape=(64,64, 1))

x = Convolution2D(4, (4, 4), padding='same')(input_img)
x = LeakyReLU(alpha=0.2)(x)
x = MaxPooling2D((8, 8), padding='same')(x)
x = Convolution2D(8, (4, 4), padding='same')(input_img)
x = LeakyReLU(alpha=0.2)(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Convolution2D(16, (2, 2), padding='same')(x)
x = LeakyReLU(alpha=0.2)(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = LeakyReLU(alpha=0.2)(x)
x = Convolution2D(16, (2, 2), padding='same')(x)
x = LeakyReLU(alpha=0.2)(x)
x = MaxPooling2D((2, 2), padding='same')(x)
flat = Flatten()(x)
encoded = Dense(30)(flat)
encoder = Model(input_img, encoded)

decoder_input= Input(shape=(30,))

decoded = Dense(512)(decoder_input)
x = Reshape((8,8,8))(decoded)
x = Convolution2D(8, (2, 2), padding='same')(x)
x = LeakyReLU(alpha=0.2)(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, (3, 3), padding='same')(x)
x = LeakyReLU(alpha=0.2)(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, (4, 4), padding='same')(x)
x = LeakyReLU(alpha=0.2)(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(1, (8, 8), padding='same')(x)



decoder = Model(decoder_input, decoded)

encoder.summary()

decoder.summary()

from tensorflow.keras import backend as K

auto_input = Input(batch_shape=(None,64,64, 1))
encoded = encoder(auto_input)
decoded = decoder(encoded)

autoencoder = Model(auto_input, decoded)
autoencoder.compile(optimizer='adam', loss="mse")
K.set_value(autoencoder.optimizer.learning_rate, 0.0001)
print(autoencoder.summary())

history = autoencoder.fit(field_data, field_data, epochs=200, batch_size=64,shuffle=True,verbose = 2)

image = field_data[2035,:,:]
decode_image = decoder.predict(encoder.predict(image.reshape(1,64,64,1)))

plt.imshow(image)

encoder.save('SW_contrastive/model/encoder_64_200.h5')

decoder.save('SW_contrastive/model/decoder_64_200.h5')

latent_code_contrastive = encoder.predict(field_data.reshape(-1,64,64,1))
np.save('SW_contrastive/data/latent_code_64_200.npy',np.array(latent_code_contrastive))

"""# supervised Contrastive"""

input_img = Input(shape=(64,64, 1))

x = Convolution2D(4, (4, 4), padding='same')(input_img)
x = LeakyReLU(alpha=0.2)(x)
x = MaxPooling2D((8, 8), padding='same')(x)
x = Convolution2D(8, (4, 4), padding='same')(input_img)
x = LeakyReLU(alpha=0.2)(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Convolution2D(16, (2, 2), padding='same')(x)
x = LeakyReLU(alpha=0.2)(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = LeakyReLU(alpha=0.2)(x)
x = Convolution2D(16, (2, 2), padding='same')(x)
x = LeakyReLU(alpha=0.2)(x)
x = MaxPooling2D((2, 2), padding='same')(x)
flat = Flatten()(x)
encoded = Dense(30)(flat)
encoder = Model(input_img, encoded)

decoder_input= Input(shape=(30,))

decoded = Dense(512)(decoder_input)
x = Reshape((8,8,8))(decoded)
x = Convolution2D(8, (2, 2), padding='same')(x)
x = LeakyReLU(alpha=0.2)(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, (3, 3), padding='same')(x)
x = LeakyReLU(alpha=0.2)(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, (4, 4), padding='same')(x)
x = LeakyReLU(alpha=0.2)(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(1, (8, 8), padding='same')(x)



decoder = Model(decoder_input, decoded)

import tensorflow_addons as tfa
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
    inputs = tf.keras.Input(shape=(64,64, 1))
    features = encoder(inputs)
    outputs = Dense(100, activation="relu")(features)
    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name="cifar-encoder_with_projection-head"
    )
    return model

inputs = tf.keras.Input(shape=(64,64, 1))
features = encoder(inputs)
last = Dense(100, activation="relu")(features)
decoded = decoder(features)
model = Model(inputs=inputs, outputs=[decoded,last])

label_data = []

for i in range(200):
  label_data += [i]*(100)

label_data = np.array(label_data).astype(np.uint8)

from tensorflow.keras.callbacks import Callback

class LossHistory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Log the loss values at the end of each epoch
        mse_loss = logs['mse']  # Modify this if your loss is named differently
        scl_loss = logs[SupervisedContrastiveLoss]  # Replace 'your_loss_component' with the actual name

        print(f"Epoch {epoch+1} - MSE Loss: {mse_loss:.4f}, SCL Loss: {scl_loss:.4f}")

# Use the custom callback during training
loss_history_callback = LossHistory()

alpha = 0.996

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    #loss=SupervisedContrastiveLoss(0.01),
    loss=["mse",SupervisedContrastiveLoss(0.01)],loss_weights=[alpha,1-alpha]
)
history = model.fit(
    x=field_data, y=[field_data,label_data], batch_size=64, epochs=800,verbose=2
)

np.save('SW_contrastive/data/SW_MSE_loss_final.npy',np.array(history.history['model_4_loss']))
np.save('SW_contrastive/data/SW_CL_loss_final.npy',np.array(history.history['dense_5_loss']))

plt.plot(np.array(history.history['model_4_loss'])[50:]*0.998,'b',linewidth = 2)

plt.plot(np.array(history.history['dense_5_loss'])[50:]*0.002,'r',linewidth = 2)

plt.legend()

plt.plot(np.array(history.history['model_1_loss'])[50:]*0.99,'b',linewidth = 2)

plt.plot(np.array(history.history['dense_2_loss'])[50:]*0.01,'r',linewidth = 2)

plt.yscale('log')

plt.legend()

alpha = 0.99

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00001),
    #loss=SupervisedContrastiveLoss(0.01),
    loss=["mse",SupervisedContrastiveLoss(0.01)],loss_weights=[alpha,1-alpha]
)
history = model.fit(
    x=field_data, y=[field_data,label_data], batch_size=64, epochs=300,verbose=2
)

plt.imshow(image)

latent_code_contrastive = encoder.predict(field_data[:20*30,:,:].reshape(-1,64,64,1))

plt.imshow(field_data[10,:,:])

encoder.save('SW_contrastive/model/encoder_64_contrastive_200.h5')

decoder.save('SW_contrastive/model/decoder_64_contrastive_200.h5')

latent_code_contrastive = encoder.predict(field_data.reshape(-1,64,64,1))

latent_code_contrastive.shape

np.save('SW_contrastive/data/latent_code_64_contrastive_200.npy',np.array(latent_code_contrastive))

