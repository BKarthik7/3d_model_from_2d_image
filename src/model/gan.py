# src/model/gan.py

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

def build_gan(generator, discriminator):
    """Define the GAN architecture."""
    discriminator.trainable = False

    gan_input = generator.input
    gan_output = discriminator(generator.output)

    gan_model = Model(inputs=gan_input, outputs=gan_output)
    return gan_model

def build_discriminator(input_shape):
    """Build the discriminator model."""
    input_layer = Input(shape=input_shape)
    x = Flatten()(input_layer)
    x = Dense(256, activation='relu')(x)
    output_layer = Dense(1, activation='sigmoid')(x)
    
    return Model(inputs=input_layer, outputs=output_layer)
