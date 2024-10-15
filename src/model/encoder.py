# src/model/encoder.py

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input
from tensorflow.keras.models import Model

def build_encoder(input_shape):
    input_layer = Input(shape=input_shape)

    # Multi-resolution feature extraction
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # Further layers...
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    
    return Model(inputs=input_layer, outputs=x)
