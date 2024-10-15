# src/model/decoder.py

from tensorflow.keras.layers import Input, Conv2DTranspose
from tensorflow.keras.models import Model

def build_decoder(encoder_output_shape):
    # Decoder to reconstruct 3D model
    decoder_input = Input(shape=encoder_output_shape)
    
    x = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(decoder_input)
    x = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(x)
    
    # Final layer outputting 3D voxel grid (e.g., 64x64x64), modified for 2D transpose conv
    voxel_output = Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)
    
    return Model(inputs=decoder_input, outputs=voxel_output)
