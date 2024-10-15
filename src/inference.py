# src/inference.py

import numpy as np
from src.model.encoder import build_encoder
from src.model.decoder import build_decoder
from src.data_processing import load_image

def predict_3d(image_path):
    """Load image and predict 3D model."""
    image = load_image(image_path)

    # Load trained encoder-decoder model
    encoder = build_encoder(input_shape=(128, 128, 3))
    decoder = build_decoder(encoder.output_shape[1:])
    generator = Model(encoder.input, decoder(encoder.output))
    
    generator.load_weights('./results/checkpoints/generator_weights.h5')

    # Predict 3D model
    predicted_3d = generator.predict(np.expand_dims(image, axis=0))
    
    return predicted_3d
