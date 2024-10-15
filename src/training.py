# src/training.py

from tensorflow.keras.models import Model
from src.model.encoder import build_encoder
from src.model.decoder import build_decoder
from src.data import load_data  # Assuming a data module that handles loading

def train():
    # Load your data and get input shape dynamically
    train_data, _ = load_data()  # Assuming load_data returns training data and labels
    input_shape = train_data.shape[1:]  # Get the shape of a single image from the dataset
    
    encoder = build_encoder(input_shape)
    decoder = build_decoder(encoder.output_shape[1:])

    # Define the complete autoencoder model using encoder and decoder
    generator = Model(encoder.input, decoder(encoder.output))
    generator.summary()

# Call the train function
train()
