# src/data.py

import numpy as np

def load_data():
    # Example data loading function
    # Replace this with actual code to load your dataset
    # Example: Load from files or generate data
    train_data = np.random.rand(1000, 64, 64, 3)  # Example dataset (1000 images of shape 64x64x3)
    train_labels = np.random.randint(0, 10, 1000)  # Example labels
    
    return train_data, train_labels
