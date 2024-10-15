# main.py

from src.training import train
from src.inference import predict_3d

if __name__ == '__main__':
    # Train the model
    train()

    # Test inference on a single image
    predicted_model = predict_3d('./dataset/images/test_logo.png')
    print(predicted_model)
