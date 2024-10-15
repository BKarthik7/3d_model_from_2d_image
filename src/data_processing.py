# src/data_processing.py

import os
import numpy as np
from skimage.io import imread
from trimesh import load_mesh

def load_image(image_path):
    """Load a 2D image."""
    image = imread(image_path)
    return image / 255.0  # Normalize

def load_3d_model(model_path):
    """Load a 3D model (e.g., mesh or voxel grid)."""
    mesh = load_mesh(model_path)
    return mesh

def load_dataset(image_dir, model_dir):
    """Load images and their corresponding 3D models."""
    images, models = [], []
    for img_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, img_file)
        model_path = os.path.join(model_dir, img_file.replace('.png', '.obj'))  # Assuming .obj format for meshes
        images.append(load_image(image_path))
        models.append(load_3d_model(model_path))
    return np.array(images), np.array(models)
