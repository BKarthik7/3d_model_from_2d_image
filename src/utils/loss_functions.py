# src/utils/loss_functions.py

import tensorflow as tf

def chamfer_distance(y_true, y_pred):
    """Chamfer Distance for point cloud comparison."""
    # Calculate pairwise distances between points in y_true and y_pred
    diff = tf.expand_dims(y_true, axis=2) - tf.expand_dims(y_pred, axis=1)
    dist = tf.reduce_sum(tf.square(diff), axis=-1)
    return tf.reduce_mean(dist)
