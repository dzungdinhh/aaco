import numpy as np
import os
import sys
import torch
from torchvision import datasets, transforms
import random

from tensorflow.keras import layers
from tensorflow.keras import models
import tensorflow as tf
from tensorflow.keras.layers import Layer

def reshape_time_to_channels(X):
    """
    X: shape (N, T, H, W) -> (N, H, W, T)
    """
    X = X.transpose((0, 2, 3, 1))  
    return X


class RandomMask(Layer):
    def __init__(self):
        super(RandomMask, self).__init__()
        self.dim = 10  # Number of time steps (channels)
        
    def call(self, inputs):
        # Get the dynamic batch size
        batch_size = tf.shape(inputs)[0]  # Dynamically get the batch size

        # Generate a random slice index for each sample in the batch
        slice_indices = tf.random.uniform(
            shape=(batch_size,), minval=1, maxval=self.dim + 1, dtype=tf.int32
        )

        # Create a batch of random permutation matrices (one for each sample)
        eye = tf.eye(self.dim)  # Identity matrix (dim x dim)
        permuted_eyes = tf.random.shuffle(tf.tile(eye[None, :, :], [batch_size, 1, 1]))  # (batch_size, dim, dim)

        # Create the masks by summing over the random slices
        masks = tf.map_fn(
            lambda x: tf.reduce_sum(x[0][:, :x[1]], axis=1, keepdims=True),
            (permuted_eyes, slice_indices),
            fn_output_signature=tf.float32
        )  # Shape: (batch_size, dim, 1)

        # Reshape and broadcast masks to match the input shape
        masks = tf.transpose(masks, perm=[0, 2, 1])  # Shape: (batch_size, 1, 1, dim)
        masks = tf.expand_dims(masks, axis=1)  # Shape: (batch_size, 1, 1, dim)
        masks = tf.broadcast_to(masks, tf.shape(inputs))  # Shape: (batch_size, height, width, dim)

        # Apply the mask to the inputs
        return inputs * masks

    def get_config(self):
        config = super().get_config().copy()
        return config
    
def resnet_block(x, filters, kernel_size=3, strides=1):
    """
    A simple ResNet block with:
    Conv -> BN -> ReLU -> Conv -> BN -> Add (skip) -> ReLU
    """
    shortcut = x

    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)

    if strides != 1:
        shortcut = layers.Conv2D(filters, 1, strides=strides, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x


def build_resnet_branch(input_shape=(28,28,10), mask_rate=0.3):
    """
    Builds a small ResNet-like model that accepts input of shape (H=28, W=28, channels=10).
    Returns: (input_tensor, output_tensor)
    """
    input_tensor = layers.Input(shape=input_shape)

    # Optionally mask the input at training time
    x = RandomMask()(input_tensor)

    # Initial projection
    x = layers.Conv2D(32, 3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # ResNet blocks
    x = resnet_block(x, filters=32, kernel_size=3, strides=1)  # block 1
    x = resnet_block(x, filters=64, kernel_size=3, strides=2)  # block 2, downsample
    x = resnet_block(x, filters=64, kernel_size=3, strides=1)  # block 3
    x = resnet_block(x, filters=128, kernel_size=3, strides=2) # block 4, downsample

    # Flatten
    x = layers.GlobalAveragePooling2D()(x)  # or Flatten() + Dense, etc.
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    return input_tensor, x

def build_multimodal_resnet(input_shape=(28,28,10),
                            mask_rate=0.3,
                            num_classes=None):
    """
    - input_shape: e.g. (28,28,10)
    - mask_rate: probability for random masking
    - regression: if True, final Dense(1, 'linear')
                  else classification with Dense(num_classes, 'softmax').

    Returns: a compiled Keras Model
    """
    # Digits branch
    digits_input, digits_features = build_resnet_branch(input_shape, mask_rate)

    # Counter branch
    counter_input, counter_features = build_resnet_branch(input_shape, mask_rate)

    # Combine
    combined = layers.Concatenate()([digits_features, counter_features])
    x = layers.Dense(128, activation='relu')(combined)


    # Classification
    if num_classes is None:
        raise ValueError("Must specify num_classes for classification")
    output = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=[digits_input, counter_input], outputs=output)
    model.compile(optimizer='adam', 
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy'])
    return model


X_train_digits_raw = np.load('/work/users/d/d/ddinh/aaco/input_data/Synthetic_images/train_digits_images.npy')
X_train_counter_raw = np.load('/work/users/d/d/ddinh/aaco/input_data/Synthetic_images/train_counter_images.npy')
y_train = np.load('/work/users/d/d/ddinh/aaco/input_data/Synthetic_images/train_labels.npy')

X_test_digits_raw  = np.load('/work/users/d/d/ddinh/aaco/input_data/Synthetic_images/test_digits_images.npy')
X_test_counter_raw = np.load('/work/users/d/d/ddinh/aaco/input_data/Synthetic_images/test_counter_images.npy')
y_test = np.load('/work/users/d/d/ddinh/aaco/input_data/Synthetic_images/test_labels.npy')

X_train_digits  = reshape_time_to_channels(X_train_digits_raw)    
X_train_counter = reshape_time_to_channels(X_train_counter_raw)
X_test_digits   = reshape_time_to_channels(X_test_digits_raw)
X_test_counter  = reshape_time_to_channels(X_test_counter_raw)

from sklearn.model_selection import train_test_split
X_train_digits, X_val_digits, X_train_counter, X_val_counter, y_train_split, y_val_split = \
    train_test_split(X_train_digits, X_train_counter, y_train, test_size=0.2, random_state=42)

print("Train digits shape:", X_train_digits.shape)       # ~80% of original
print("Train counter shape:", X_train_counter.shape)
print("Val digits shape:", X_val_digits.shape)           # ~20% of original
print("Val counter shape:", X_val_counter.shape)
print("y_train_split shape:", y_train_split.shape)
print("y_val_split shape:", y_val_split.shape)


model = build_multimodal_resnet(
    input_shape=(28,28,10),
    mask_rate=0.3,        # random masking rate
    num_classes = np.max(y_train) + 1
)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath='/work/users/d/d/ddinh/aaco/models/synthetic_images.keras',
    monitor='val_loss',          # Monitor validation loss
    mode='min',                  # Minimize the monitored value
    save_best_only=True,         # Save only the best model
    verbose=1
)


history = model.fit(
    x=[X_train_digits, X_train_counter],
    y=y_train_split,
    epochs=20,
    batch_size=256,
    validation_data=([X_val_digits, X_val_counter], y_val_split),
    callbacks=[checkpoint_cb],
    verbose=1
)

print("\nEvaluating on Test set:")
test_loss = model.evaluate([X_test_digits, X_test_counter], y_test, verbose=1)
print("Test Loss:", test_loss)

