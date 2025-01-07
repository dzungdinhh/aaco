import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from xgboost import XGBClassifier
from xgboost import XGBRegressor
import torch.nn as nn
import tqdm
import torch
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
sys.path.append('/work/users/d/d/ddinh/aaco/src')
# from load_dataset import load_adni_data
# from cvar_sensing.utils import prepare_time_series, batch_interp_nd
# import torch

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import Sequence
from tensorflow.keras import layers, Model
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
# %%
X_train_digits = np.load('/work/users/d/d/ddinh/aaco/input_data/Synthetic_raw/train_digits.npy')
X_train_counter = np.load('/work/users/d/d/ddinh/aaco/input_data/Synthetic_raw/train_counter.npy')
y_train = np.load('/work/users/d/d/ddinh/aaco/input_data/Synthetic_raw/train_labels.npy')
X_test_digits = np.load('/work/users/d/d/ddinh/aaco/input_data/Synthetic_raw/test_digits.npy')
X_test_counter = np.load('/work/users/d/d/ddinh/aaco/input_data/Synthetic_raw/test_counter.npy')
y_test = np.load('/work/users/d/d/ddinh/aaco/input_data/Synthetic_raw/test_labels.npy')

train_y = np.eye(np.max(y_train) + 1)[y_train]
y_test = np.eye(np.max(y_test) + 1)[y_test]

train_x = np.concatenate([X_train_digits, X_train_counter], axis=1)
X_test = np.concatenate([X_test_digits, X_test_counter], axis=1)

# %%
classifier = XGBClassifier(n_estimators=256, device='gpu')
classifier.load_model('/work/users/d/d/ddinh/aaco/models/synthetic_raw.model')

# %%
# todo: implement progressively acquire features version 

def create_base_embedding_model(
    input_shape=(None,),  # You can replace None with num_features
    embedding_dim=32
):
    inputs = keras.Input(shape=input_shape)
    
    x = layers.Dense(32, activation=layers.LeakyReLU(alpha=0.01))(inputs)
    x = layers.Dense(32, activation=layers.LeakyReLU(alpha=0.01))(x)
    x = layers.Dense(32, activation=layers.LeakyReLU(alpha=0.01))(x)
    
    # Embedding head
    embedding = layers.Dense(embedding_dim, activation=None)(x)
    
    base_model = keras.Model(inputs=inputs, outputs=embedding, name="BaseEmbeddingModel")
    return base_model


def create_siamese_model(num_features, embedding_dim=32):
    """
    Builds a siamese model using the base embedding model.
    Takes two inputs (input_a and input_b), outputs the Euclidean distance.
    """
    # Create the base model
    base_model = create_base_embedding_model(
        input_shape=(num_features,),
        embedding_dim=embedding_dim
    )
    
    # Define two inputs
    input_a = keras.Input(shape=(num_features,), name='input_a')
    input_b = keras.Input(shape=(num_features,), name='input_b')
    
    # Pass both inputs through the base embedding model
    emb_a = base_model(input_a)
    emb_b = base_model(input_b)
    
    # Compute the Euclidean distance between embeddings
    # (add a small epsilon to avoid sqrt(0))
    distance_layer = layers.Lambda(
        lambda tensors: tf.norm(tensors[0] - tensors[1], axis=1, keepdims=True) + 1e-12
    )
    distance = distance_layer([emb_a, emb_b])
    
    # Create the full siamese model
    siamese_model = keras.Model(inputs=[input_a, input_b],
                                outputs=distance,
                                name="SiameseModel")
    return siamese_model


def x_masked(x, d=20, num_timestamps=10, num_modalities=2):
    masks = np.concatenate(
        [np.sum(np.random.permutation(np.eye(d))[:, :np.random.randint(int(d*(3/4)))], 1, keepdims=True) for _ in range(x.shape[0])], 1
        )
    masks = np.float32(masks.T)

    # ensure there is no 1 at the end of time step 
    for i in range(num_modalities):
        masks[:, num_timestamps * i + num_timestamps - 1] = 0
        
    masks_zero = np.sum(masks, axis=1) == 0
    
    
    while np.sum(masks_zero) > 0:
        # masks[masks_zero] = np.float32(np.concatenate(
        #     [np.sum(np.random.permutation(np.eye(d))[:, :np.random.randint(d)], 1, keepdims=True) for _ in range(np.sum(masks_zero))], 1
        #     )).T
        masks[masks_zero] = np.float32(np.concatenate(
            [np.sum(np.random.permutation(np.eye(d))[:, :np.random.randint(int(d*(3/4)))], 1, keepdims=True) for _ in range(np.sum(masks_zero))], 1
            )).T
        for i in range(num_modalities):
            masks[:, num_timestamps * i + num_timestamps - 1] = 0
            
        masks_zero = np.sum(masks, axis=1) == 0
        
    return np.copy(x), tf.cast(masks, tf.float32)


def compute_loss_timestep(y_true, x_data, mask, classifier, acquisition_cost, loss_function, num_timestamps=10, num_modalities=2):
    y_pred = classifier.predict_proba(np.concatenate([x_data * mask, mask], axis=1))
    y_pred = torch.Tensor(y_pred)
    if isinstance(y_true, tf.Tensor):
        y_true = torch.Tensor(y_true.numpy())
    else:
        y_true = torch.Tensor(y_true)
    total_cost = torch.zeros(mask.shape[0])
    total_cost += mask.sum(axis=1) * acquisition_cost
    total_loss = loss_function(y_pred, y_true).numpy() + total_cost.numpy()
    return total_loss

def get_potential_features(x, y, classifier, prev_masks, acquisition_cost, d=20, num_masks=1500, topk=5, num_timestamps=10, num_modalities=2):
    new_masks = np.concatenate(
        [np.sum(np.random.permutation(np.eye(d))[:, :np.random.randint(d)], 1, keepdims=True) for _ in range(num_masks)], 1
        )
    new_masks = np.float32(new_masks.T)

    # repeat for parallelization
    x_rep = np.repeat(x, num_masks, axis=0)
    y_rep = np.repeat(y, num_masks, axis=0)
    new_masks = np.concatenate([new_masks for _ in range(x.shape[0])], 0)
    prev_masks_rep = np.repeat(prev_masks, num_masks, axis=0)

    # combine previous masks with new masks
    N = x_rep.shape[0]
    segments_previous = prev_masks_rep.reshape(N, num_modalities, num_timestamps)
    segments_after = new_masks.reshape(N, num_modalities, num_timestamps)
    last_indices = np.where(segments_previous == 1, np.arange(num_timestamps), -1)
    last_indices_per_segment = np.max(last_indices, axis=2)  
    final_last_index = np.max(last_indices_per_segment, axis=1)  
    mask = np.arange(num_timestamps)[None, None, :] > final_last_index[:, None, None]
    final_segments = np.where(mask, segments_after, segments_previous)
    final = final_segments.reshape(N, num_timestamps * num_modalities)
    
    # compute loss for current and future masks
    current_loss = compute_loss_timestep(y, x, prev_masks, classifier, acquisition_cost, nn.CrossEntropyLoss(reduction='none'))
    future_loss = compute_loss_timestep(y_rep, x_rep, final, classifier, acquisition_cost, nn.CrossEntropyLoss(reduction='none'))
    
    # get the top k min loss
    top_k_sets = []
    distributions = []
    terminations = []
    percentage_unique = []
    for i in range(x.shape[0]):
        min_current_loss = current_loss[i]
        single_sample_future_loss = np.copy(future_loss[i * num_masks: (i + 1) * num_masks])
        single_sample_future_set = np.copy(final[i * num_masks: (i + 1) * num_masks])
        
        top_k_loss = np.argsort(single_sample_future_loss)
        min_k_future_loss = single_sample_future_loss[top_k_loss]
        min_k_subsets = single_sample_future_set[top_k_loss]
        
        unique_min_k_subsets, unique_indices = np.unique(min_k_subsets, axis=0, return_index=True)
        unique_min_k_future_loss = min_k_future_loss[unique_indices]
        
        sorted_unique_indices = np.argsort(unique_min_k_future_loss)
        unique_min_k_future_loss = unique_min_k_future_loss[sorted_unique_indices]
        unique_min_k_subsets = unique_min_k_subsets[sorted_unique_indices]
        
        if unique_min_k_subsets.shape[0] <= 1:
            percentage_unique.append(0)
            topk_min = topk
            if np.equal(unique_min_k_subsets[0], prev_masks[i]).all():
                termination = np.ones(topk)
                unique_min_k_subsets = np.repeat(prev_masks[i][None, :], topk, axis=0)
                subset_losses = np.ones(topk) * min_current_loss
            else:
                if unique_min_k_future_loss[0] >= min_current_loss:
                    termination = np.ones(topk)
                    unique_min_k_subsets = np.repeat(prev_masks[i][None, :], topk, axis=0)
                    subset_losses = np.ones(topk) * min_current_loss
                else: 
                    termination = np.zeros(topk)
                    unique_min_k_subsets = np.repeat(unique_min_k_subsets[0], topk, axis=0)
                    subset_losses = np.repeat(unique_min_k_future_loss[0], topk)
        else: 
            percentage_unique.append(1)
            topk_min = min(topk, unique_min_k_subsets.shape[0])
            termination = np.zeros(topk_min)
            subset_losses = unique_min_k_future_loss[:topk_min]

            for j in range(topk_min):
                if unique_min_k_future_loss[j] >= min_current_loss:
                    termination[j] = 1
                    unique_min_k_subsets[j] = prev_masks[i]
                    
        unique_min_k_subsets = unique_min_k_subsets[:topk_min]
        unique_min_k_subsets[:,:d] -= prev_masks[i]
        unique_min_k_subsets = np.concatenate([unique_min_k_subsets, termination[:, None]], axis=1) # shape (topk, d+1)
        
        # turn into probabilities
        unique_min_k_subsets = unique_min_k_subsets / np.sum(unique_min_k_subsets, axis=1)[:, None]
        subset_losses = subset_losses[:topk_min]
        weights = np.exp(-subset_losses[:topk_min])
        weights /= np.sum(weights)
        distribution = np.sum(unique_min_k_subsets * weights[:, None], axis=0) 
        
        distributions.append(distribution)
        terminations.append(termination)

    distributions = np.array(distributions) # shape (batch_size, d)
    distributions = tf.cast(distributions, tf.float32)
    distributions = distributions / (tf.reduce_sum(distributions, axis=1, keepdims=True) + 1e-12)
    # print("check sum of distributions: ", np.sum(distributions, axis=1))
    
    return distributions

def pair_generator(x, y):
    # given x, y, generate the corresponding pairs
    # really simple, just shuffle the indices and make sure they are not the same index
    n_samples = x.shape[0]
    indices = np.random.permutation(n_samples)
    while np.any(indices == np.arange(n_samples)):
        indices = np.random.permutation(n_samples)
    pairs = x[indices].copy()
    labels = y.numpy()[indices].copy()
    return pairs, labels

import tensorflow as tf
from tensorflow.keras import layers

class EuclideanDistanceLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(EuclideanDistanceLayer, self).__init__(**kwargs)

    def call(self, inputs):
        emb_a, emb_b = inputs
        squared_diff = tf.square(emb_a - emb_b)
        sum_squared = tf.reduce_sum(squared_diff, axis=1, keepdims=True)
        distance = tf.sqrt(sum_squared + 1e-12)
        return distance

def mse_distance_loss(y_true, y_pred):
    """
    y_true: Tensor of shape (batch_size, 1) representing distribution distances
    y_pred: Tensor of shape (batch_size, 1) representing embedding distances
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    tf.debugging.check_numerics(y_true, "NaN/Inf in y_true")
    tf.debugging.check_numerics(y_pred, "NaN/Inf in y_pred")

    return tf.reduce_mean(tf.square(y_true - y_pred))

def train_step(model, optimizer, x, y, classifier, acquisition_cost, alpha=1):
    x, prev_masks = x_masked(x)
    
    x_rep, y_rep = pair_generator(x, y)
    
    distributions = get_potential_features(np.copy(x), np.copy(y), classifier, np.copy(prev_masks), acquisition_cost)
    distributions_rep = get_potential_features(np.copy(x_rep), np.copy(y_rep), classifier, np.copy(prev_masks), acquisition_cost)

    
    target_distance = tf.norm(distributions - distributions_rep, axis=1, keepdims=True) + 1e-12
    with tf.GradientTape() as tape:
        pred_distance = model([x * prev_masks, x_rep * prev_masks], training=False)
        loss = mse_distance_loss(target_distance, pred_distance)
        print("LOSS: ", loss)
    
    # Backpropagation
    grads = tape.gradient(loss, model.trainable_variables)
    for g, v in zip(grads, model.trainable_variables):
        if g is not None:
            print(f"Gradient for {v.name}: max={tf.reduce_max(g).numpy()}, min={tf.reduce_min(g).numpy()}")
            tf.debugging.check_numerics(g, f"NaN/Inf in gradient for {v.name}")


    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Check weights after update
    for v in model.trainable_variables:
        tf.debugging.check_numerics(v, f"NaN/Inf in weights of {v.name}")

    return loss

def validation_step(model, x, y, classifier, acquisition_cost, alpha=1):
    x, prev_masks = x_masked(x)
    x_rep, y_rep = pair_generator(x, y)
    
    distributions = get_potential_features(np.copy(x), np.copy(y), classifier, np.copy(prev_masks), acquisition_cost)
    distributions_rep = get_potential_features(np.copy(x_rep), np.copy(y_rep), classifier, np.copy(prev_masks), acquisition_cost)
    
    pred_distance = model([x * prev_masks, x_rep * prev_masks])
    target_distance = tf.norm(distributions - distributions_rep, axis=1, keepdims=True)

    loss = mse_distance_loss(target_distance, pred_distance)
    
    return loss


X, X_val, y, y_val = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.shuffle(10000).batch(32).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

num_features = 20
model = create_siamese_model(num_features, embedding_dim=32)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
# todo: cahnga batch size back 
# get rid of the break statement

acquisition_cost = 0.015
num_epochs = 50
best_val_loss = float('inf')
print("***Start Training***")
for epoch in range(num_epochs):
    epoch_loss = 0.0
    step_count = 0
    print(f"Epoch {epoch+1}/{num_epochs}")
    for batch_x, batch_y in dataset:
        print(f"Training batch {step_count}")
        loss = train_step(model, optimizer, batch_x, batch_y, classifier, acquisition_cost)
        epoch_loss += loss.numpy()
        step_count += 1
        if step_count == 2:
            break
    train_loss = epoch_loss / step_count
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
    val_loss_sum = 0.0
    val_steps = 0
    for val_x_batch, val_y_batch in val_dataset:
        print(f"Validation batch {val_steps}")
        loss_val = validation_step(model, val_x_batch, val_y_batch, classifier, acquisition_cost)
        val_loss_sum += loss_val.numpy()
        val_steps += 1
        break
    val_loss = val_loss_sum / val_steps
    
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model.save('/work/users/d/d/ddinh/aaco/models/siamese.h5')
        model.save('/work/users/d/d/ddinh/aaco/models/siamese.keras')
        model.save_weights('/work/users/d/d/ddinh/aaco/models/siamese.weights.h5')
        print("Model saved.")

print("Training Complete.")
