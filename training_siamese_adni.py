# %%
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
sys.path.append('/work/users/d/d/ddinh/aaco/src')
from load_dataset import load_adni_data
from cvar_sensing.utils import prepare_time_series, batch_interp_nd
import torch

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
def load_data(file_path):
    ds = load_adni_data(file_path=file_path)
    x = ds.x
    y = ds.y
    mask_nan = np.isnan(x)
    x[mask_nan] = 0
    
    mask_nan_y = np.isnan(y)
    y[mask_nan_y] = 0
    return x, y

train_x, train_y = load_data("/work/users/d/d/ddinh/aaco/input_data/train_data.npz")
val_x, val_y = load_data("/work/users/d/d/ddinh/aaco/input_data/val_data.npz")
test_x, test_y = load_data("/work/users/d/d/ddinh/aaco/input_data/test_data.npz")

num_ts = train_x.shape[1]

# %%
def forward_fill_imputation(x, y):
    N, T, M = x.shape
    _, _, C = y.shape
    
    for i in range(N):
        # Forward fill X on a per-modality basis
        for t in range(1, T):
            for m in range(M):
                if x[i, t, m] == 0:
                    x[i, t, m] = x[i, t-1, m]

        # Forward fill Y if all classes at time t are zero
        for t in range(1, T):
            if np.all(y[i, t] == 0):
                y[i, t] = y[i, t-1]
                
    return x, y

train_x, train_y = forward_fill_imputation(train_x, train_y)
val_x, val_y = forward_fill_imputation(val_x, val_y)
test_x, test_y = forward_fill_imputation(test_x, test_y)

# %%
train_x = np.transpose(train_x, (0, 2, 1)).reshape(-1, train_x.shape[1] * train_x.shape[2])
val_x = np.transpose(val_x, (0, 2, 1)).reshape(-1, val_x.shape[1] * val_x.shape[2])
test_x = np.transpose(test_x, (0, 2, 1)).reshape(-1, test_x.shape[1] * test_x.shape[2])

# %%
classifier = tf.keras.models.load_model('/work/users/d/d/ddinh/aaco/models/mlp.keras')

# %%

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
def create_base_embedding_model(
    input_shape=(None,),  
    embedding_dim=32
):
    
    inputs = keras.Input(shape=input_shape)
    
    x = layers.Dense(32)(inputs)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dense(32)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dense(32)(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    
    embedding = layers.Dense(embedding_dim, activation=None)(x)
    
    base_model = keras.Model(inputs=inputs, outputs=embedding, name="BaseEmbeddingModel")
    return base_model


def create_siamese_model(num_features, embedding_dim=32):
    base_model = create_base_embedding_model(
        input_shape=(num_features,),
        embedding_dim=embedding_dim
    )

    input_a = keras.Input(shape=(num_features,), name='input_a')
    input_b = keras.Input(shape=(num_features,), name='input_b')
    
    emb_a = base_model(input_a)
    emb_b = base_model(input_b)
    
    siamese_model = keras.Model(inputs=[input_a, input_b],
                                outputs=[emb_a, emb_b],
                                name="SiameseModel")
    return siamese_model

def x_masked(x, d=48, num_timestamps=12, num_modalities=4):
    masks = np.concatenate(
        [np.sum(np.random.permutation(np.eye(d))[:, :np.random.randint(int(d*(1/2)))], 1, keepdims=True) for _ in range(x.shape[0])], 1
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
            [np.sum(np.random.permutation(np.eye(d))[:, :np.random.randint(int(d*(1/2)))], 1, keepdims=True) for _ in range(np.sum(masks_zero))], 1
            )).T
        for i in range(num_modalities):
            masks[:, num_timestamps * i + num_timestamps - 1] = 0
            
        masks_zero = np.sum(masks, axis=1) == 0
        
    return np.copy(x), tf.cast(masks, tf.float32)

def compute_accumulated_loss(y_pred, y_true, loss_function, timesteps=12):
    mask = (y_true.sum(dim=-1) != 0)
    
    per_step_loss = []
    for t in range(timesteps):
        pred_t = y_pred[:, t, :] 
        target_t = y_true[:, t, :]      
        loss_t = loss_function(pred_t, target_t)
        per_step_loss.append(loss_t)

    per_step_loss = torch.stack(per_step_loss, dim=1)
    
    if per_step_loss.dim() == 3:
        per_step_loss = per_step_loss.mean(dim=-1) 
    per_step_loss = per_step_loss * mask  

    valid_counts = mask.sum(dim=-1).clamp(min=1) 
    accumulated_loss = per_step_loss.sum(dim=-1) / valid_counts

    return accumulated_loss

def compute_loss_timestep(y_true, x_data, mask, classifier, acquisition_cost, loss_function, num_timestamps=12, num_modalities=4):
    y_pred = np.zeros(y_true.shape)
    for timestamp in range(num_timestamps):
        x_input = np.zeros(x_data.shape)
        mask_input = np.zeros(mask.shape)
        
        for modality_index in range(num_modalities):
            x_input[:, modality_index * num_timestamps: modality_index * num_timestamps + timestamp + 1] = np.copy(
                x_data[:, modality_index * num_timestamps: modality_index * num_timestamps + timestamp + 1])
            mask_input[:, modality_index * num_timestamps: modality_index * num_timestamps + timestamp + 1] = np.copy(
                mask[:, modality_index * num_timestamps: modality_index * num_timestamps + timestamp + 1])

        timestamp_rep = np.repeat(timestamp, x_input.shape[0]).reshape(-1, 1)
        pred = classifier.predict(np.concatenate([x_input * mask_input, timestamp_rep], axis=-1), verbose=0)
        y_pred[:, timestamp, :] = pred
    y_pred = torch.Tensor(y_pred)
    if isinstance(y_true, tf.Tensor):
        y_true = torch.Tensor(y_true.numpy())
    else:
        y_true = torch.Tensor(y_true)
    total_cost = torch.zeros(mask.shape[0])
    for modality in range(num_modalities):
        modality_cost = 1 if modality in [0, 1] else 0.5
        total_cost += mask[:, modality * num_timestamps: (modality + 1) * num_timestamps].sum(1) * acquisition_cost * modality_cost
    total_loss = compute_accumulated_loss(y_pred, y_true, loss_function) + total_cost
    return total_loss.numpy()

def get_potential_features(x, y, classifier, prev_masks, acquisition_cost, d=48, num_masks=1500, topk=5, num_timestamps=12, num_modalities=4):
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
                round_value = 5
                if np.round(unique_min_k_future_loss[j], round_value) > np.round(min_current_loss, round_value):
                    termination[j] = 1
                    unique_min_k_subsets[j] = prev_masks[i]
        unique_min_k_subsets = unique_min_k_subsets[:topk_min]
        unique_min_k_subsets[:,:d] -= prev_masks[i]
        unique_min_k_subsets = np.concatenate([unique_min_k_subsets, termination[:, None]], axis=1) # shape (topk, d+1)
        
        # turn into probabilities
        # unique_min_k_subsets = unique_min_k_subsets / np.sum(unique_min_k_subsets, axis=1)[:, None]
        denom = np.sum(unique_min_k_subsets, axis=1, keepdims=True)
        denom[denom < 1e-12] = 1e-12
        unique_min_k_subsets /= denom


        subset_losses = subset_losses[:topk_min]
        weights = np.exp(-subset_losses[:topk_min])
        weights /= np.sum(weights)
        distribution = np.sum(unique_min_k_subsets * weights[:, None], axis=0) 
        
        distributions.append(distribution)
        terminations.append(termination)
    # print("Percentage terminated: ", np.mean(np.sum(terminations, axis=1) == 1))
    distributions = np.array(distributions) # shape (batch_size, d)
    distributions = tf.cast(distributions, tf.float32)
    distributions = distributions / (tf.reduce_sum(distributions, axis=1, keepdims=True) + 1e-12) # for numerical stability
    # print("check sum of distributions: ", np.sum(distributions, axis=1))
    
    return distributions

# def compute_neighbors(x, distribution):
#     x = tf.cast(x, tf.float32)
#     distribution = tf.cast(distribution, tf.float32)
    
#     sorted_indices = []
#     for i in range(x.shape[0]):
#         distance = tf.norm(distribution - distribution[i], axis=1) 
#         distance = tf.concat([distance[:i], distance[i+1:]], axis=0) # remove the distance to itself
#         sorted_index = tf.argsort(distance)
#         sorted_indices.append(sorted_index)
        
#     return sorted_indices
    

# def get_pairs(x, distribution, k=3):
#     sorted_indices = compute_neighbors(x, distribution)
#     pairs = []
#     for i in range(x.shape[0]):
#         positive_pairs = sorted_indices[i][:k]
#         negative_pairs = sorted_indices[i][k:] 
#         used_negative_pairs = []
#         for j in range(k * 2):
#             temp_pairs = []
#             temp_pairs.append(x[i])
#             temp_pairs.append(x[positive_pairs[j % k]])
#             random_negative = np.random.choice(negative_pairs)
            
#             while random_negative in used_negative_pairs:
#                 random_negative = np.random.choice(negative_pairs)
#             temp_pairs.append(x[random_negative])
#             used_negative_pairs.append(random_negative)
            
#             pairs.append(temp_pairs)
            
#     pairs = np.array(pairs) # shape (batch_size * k * 2, 3, d)
#     x_anchor, x_positive, x_negative = pairs[:, 0], pairs[:, 1], pairs[:, 2]
#     x_anchor = tf.concat([x_anchor, x_anchor], axis=0)
#     x_pair = tf.concat([x_positive, x_negative], axis=0)
#     target = tf.concat([tf.ones(x_positive.shape[0]), tf.zeros(x_negative.shape[0])], axis=0)
    
#     # shuffle the pairs
#     indices = tf.range(start=0, limit=x_pair.shape[0], dtype=tf.int32)
#     shuffled_indices = tf.random.shuffle(indices)
#     x_anchor = tf.gather(x_anchor, shuffled_indices)
#     x_pair = tf.gather(x_pair, shuffled_indices)
#     target = tf.gather(target, shuffled_indices)
    
#     return x_anchor, x_pair, target

# # Contrastive loss function
# def contrastive_loss(y_true, distances, margin=1.0):
#     positive_loss = y_true * tf.square(distances)  # Similar pairs
#     negative_loss = (1 - y_true) * tf.square(tf.maximum(margin - distances, 0))  # Dissimilar pairs
#     return tf.reduce_mean(0.5 * (positive_loss + negative_loss))

def soft_contrastive_loss(embeddings_i, embeddings_j, similarity, margin=1.0):
    distances = tf.reduce_sum(tf.square(embeddings_i - embeddings_j), axis=1)
    positive_loss = similarity * tf.square(distances)
    negative_loss = (1 - similarity) * tf.square(tf.nn.relu(margin - distances))
    loss = tf.reduce_mean(0.5 * (positive_loss + negative_loss))

    return loss

def compute_similarity(distribution_i, distribution_j, alpha=1.0):
    epsilon = 1e-8
    distribution_i = tf.clip_by_value(distribution_i, epsilon, 1.0)
    distribution_j = tf.clip_by_value(distribution_j, epsilon, 1.0)
    kl_divergence = tf.reduce_sum(distribution_i * tf.math.log(distribution_i / distribution_j), axis=1)
    similarity = tf.exp(-alpha * kl_divergence)

    return similarity

def pair_generator(x, y):
    n_samples = x.shape[0]
    indices = np.random.permutation(n_samples)
    while np.any(indices == np.arange(n_samples)):
        indices = np.random.permutation(n_samples)
    pairs = x[indices].copy()
    labels = y.numpy()[indices].copy()
    return pairs, labels

# %%
def train_step(model, optimizer, x, y, classifier, acquisition_cost, alpha=1):
    x, prev_masks = x_masked(x)
    x_rep, y_rep = pair_generator(x, y)
    
    distributions = get_potential_features(np.copy(x), np.copy(y), classifier, np.copy(prev_masks), acquisition_cost)
    distributions_rep = get_potential_features(np.copy(x_rep), np.copy(y_rep), classifier, np.copy(prev_masks), acquisition_cost)
    similarity = compute_similarity(distributions, distributions_rep, alpha)
    # x_anchor, x_pair, target = get_pairs(np.copy(x) * prev_masks, distributions)
    
    with tf.GradientTape() as tape:
        x_input = x * prev_masks
        x_rep_input = x_rep * prev_masks
        
        emb1, emb2 = model([x_input, x_rep_input])
        # distances = tf.reduce_sum(tf.square(emb1 - emb2), axis=1)
        loss = soft_contrastive_loss(emb1, emb2, similarity)
        
        # cross entropy loss for emb1 vs emb2
        # loss = tf.reduce_mean(distances)
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

def validation_step(model, x, y, classifier, acquisition_cost, alpha=1):
    x_val, prev_masks_val = x_masked(x)
    x_val_rep, y_val_rep = pair_generator(x_val, y)
    distributions_val = get_potential_features(np.copy(x_val), np.copy(y), classifier, np.copy(prev_masks_val), acquisition_cost)
    distributions_val_rep = get_potential_features(np.copy(x_val_rep), np.copy(y_val_rep), classifier, np.copy(prev_masks_val), acquisition_cost)
    
    similarity_val = compute_similarity(distributions_val, distributions_val_rep, alpha)
    
    x_val_input = x_val * prev_masks_val
    x_val_rep_input = x_val_rep * prev_masks_val
    
    emb1_val, emb2_val = model([x_val_input, x_val_rep_input])
    loss_val = soft_contrastive_loss(emb1_val, emb2_val, similarity_val)
    return loss_val


X, X_val, y, y_val = train_x, val_x, train_y, val_y
print(X.shape, y.shape, X_val.shape, y_val.shape)
dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.shuffle(10000).batch(64).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.batch(64).prefetch(tf.data.AUTOTUNE)

num_features = 48
model = create_siamese_model(num_features=num_features, embedding_dim=32)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# todo: cahnga batch size back 
# get rid of the break statement

acquisition_cost = 0.025
num_epochs = 100
best_val_loss = float('inf')
print("***Start Training***", flush=True)
for epoch in range(num_epochs):
    epoch_loss = 0.0
    step_count = 0
    print(f"Epoch {epoch+1}/{num_epochs}", flush=True)
    for batch_x, batch_y in dataset:
        loss = train_step(model, optimizer, batch_x, batch_y, classifier, acquisition_cost)
        epoch_loss += loss.numpy()
        step_count += 1
    
    train_loss = epoch_loss / step_count
    
    val_loss_sum = 0.0
    val_steps = 0
    for val_x_batch, val_y_batch in val_dataset:
        loss_val = validation_step(model, val_x_batch, val_y_batch, classifier, acquisition_cost)
        val_loss_sum += loss_val.numpy()
        val_steps += 1
        
    val_loss = val_loss_sum / val_steps
    
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}", flush=True)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model.save('/work/users/d/d/ddinh/aaco/models/siamese_adni.h5')
        model.save('/work/users/d/d/ddinh/aaco/models/siamese_adni.keras')
        model.save_weights('/work/users/d/d/ddinh/aaco/models/siamese_adni.weights.h5')
        print("Model saved.", flush=True)

    print("")
print("Training Complete.")


# %%
def check_weights_for_nan(model):
    """
    Check all weights of a Keras model for NaN values.

    Args:
        model (tf.keras.Model): The model to check.

    Returns:
        None: Prints which layers or weights have NaN values.
    """
    for layer in model.layers:
        for weight in layer.weights:
            weight_name = weight.name
            if tf.reduce_any(tf.math.is_nan(weight)).numpy():
                print(f"NaN detected in weight: {weight_name}")
            else:
                print(f"Weight {weight_name} is clean (no NaNs).")

check_weights_for_nan(model)


# %%



