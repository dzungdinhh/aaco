import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import yaml
from itertools import chain, combinations
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from tensorflow.keras.metrics import AUC
import os
import sys

sys.path.append('./src')
from classifier import classifier_xgb_dict, classifier_ground_truth, classifier_xgb, NaiveBayes
from mask_generator import random_mask_generator, all_mask_generator, generate_all_masks, different_masking
from keras.models import load_model

from load_dataset import load_adni_data

# Load configuration from YAML file
def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Load data based on the dataset provided in the configuration
def load_data(dataset_name):
    if dataset_name == "cube":
        ### Read in Cube data
        cube = np.load("input_data/cube_20_0.3.pkl", allow_pickle=True)
        cube_train = cube.get('train')
        X_train = torch.from_numpy(cube_train[0])
        y_train = F.one_hot(torch.from_numpy(cube_train[1]).long())
        cube_val = cube.get('valid')
        X_valid = torch.from_numpy(cube_val[0])
        y_valid = F.one_hot(torch.from_numpy(cube_val[1]).long())
        initial_feature = 6

    elif dataset_name == "grid":
        ### Read in the grid data
        grid = np.load("input_data/grid_data.pkl", allow_pickle=True)
        grid_train = grid.get('train')
        X_train = grid_train[0]
        y_train = F.one_hot(grid_train[1].squeeze().long())
        grid_val = grid.get('valid')
        X_valid = grid_val[0]
        y_valid = F.one_hot(grid_val[1].squeeze().long())
        ### Normalize data
        X_valid = (X_valid - torch.mean(X_train, dim=0)) / torch.std(X_train, dim=0)
        X_train = (X_train - torch.mean(X_train, dim=0)) / torch.std(X_train, dim=0)
        initial_feature = 1

    elif dataset_name == "gas10":
        ### Read in the gas data
        gas = np.load("input_data/gas.pkl", allow_pickle=True)
        gas_train = gas.get('train')
        X_train = torch.from_numpy(gas_train[0])
        y_train = F.one_hot(torch.from_numpy(gas_train[1]).long())
        gas_val = gas.get('valid')
        X_valid = torch.from_numpy(gas_val[0])
        y_valid = F.one_hot(torch.from_numpy(gas_val[1]).long())
        # Normalize features
        X_valid = (X_valid - torch.mean(X_train, dim=0)) / torch.std(X_train, dim=0)
        X_train = (X_train - torch.mean(X_train, dim=0)) / torch.std(X_train, dim=0)
        initial_feature = 6

    elif dataset_name == "MNIST":
        ### Read in the MNIST data
        mnist = np.load("input_data/MNIST.pkl", allow_pickle=True)
        mnist_train = mnist.get('train')
        X_train = mnist_train[0]
        y_train = mnist_train[1]
        mnist_val = mnist.get('valid')
        X_valid = mnist_val[0]
        y_valid = mnist_val[1]
        # Normalize features
        X_valid = (X_valid - torch.mean(X_train, dim=0)) / torch.std(X_train, dim=0)
        X_train = (X_train - torch.mean(X_train, dim=0)) / torch.std(X_train, dim=0)
        initial_feature = 100
        
    elif dataset_name == "pedestrian":
        train = np.loadtxt('/work/users/d/d/ddinh/aaco/input_data/MelbournePedestrian/MelbournePedestrian_TRAIN.txt')
        X_train = train[:, 1:]
        y_train = train[:, 0]
        valid = np.loadtxt('/work/users/d/d/ddinh/aaco/input_data/MelbournePedestrian/MelbournePedestrian_TEST.txt')
        X_valid = valid[:, 1:]
        y_valid = valid[:, 0]

        y_train = np.eye(11)[y_train.astype(int)]
        y_valid = np.eye(11)[y_valid.astype(int)]
        y_train = y_train[:, 1:]
        y_valid = y_valid[:, 1:]
        
        # convert to tensor
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_valid = torch.tensor(X_valid, dtype=torch.float32)
        y_valid = torch.tensor(y_valid, dtype=torch.float32)
        
        initial_feature = 0
    elif dataset_name == "adni":
        def load_data(file_path):
            ds = load_adni_data(file_path=file_path)
            x = ds.x
            y = ds.y
            mask_nan = np.isnan(x)
            x[mask_nan] = 0

            mask_nan_y = np.isnan(y)
            y[mask_nan_y] = 0
            return np.transpose(x, (0,2,1)).reshape(x.shape[0], -1), y
        
        X_train, y_train = load_data("/work/users/d/d/ddinh/aaco/input_data/train_data.npz")
        X_train_val, y_train_val = load_data("/work/users/d/d/ddinh/aaco/input_data/val_data.npz")
        X_train = np.concatenate([X_train, X_train_val], axis=0)
        y_train = np.concatenate([y_train, y_train_val], axis=0)
        print("*** x train:", X_train.shape)
        # val_x, val_y = load_data("/work/users/d/d/ddinh/aaco/input_data/val_data.npz")
        X_valid, y_valid = load_data("/work/users/d/d/ddinh/aaco/input_data/test_data.npz")
        print("*** x test:", X_valid.shape)
        
        initial_feature = 0
        feature_count = X_train.shape[1]
        
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_valid = torch.tensor(X_valid, dtype=torch.float32)
        y_valid = torch.tensor(y_valid, dtype=torch.float32)
    feature_count = X_train.shape[1]  # Total number of features in the dataset
    return X_train, y_train, X_valid, y_valid, initial_feature, feature_count


# Load the appropriate classifier based on dataset and model
def load_classifier(dataset_name, X_train, y_train, input_dim):
    if dataset_name == "cube":
        # Use the ground truth classifier for Cube dataset
        return classifier_ground_truth(num_features=20, num_classes=8, std=0.3)
    
    elif dataset_name == "grid" or dataset_name == "gas10":
        # Use XGB dictionary classifier for Grid and Gas10 datasets
        return classifier_xgb_dict(output_dim=y_train.shape[1], input_dim=input_dim, subsample_ratio=0.01, X_train=X_train, y_train=y_train)

    elif dataset_name == "MNIST":
        # Load XGBoost model for MNIST dataset
        xgb_model = XGBClassifier()
        xgb_model.load_model('models/xgb_classifier_MNIST_random_subsets_5.json')
        return classifier_xgb(xgb_model)
    elif dataset_name == "pedestrian":
        xgb_model = XGBClassifier()
        xgb_model.load_model('/work/users/d/d/ddinh/aaco/models/pedestrian.model')
        return xgb_model
    elif dataset_name == "adni":
        return tf.keras.models.load_model('/work/users/d/d/ddinh/aaco/models/mlp.keras')


        # xgb_model = XGBClassifier(n_estimators=256)
        # xgb_model.load_model('/work/users/d/d/ddinh/aaco/models/adni_different_masking.model')
        # return xgb_model
    else:
        raise ValueError("Unsupported dataset or model")
        

def get_knn(X_train, X_query, masks, num_neighbors, instance_idx=0, exclude_instance=True, classifier=None, num_ts=12, num_modality=4):
    """
    Args:
    X_train: N x d Train Instances
    X_query: 1 x d Query Instances
    masks: d x R binary masks to try
    num_neighbors: Number of neighbors (k)
    """
    # embedding_model = lambda x: classifier.layers[1](classifier.layers[0](x))

    embeddings_train = []
    embeddings_query = []
    X_train_masked = np.copy(X_train) * tf.transpose(masks, (1, 0))
    X_query_masked = np.copy(X_query) * tf.transpose(masks, (1, 0))
    # repeat the query instance to match the size of the train instances
    X_query_masked = np.repeat(X_query_masked, X_train_masked.shape[0], axis=0)
    
    embeddings_query, embeddings_train = classifier.predict([X_query_masked, X_train_masked], verbose=0)
    
    # embeddings_train = np.array(embeddings_train)
    # embeddings_train = np.mean(embeddings_train, axis=0)
    # embeddings_query = np.mean(embeddings_query, axis=0)

    X_train_squared = embeddings_train ** 2
    X_query_squared = embeddings_query ** 2
    X_train_X_query = embeddings_train * embeddings_query
    X_train_squared = np.sum(X_train_squared, axis=-1)
    X_query_squared = np.sum(X_query_squared, axis=-1)
    X_train_X_query = np.sum(X_train_X_query, axis=-1)
    
    dist_squared = X_train_squared - 2.0 * X_train_X_query + X_query_squared
    # print(dist_squared.shape)
    dist_squared = torch.Tensor(dist_squared)
    
    if exclude_instance:
        idx_topk = torch.topk(dist_squared, num_neighbors + 1, dim=0, largest=False)[1]
        return idx_topk[idx_topk != instance_idx][:num_neighbors]
    else:
        return torch.topk(dist_squared, num_neighbors, dim=0, largest=False)[1]


# Helper function to load the mask generator based on the dataset
def load_mask_generator(dataset_name, input_dim):
    if dataset_name in ["cube", "MNIST"]:
        return random_mask_generator(10000, input_dim, 1000)
    elif dataset_name == "grid" or dataset_name == "gas10":
        all_masks = generate_all_masks(input_dim)  # Generate all possible masks for grid and gas10
        return all_mask_generator(all_masks)
    elif dataset_name == "pedestrian": 
        return random_mask_generator(10000, input_dim, 1000)
    elif dataset_name == "adni": 
        return random_mask_generator(10000, input_dim, 3000)
        # return different_masking(2000, [12, 4])
    else:
        raise ValueError("Unsupported dataset for mask generation")
        

def compute_accumulated_loss(y_pred, y_true, loss_function):
    accumulated_loss = torch.zeros(y_pred.size(0))
    
    for inst in range(y_pred.size(0)):  
        instance_loss = 0.0
        count = 0
        for t in range(y_pred.size(1)):  
            if not torch.all(y_true[inst, t] == 0):  
                timestep_loss = loss_function(y_pred[inst, t], y_true[inst, t])
                instance_loss += timestep_loss
                count += 1
                
        instance_loss /= count    
        accumulated_loss[inst] = instance_loss.item()
    
    return accumulated_loss

        
def aaco_rollout(X_train, y_train, X_valid, y_valid, classifier, mask_generator, initial_feature, config):
    # Load parameters from the config
    feature_count = X_train.shape[1]
    acquisition_cost = config['acquisition_cost']
    nearest_neighbors = config['nearest_neighbors']
    hide_val = 10
    num_instances = config['num_instances']  # Number of instances to loop through
    
    # Decide whether to use training or validation data
    if config['train_or_validation'] == "train":
        X = X_train
        y = y_train
        not_i = True  # Ensure instance isn't its own neighbor in KNN
    else:
        X = X_valid
        y = y_valid
        not_i = False  # Allow instance to be its own neighbor in KNN
    
    # Initialize lists to store results
    X_rollout = []
    y_rollout = []
    action_rollout = []
    mask_rollout = []
    
    # Define the loss function
    loss_function = nn.CrossEntropyLoss(reduction='none')

    ##############################################
    ##### AACO Rollout
    ##############################################
    """
    Dzung:
    just loop through all the features, and add the feature if it improves the loss, otherwise continue to the next feature
    this might only work for single modality
    
    TODO: add the option to continue or stop
    maybe using the current loss to decide whether to continue or stop
    """
    num_ts = y_train.shape[1]
    num_modality = int(feature_count / num_ts)

    
    embedding_model = load_model('/work/users/d/d/ddinh/aaco/models/siamese_adni.h5')
    
    for i in range(num_instances):  # Loop through the specified number of instances
        print(f"Instance {i}")
        
        # Initialize the current mask (start with no features)
        mask_curr = torch.zeros((1, feature_count))
        smallests_val = 0
        
        # for j in range(feature_count + 1):
        for j in range(num_ts + 1):
            if (smallests_val == num_ts - 1) or (j == num_ts + 1):
                # No more features to acquire, add prediction action
                action = torch.zeros(1, feature_count + 1)
                action[0, feature_count] = 1  # Action to predict (last column indicates prediction)
                action_rollout.append(action)
                X_rollout.append(X[[i]])
                y_rollout.append(y[[i]])
                mask_rollout.append(mask_curr)
                break
            
            if j == 0:
                # Select the initial feature deterministically
                mask_rollout.append(mask_curr.clone().detach())
                for k in range(num_modality):
                    mask_curr[0, (initial_feature + k * num_ts)] = 1
                
                action = torch.zeros(1, feature_count + 1)
                for k in range(num_modality):
                    action[0, (initial_feature + k * num_ts)] = 1
                X_rollout.append(X[[i]])
                y_rollout.append(y[[initial_feature]])
                action_rollout.append(action)
            else:
                # Get the nearest neighbors based on the observed feature mask
                idx_nn = get_knn(X_train, X[[i]], mask_curr.T, nearest_neighbors, i, not_i, embedding_model).squeeze()
                # print(f"Neighbors gathered for instance {i} at {datetime.datetime.now()}")
                
                # Generate random masks and get the next set of possible masks
                new_masks = mask_generator(mask_curr)
                mask = torch.maximum(new_masks, mask_curr.repeat(new_masks.shape[0], 1))
                mask[0] = mask_curr # Ensure the current mask is included
                
                # remove nan values in test x from mask
                mask_val_zero = X[[i]] == 0
                mask[:,mask_val_zero[0]] = 0
                
                # remove the nan values in y true
                mask_y = np.sum(y[[i]].numpy(), axis=-1) == 0
                mask_y = mask_y.reshape(1, -1).repeat(mask.shape[0], 0)
                # print(mask_y.shape)
                for m in range(num_modality):
                    mask[:, m * num_ts: (m + 1) * num_ts][mask_y] = 0
                
                # replace the generated mask with the current mask                 
                for k in range(num_modality):  
                    current_mask = np.copy(mask_curr[:,(k * num_ts): (k + 1) * num_ts])
                    current_mask = torch.Tensor(current_mask)
                    mask[:, (k * num_ts): (k * num_ts) + int(smallests_val) + 1] = current_mask[:,:int(smallests_val) + 1]
                

                # Get only unique masks
                mask = mask.unique(dim=0)
                n_masks_updated = mask.shape[0]
                
                if n_masks_updated == 1:
                    # no new features to acquire
                    action = torch.zeros(1, feature_count + 1)
                    action[0, feature_count] = 1  # Action to predict (last column indicates prediction)
                    action_rollout.append(action)
                    X_rollout.append(X[[i]])
                    y_rollout.append(y[[i]])
                    mask_rollout.append(mask_curr)
                    break
                
                # Predictions based on the classifier
                x_rep = X_train[idx_nn].repeat(n_masks_updated, 1)
                mask_zero = x_rep == 0
                
                mask_rep = torch.repeat_interleave(mask, nearest_neighbors, 0)
                mask_rep[mask_zero] = 0 # Dzung: mask nan values from mask
                
                
                y_rep = y_train[idx_nn].repeat(n_masks_updated, 1,1).float() # n, 12, 3
                y_rep_nan = torch.isnan(y_rep)
                y_rep[y_rep_nan] = 0
                y_rep = y_rep.numpy()
                
                
                mask_zero_y = np.sum(y_rep, axis=-1) == 0
                for m in range(num_modality):
                    mask_rep[:, m * num_ts: (m + 1) * num_ts][mask_zero_y] = 0
                    x_rep[:, m * num_ts: (m + 1) * num_ts][mask_zero_y] = 0
                # mask_rep[mask_zero_y] = 0
                # x_rep[mask_zero_y] = 0
                
                idx_nn_rep = idx_nn.repeat(n_masks_updated)
                
                # y_pred = classifier(torch.cat([torch.mul(x_rep, mask_rep) - (1 - mask_rep) * hide_val, mask_rep], -1), idx_nn)
                
                y_pred = np.zeros(y_rep.shape)
                for ts in range(num_ts):
                    x_input = np.zeros(x_rep.shape)
                    mask_input = np.zeros(mask_rep.shape)
                    
                    for k in range(num_modality):
                        x_input[:,k * num_ts: k * num_ts + ts + 1] = np.copy(x_rep[:,k * num_ts: k * num_ts + ts + 1])
                        mask_input[:,k * num_ts: k * num_ts + ts + 1] = np.copy(mask_rep[:,k * num_ts: k * num_ts + ts + 1])
                    
                    x_input = torch.Tensor(x_input)
                    mask_input = torch.Tensor(mask_input)
                    ts_rep = np.repeat(ts, x_input.shape[0]).reshape(-1, 1)
                    pred = classifier.predict(np.concatenate([x_input * mask_input, ts_rep], axis=-1), verbose=0)
                    y_pred[:,ts,:] = pred
                    
                y_pred = torch.Tensor(y_pred)
                y_rep = torch.Tensor(y_rep)
                
                cost = torch.zeros(mask_rep.shape[0])
                for m in range(num_modality):
                    modality_cost = 1 if m in [0, 1] else 0.5
                    cost += mask_rep[:, m * num_ts: (m + 1) * num_ts].sum(dim=1) * acquisition_cost * modality_cost
                # import pdb; pdb.set_trace()
                loss = compute_accumulated_loss(y_pred, y_rep, loss_function) + cost 
                loss = torch.stack([loss[i * nearest_neighbors:(i+1) * nearest_neighbors].mean() for i in range(n_masks_updated)])
                
                # Find the best mask (one with the lowest loss)
                loss_argmin = loss.argmin()
                mask_i = mask[loss_argmin]
                mask_diff = mask_i - mask_curr
                # print(mask_diff)
                # Check if no new features are acquired
                if mask_diff.sum().item() == 0:
                    # No more features to acquire, add prediction action
                    action = torch.zeros(1, feature_count + 1)
                    action[0, feature_count] = 1  # Action to predict (last column indicates prediction)
                    action_rollout.append(action)
                    X_rollout.append(X[[i]])
                    y_rollout.append(y[[i]])
                    mask_rollout.append(mask_curr)
                    break
                else:
                    non_zero = mask_diff.nonzero()[:, 1] 

                    smallests = []
                    for k in range(num_modality):
                        parts = mask_diff[:, k * num_ts: (k + 1) * num_ts]
                        if len(parts.nonzero()) > 0:
                            smallests.append(parts.nonzero()[0][1].item())
                        else:
                            smallests.append(np.inf)
                    
                    # import pdb; pdb.set_trace()
                    # print(smallests)
                    smallests = np.array(smallests)                    
                    smallests_val = np.min(smallests)
                    action_idx = np.where(smallests == smallests_val)[0] * num_ts + smallests[smallests == smallests_val]
                    # remove action_idx corresponding to nan features
                    action_non_nan = X[[i]][0, action_idx] != 0
                    action_idx = action_idx[action_non_nan.numpy()]
                    
                    print(smallests, action_idx)
                    # action_idx = np.argmin(smallests) * num_ts + smallests[np.argmin(smallests)]
                    # what if we have multiple actions? 
                    
                    X_rollout.append(X[[i]])
                    y_rollout.append(y[[i]])
                    mask_rollout.append(mask_curr.clone().detach())
                    action = torch.zeros(1, feature_count + 1)
                    action[0, action_idx] = 1
                    action_rollout.append(action)
                    
                    # Update the current mask
                    mask_curr[:, action_idx] = 1
                    

        results_dir = './results/'
        os.makedirs(results_dir, exist_ok=True)
        
        data = {
            'X': torch.cat(X_rollout), 
            'mask': torch.cat(mask_rollout), 
            'Action': torch.cat(action_rollout), 
            'y': torch.cat(y_rollout)
        }
        
        file_name = f"{results_dir}dataset_{config['dataset']}_siamese_{config['acquisition_cost']}.pt"
        torch.save(data, file_name)
    # Save the results
    results_dir = './results/'
    os.makedirs(results_dir, exist_ok=True)
    
    data = {
        'X': torch.cat(X_rollout), 
        'mask': torch.cat(mask_rollout), 
        'Action': torch.cat(action_rollout), 
        'y': torch.cat(y_rollout)
    }
    
    file_name = f"{results_dir}dataset_{config['dataset']}_siamese_{config['acquisition_cost']}.pt"
    torch.save(data, file_name)
    print(f"Results saved to {file_name}")

        
if __name__ == "__main__":
    # Load configuration file
    config = load_config("config.yaml")

    # Load data based on dataset name in config
    X_train, y_train, X_valid, y_valid, initial_feature, feature_count = load_data(config['dataset'])

    # Extract other parameters from configuration
    nearest_neighbors = config['nearest_neighbors']
    acquisition_cost = config['acquisition_cost']
    train_or_validation = config['train_or_validation']
    dataset_name = config['dataset']
    
    print("Data loaded successfully")
    print("Timestamp:", datetime.datetime.now())
    
    # Load the classifier based on the dataset and outcome model
    classifier = load_classifier(dataset_name, X_train, y_train, input_dim=X_train.shape[1])

    print("Classifier loaded")
    print("Timestamp:", datetime.datetime.now())
    
    # Load the appropriate mask generator based on dataset
    mask_generator = load_mask_generator(dataset_name, feature_count)

    print("Masks generated")
    print("Timestamp:", datetime.datetime.now())
    
    aaco_rollout(X_train, y_train, X_valid, y_valid, classifier, mask_generator, initial_feature, config)



