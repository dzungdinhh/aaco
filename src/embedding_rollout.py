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
    X_train_masked = np.copy(X_train) * tf.transpose(masks, (1, 0))
    X_query_masked = np.copy(X_query) * tf.transpose(masks, (1, 0))
    

    distribution_train, _, embeddings_train = classifier.predict(X_train_masked, verbose=0)
    distribution_query, next_fea, embeddings_query = classifier.predict(X_query_masked, verbose=0)
    
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
        # return the top k neighbors \and the distribution
        return idx_topk[idx_topk != instance_idx][:num_neighbors], distribution_train[idx_topk != instance_idx], next_fea
    else:
        idx_topk = torch.topk(dist_squared, num_neighbors, dim=0, largest=False)[1]
        distribution = distribution_train[idx_topk]
        return idx_topk, distribution, next_fea


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

    class EmbeddingModels(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.num_features = 12 * 4
            self.base_model = tf.keras.Sequential([
                tf.keras.layers.InputLayer(input_shape=(self.num_features,)),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
            ])
            self.embedding_head = tf.keras.layers.Dense(self.num_features + 1)
            self.prediction_head = tf.keras.layers.Dense(self.num_features + 1)

        def call(self, x):
            x = self.base_model(x)
            embedding_space = self.embedding_head(x)
            prediction = self.prediction_head(x)
            return embedding_space, prediction, x
        
        def get_config(self):
            config = super().get_config()
            config.update({
                'num_features': self.num_features,
            })
            return config

        @classmethod
        def from_config(cls, config):
            return cls(**config)
        
    num_features = 48  # Replace with your actual number of features
    embedding_model = EmbeddingModels()

    # Build the model by calling it on some input (required before loading weights)
    dummy_input = tf.zeros((1, num_features))
    embedding_model(dummy_input)

    # Load the weights
    embedding_model.load_weights('/work/users/d/d/ddinh/aaco/models/embedding_gt.weights.h5')

    # embedding_model = tf.keras.models.load_model('/work/users/d/d/ddinh/aaco/models/embedding_gt.h5', custom_objects={'EmbeddingModels': EmbeddingModels})
    X_rollout = []
    y_rollout = []
    action_rollout = []
    mask_rollout = []
    
    for i in range(num_instances):  # Loop through the specified number of instances
        print(f"Instance {i}")
        
        # Initialize the current mask (start with no features)
        mask_curr = np.zeros((1, feature_count))
        smallests_val = 0
        flag_ts = 0
        for j in range(num_ts + 1):
            if j == 0:
                # Select the initial feature deterministically
                
                # todo: select 1 feature only
                mask_rollout.append(np.copy(mask_curr))
                for k in range(1):
                    mask_curr[0, (initial_feature + k * num_ts)] = 1
                
                action = torch.zeros(1, feature_count + 1)
                for k in range(1):
                    action[0, (initial_feature + k * num_ts)] = 1
                X_rollout.append(X[[i]])
                y_rollout.append(y[[initial_feature]])
                action_rollout.append(action)
            else: 
                idx_nn, distribution_logits, next_fea_pred = get_knn(X_train, X[[i]], mask_curr.T, 5, i, not_i, embedding_model)
                print("next_fea_pred", F.softmax(torch.tensor(next_fea_pred)[0], dim=0).numpy().argmax(), next_fea_pred)
                idx_nn = idx_nn.squeeze()
                
                # convert to distribution
                # todo: maybe weight based on the distance
                
                distribution_logits = np.mean(distribution_logits, axis=0)
                distribution = F.softmax(torch.tensor(distribution_logits), dim=0).numpy()
                # todo: mask out distribution already in the mask 
                # todo: mask out distribution from previous steps
                distribution[:num_features] = distribution[:num_features] * (1 - mask_curr)
                nan_x_mask = X[i] == 0
                distribution[:num_features][nan_x_mask] = 0
                
                for ts in range(num_ts):
                    distribution[ts * num_modality: ts * num_modality + flag_ts] = 0
                
                next_feature = np.argmax(distribution)
                if (next_feature % num_modality) >= flag_ts:
                    flag_ts = next_feature % num_modality 

                print(distribution)
                print(f"Next feature: {next_feature % num_modality}")
                
                if next_feature == num_features:
                    break
                else: 
                    mask_curr[0, next_feature] = 1
                    action = torch.zeros(1, feature_count + 1)
                    action[0, next_feature] = 1
                    mask_rollout.append(np.copy(mask_curr))
                    X_rollout.append(X[[i]])
                    y_rollout.append(y[[i]])
                    action_rollout.append(action)

        results_dir = './results/'
        os.makedirs(results_dir, exist_ok=True)
        
        data = {
            'X': torch.cat(X_rollout), 
            'mask': torch.cat([torch.tensor(mask) for mask in mask_rollout]), 
            'Action': torch.cat(action_rollout), 
            'y': torch.cat(y_rollout)
        }
        
        file_name = f"{results_dir}dataset_{config['dataset']}_mlp_embedding_rollout_{config['acquisition_cost']}.pt"
        torch.save(data, file_name)
    # Save the results
    results_dir = './results/'
    os.makedirs(results_dir, exist_ok=True)
    
    data = {
        'X': torch.cat(X_rollout), 
        'mask': torch.cat([torch.tensor(mask) for mask in mask_rollout]), 
        'Action': torch.cat(action_rollout), 
        'y': torch.cat(y_rollout)
    }
    
    file_name = f"{results_dir}dataset_{config['dataset']}_mlp_embedding_rollout_{config['acquisition_cost']}.pt"
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



