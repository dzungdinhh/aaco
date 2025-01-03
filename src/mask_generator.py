import numpy as np
import torch
from itertools import chain, combinations


def powerset(iterable):
    """
    Generate all possible subsets (powerset) of the iterable, excluding the empty set.
    powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    s = list(iterable)
    return [list(x) for x in list(chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1)))]

def generate_all_masks(input_dim):
    """
    Generate all possible masks for a given feature dimension.
    """
    subsets = powerset(range(input_dim))
    all_masks = np.zeros((len(subsets), input_dim))

    for i in range(len(subsets)):
        all_masks[i, subsets[i]] = 1
    
    return all_masks

class all_mask_generator():
    def __init__(self, all_masks):
        self.all_masks = torch.from_numpy(all_masks)
        
    def __call__(self, mask_curr):
        return self.all_masks

    
def generate_ball(N, d1, d2):
    Ball = np.concatenate(
  [np.sum(np.random.permutation(np.eye(d1))[:, :np.random.randint(d2)], 1, keepdims=True) for _ in range(N)],
  1)
    return Ball


class random_mask_generator():
    def __init__(self, num_samples, feature_dim, num_generated_masks):
        self.num_samples = num_samples
        self.feature_dim = feature_dim
        self.num_generated_masks = num_generated_masks
        
    def __call__(self, mask_curr):
        ball = generate_ball(self.num_generated_masks, self.feature_dim, self.feature_dim)
        return torch.tensor(ball[:, np.random.permutation(self.num_generated_masks)[:self.num_generated_masks]], dtype=torch.float32).T
    
class different_masking():
    def __init__(self, num_generated_masks, dims):
        self.num_generated_masks = num_generated_masks
        self.dims = dims
        
    def __call__(self, temp, pm=0.2, pd_m=0.4):
        n, T, M = self.num_generated_masks, self.dims[0], self.dims[1]
        mask_m1 = np.random.binomial(1, pm, size=(n, T, M))
        
        t_max = np.random.randint(0, T, size=n)
        mask_m2 = np.ones((n, T, M))
        for i in range(n):
            mask_m2[i, t_max[i]:] = 0
        
        modality_drop = np.random.binomial(1, pd_m, size=(n, M))
        mask_m3 = np.ones((n, T, M))
        for m in range(M):
            mask_m3[:, :, m] *= modality_drop[:, m].reshape(-1, 1)

        final_mask = mask_m1 * mask_m2 * mask_m3

        return torch.Tensor(final_mask.reshape(n, -1))