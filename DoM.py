import torch
import numpy as np
from torch import nn

from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

'''
Paper Reproduction of 

Refusal in Language Models Is Mediated by a Single Direction (2406.11717v3)

Youran Wang
'''

'''
TODO: Temp edition
'''
class ActivationExtractor:
    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.activations = {}
        self.handles = []
        for layer_id in layers:
            # Transformer shape (Llama/Qwen)
            layer = dict(model.named_modules())[f"model.layers.{layer_id}"]
            handle = layer.register_forward_hook(self._get_hook(layer_id))
            self.handles.append(handle)

    def _get_hook(self, layer_id):
        def hook(module, input, output):
            # output (batch, seq_len, hidden_dim)
        
            if isinstance(output, tuple):
                self.activations[layer_id] = output[0].detach()
            else:
                self.activations[layer_id] = output.detach()
        return hook

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()

def get_mean_activation(extractor, token_range):
    """
    token_range: (start, end) 索引
    """
    results = {}
    st, ed = token_range
    for layer_id, act in extractor.activations.items():
        # act shape: (batch, seq_len, dim)
        mean_act = act[:, st:ed, :].mean(dim=1) 
        results[layer_id] = mean_act.cpu().numpy()
    return results

'''
DoM method
'''
def compute_dom_vectors(pos_activations, neg_activations):
    """
    activations format: {layer_id: [num_samples, hidden_dim]}
    """
    dom_directions = {}
    for layer_id in pos_activations.keys():
        mu_pos = np.mean(pos_activations[layer_id], axis=0)
        mu_neg = np.mean(neg_activations[layer_id], axis=0)
        direction = mu_pos - mu_neg
        # normalization
        dom_directions[layer_id] = direction / np.linalg.norm(direction)
    return dom_directions

def linear_cka(X, Y):
    """
    X: [num_features_A, dim_A]
    Y: [num_features_B, dim_B]
    """
    def center(K):
        n = K.shape[0]
        unit = np.ones([n, n]) / n
        return K - unit @ K - K @ unit + unit @ K @ unit

    dot_product = X @ Y.T
    norm_x = np.linalg.norm(X @ X.T)
    norm_y = np.linalg.norm(Y @ Y.T)
    
    return np.linalg.norm(center(X @ X.T) @ center(Y @ Y.T)) / (norm_x * norm_y)

'''
RSA method
'''
def compute_rsa(matrix_A, matrix_B):
    """
    matrix_A: [n_samples, dim_A]
    matrix_B: [n_samples, dim_B]
    """
    # 
    rdm_A = 1 - cosine_similarity(matrix_A)
    rdm_B = 1 - cosine_similarity(matrix_B)
    
    #  
    triu_indices = np.triu_indices(rdm_A.shape[0], k=1)
    vector_A = rdm_A[triu_indices]
    vector_B = rdm_B[triu_indices]
    
    # 
    corr, _ = spearmanr(vector_A, vector_B)
    return corr


# TODO: build different models
layers_to_check = [10, 20, 30]
extractor_A = ActivationExtractor(model_A, layers_to_check)
extractor_B = ActivationExtractor(model_B, layers_to_check)

# 1.
# for prompt, trace, pos_idx in dataset:
#     ... model forward ...
#     ... get_mean_activation ...

# 2. calculate DoM
dom_A = compute_dom_vectors(pos_acts_A, neg_acts_A) # {layer: vec}
dom_B = compute_dom_vectors(pos_acts_B, neg_acts_B) # {layer: vec}

# 3. cross-model comparison
results_matrix = np.zeros((len(layers_to_check), len(layers_to_check)))

for i, l_a in enumerate(layers_to_check):
    for j, l_b in enumerate(layers_to_check):
        # vector reshape to (1, dim) 
        vec_a = dom_A[l_a].reshape(1, -1)
        vec_b = dom_B[l_b].reshape(1, -1)
        
        # Matrix: CKA/RSA
        # singular vector: Cosine Similarity
        # feature direction: CKA
        results_matrix[i, j] = cosine_similarity(vec_a, vec_b)[0,0]

print("Cross-model geometric similarity matrix:", results_matrix)