import random
import numpy as np
import torch
import yaml


def save_hyperparams(signature, local_scope, save_path, print_summary: bool = False):
    hyperparams = {}
    for param in signature.parameters:
        hyperparams[param] = local_scope[param]
    with open(save_path, 'w') as f:
        yaml.dump(hyperparams, f)
    if print_summary:
        print("Hyperparameter Summary:")
        print(open(save_path).read())


def make_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def add_noise(obj, index_obj, num_of_objs):
    mean = torch.tensor(0.2)
    std = torch.tensor(0.05)
    noise = torch.abs(torch.normal(mean=mean, std=std)).item()
    rand_noises = torch.randint(1, 5, (num_of_objs - 1,)).tolist()
    rand_noises = [i * noise / sum(rand_noises) for i in rand_noises]
    rand_noises.insert(index_obj, 1 - noise)

    for i, noise in enumerate(rand_noises):
        obj[i] = rand_noises[i]
    return obj


def simulate_prob(extracted_states, num_of_objs, key_picked):
    for i, obj in enumerate(extracted_states):
        obj = add_noise(obj, i, num_of_objs)
        extracted_states[i] = obj
    if key_picked:
        extracted_states[:, 1] = 0
    return extracted_states
