import math
import random
import numpy as np
import torch
import yaml
from pathlib import Path
import os
import re

from .agents.logic_agent import NsfrActorCritic
from .agents.neural_agent import ActorCritic
from nudge.env import NudgeBaseEnv


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


def load_model(model_dir,
               env_kwargs_override: dict = None,
               device=torch.device('cuda:0')):
    # Determine all relevant paths
    model_dir = Path(model_dir)
    config_path = model_dir / "config.yaml"
    checkpoint_dir = model_dir / "checkpoints"
    most_recent_step = get_most_recent_checkpoint_step(checkpoint_dir)
    checkpoint_path = checkpoint_dir / f"step_{most_recent_step}.pth"

    # Load model's configuration
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    algorithm = config["algorithm"]
    environment = config["environment"]
    env_kwargs = config["env_kwargs"]
    env_kwargs.update(env_kwargs_override)

    # Setup the environment
    env = NudgeBaseEnv.from_name(environment, mode=algorithm, **env_kwargs)

    rules = config["rules"]

    print("Loading...")
    # Initialize the model
    if algorithm == 'ppo':
        model = ActorCritic(env).to(device)
    else:  # algorithm == 'logic'
        model = NsfrActorCritic(env, device=device, rules=rules).to(device)

    # Load the model weights
    with open(checkpoint_path, "rb") as f:
        model.load_state_dict(state_dict=torch.load(f))

    return model


def yellow(text):
    return "\033[93m" + text + "\033[0m"


def exp_decay(episode: int):
    """Reaches 2% after about 850 episodes."""
    return max(math.exp(-episode / 500), 0.02)


def get_most_recent_checkpoint_step(checkpoint_dir):
    checkpoints = os.listdir(checkpoint_dir)
    highest_step = 0
    pattern = re.compile("[0-9]+")
    for i, c in enumerate(checkpoints):
        match = pattern.search(c)
        if match is not None:
            step = int(match.group())
            if step > highest_step:
                highest_step = step
    return highest_step
