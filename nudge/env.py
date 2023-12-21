from typing import Dict
from abc import ABC
from nsfr.utils.common import load_module
import torch


class NudgeBaseEnv(ABC):
    pred2action: Dict[str, int]  # predicate name to action index

    def __init__(self, mode: str):
        self.mode = mode  # either 'ppo' or 'logic'

    def reset(self):
        raise NotImplementedError

    def step(self, action) -> ((torch.tensor, torch.tensor), float, bool):
        """Returns (logic_state, neural_state), reward, done"""
        raise NotImplementedError

    def extract_logic_state(self, raw_state) -> torch.tensor:
        """Turns the raw state representation into logic representation."""
        raise NotImplementedError

    def extract_neural_state(self, raw_state) -> torch.tensor:
        """Turns the raw state representation into neural representation."""
        raise NotImplementedError

    def convert_state(self, state) -> (torch.tensor, torch.tensor):
        return self.extract_logic_state(state), self.extract_neural_state(state)

    def map_action(self, model_action) -> int:
        """Converts a model action to the corresponding env action."""
        if self.mode == 'ppo':
            return model_action + 1
        else:  # logic
            pred_names = list(self.pred2action.keys())
            for pred_name in pred_names:
                if pred_name in model_action:
                    return self.pred2action[pred_name]
            raise ValueError(f"Invalid predicate '{model_action}' provided. "
                             f"Must contain any of {pred_names}.")

    @staticmethod
    def from_name(name: str, **kwargs):
        env_path = f"envs/{name}/env.py"
        env_module = load_module(env_path)
        return env_module.NudgeEnv(**kwargs)