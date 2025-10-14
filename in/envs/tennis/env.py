from typing import Sequence
import torch
from nudge.env import NudgeBaseEnv
from ocatari.core import OCAtari
import numpy as np


class NudgeEnv(NudgeBaseEnv):
    name = "tennis"
    pred2action = {
        'noop': 0,
        'fire': 1,
        'up': 2,
        'right': 3,
        'left': 4,
        'down': 5,
        'upright': 6,
        'upleft': 7,
        'downright': 8,
        'downleft': 9,
        'upfire': 10,
        'rightfire': 11,
        'leftfire': 12,
        'downfire': 13,
        'uprightfire': 14,
        'upleftfire': 15,
        'downrightfire': 16,
        'downleftfire': 17,

    }
    pred_names: Sequence

    def __init__(self, mode: str, render_mode="rgb_array", render_oc_overlay=False):
        super().__init__(mode)
        self.env = OCAtari(env_name="ALE/Tennis-v5", mode="ram",
                           render_mode=render_mode, render_oc_overlay=render_oc_overlay)
        print(self.env.objects)

    def reset(self):
        self.env.reset()
        state = self.env.objects
        return self.convert_state(state)

    def step(self, action, is_mapped: bool = False):
        if not is_mapped:
            action = self.map_action(action)
        _, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        state = self.env.objects
        return self.convert_state(state), reward, done

    def extract_logic_state(self, raw_state):
        n_features = 4
        n_objects = 4
        logic_state = np.zeros((n_objects, n_features))
        for i, entity in enumerate(raw_state):
            if entity.category == "Player":
                logic_state[i][0] = 1
            elif entity.category == 'Enemy':
                logic_state[i][1] = 1
            elif entity.category == 'Ball':
                logic_state[i][2] = 1
            elif "BallShadow" in entity.category:
                logic_state[i][3] = 1
            logic_state[i][-4:] = np.array(entity.h_coords).flatten()
        return torch.tensor(logic_state)

    def extract_neural_state(self, raw_state):
        #return torch.Tensor(raw_state).unsqueeze(0)
        return torch.flatten(self.extract_logic_state(raw_state))



    def close(self):
        self.env.close()



