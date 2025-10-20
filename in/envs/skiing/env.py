from typing import Sequence
import torch
from nudge.env import NudgeBaseEnv
from ocatari.core import OCAtari
import numpy as np



class NudgeEnv(NudgeBaseEnv):
    name = "skiing"
    pred2action = {
        'noop': 0,
        'right': 1,
        'left': 2,
        'down': 3,
    }
    pred_names: Sequence

    def __init__(self, mode: str, render_mode="rgb_array", render_oc_overlay=False):
        super().__init__(mode)
        self.env = OCAtari(env_name="ALE/Skiing-v5", mode="ram",
                           render_mode=render_mode, render_oc_overlay=render_oc_overlay)


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
        num_of_feature = 6
        num_of_object = 14
        logic_state = np.zeros((num_of_object, num_of_feature))

        for i, entity in enumerate(raw_state):
            if entity.category == "Player" and i == 0:
                logic_state[0][0] = 1
                logic_state[0][-2:] = entity.xy
            elif entity.category == 'Tree':
                logic_state[i - 1][1] = 1
                logic_state[i - 1][-2:] = entity.xy
            elif entity.category == 'Mogul':
                logic_state[i - 1][1] = 1
                logic_state[i - 1][-2:] = entity.xy
            elif entity.category == 'Flag':
                logic_state[i - 1][1] = 1
                logic_state[i - 1][-2:] = entity.xy
            logic_state[i][-4:] = np.array(entity.h_coords).flatten()
        return torch.tensor(logic_state)

    def extract_neural_state(self, raw_state):
        # neural_state = []
        # for i, inst in enumerate(raw_state):
        #     if inst.category == "Chicken" and i == 1:
        #         neural_state.append([1, 0, 0, 0] + list(inst.xy))
        #     elif inst.category == "Tree":
        #         neural_state.append([0, 1, 0, 0] + list(inst.xy))
        #     elif inst.category == "Mogul":
        #         neural_state.append([0, 0, 1, 0] + list(inst.xy))
        #     elif inst.category == "Flag":
        #         neural_state.append([0, 0 , 0, 1] + list(inst.xy))
        #
        # return np.array(neural_state).reshape(-1)
        return torch.flatten(self.extract_logic_state(raw_state).float()).unsqueeze(0)

    def close(self):
        self.env.close()
