from typing import Sequence

from nudge.env import NudgeBaseEnv
from ocatari.core import OCAtari
import numpy as np


class NudgeEnv(NudgeBaseEnv):
    name = "asterix"
    pred2action = {
        'noop': 0,
        'up': 1,
        'right': 2,
        'left': 3,
        'down': 4,
    }
    pred_names: Sequence

    def __init__(self, mode: str, render_mode="rgb_array", render_oc_overlay=False):
        super().__init__(mode)
        self.env = OCAtari(env_name="ALE/Asterix-v5", mode="ram",
                           render_mode=render_mode, render_oc_overlay=render_oc_overlay)

    def reset(self):
        self.env.reset()
        state = self.env.objects
        return self.convert_state(state)

    def step(self, action, is_mapped: bool = False):
        if not is_mapped:
            action = self.map_action(action)
        _, reward, done, _, _ = self.env.step(action)
        state = self.env.objects
        return self.convert_state(state), reward, done

    def extract_logic_state(self, raw_state):
        n_features = 6
        n_objects = 11
        logic_state = np.zeros((n_objects, n_features))

        for i, entity in enumerate(raw_state):
            if entity.category == "Player":
                logic_state[i][0] = 1
                logic_state[i][-2:] = entity.xy
            elif entity.category == 'Enemy':
                logic_state[i][1] = 1
                logic_state[i][-2:] = entity.xy
            elif "Reward" in entity.category:
                logic_state[i][2] = 1
                logic_state[i][-2:] = entity.xy
            else:
                logic_state[i][3] = 1
                logic_state[i][-2:] = entity.xy

        return logic_state

    def extract_neural_state(self, raw_state):
        neural_state = []
        for i, inst in enumerate(raw_state):
            if inst.category == "Player" and i == 0:
                neural_state.append([1, 0, 0, 0] + list(inst.xy))
            elif inst.category == "Enemy":
                neural_state.append([0, 1, 0, 0] + list(inst.xy))
            elif "Reward" in inst.category:
                neural_state.append([0, 0, 1, 0] + list(inst.xy))
            else:
                neural_state.append([0, 0, 0, 1] + list(inst.xy))

        if len(neural_state) < 11:
            neural_state.extend([[0] * 6 for _ in range(11 - len(neural_state))])

        return neural_state

    def close(self):
        self.env.close()
